"""mdp inference -- 배치 추론 + (선택) 평가."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from mdp.cli.output import (
    build_error,
    build_result,
    emit_result,
    is_json_mode,
    resolve_model_source_plan,
)

logger = logging.getLogger(__name__)


# ── 데이터 인터페이스 검증 ──


def _resolve_fields(
    checkpoint_fields: dict[str, str] | None,
    cli_fields: list[str] | None,
) -> dict[str, str] | None:
    """CLI --fields 오버라이드를 체크포인트 fields에 병합한다."""
    fields = dict(checkpoint_fields) if checkpoint_fields else {}
    if cli_fields:
        for pair in cli_fields:
            if "=" not in pair:
                raise ValueError(f"--fields 형식 오류: '{pair}' (올바른 형식: role=column)")
            role, col = pair.split("=", 1)
            fields[role] = col
    return fields or None


def _validate_data_interface(
    fields: dict[str, str] | None,
    dataset_columns: list[str],
) -> None:
    """fields 매핑과 test 데이터 컬럼을 비교한다."""
    if not fields:
        return
    label_roles = {"label", "target", "token_labels"}
    input_roles = {role for role in fields if role not in label_roles}
    input_columns = {fields[role] for role in input_roles}
    missing = input_columns - set(dataset_columns)
    if missing:
        hint_parts = [f"{role}={fields[role]}" for role in input_roles]
        raise ValueError(
            f"모델이 기대하는 입력 컬럼 {missing}이 데이터에 없습니다.\n"
            f"  체크포인트 fields: {fields}\n"
            f"  데이터 컬럼: {dataset_columns}\n"
            f"  해결: --fields {' '.join(hint_parts)} 으로 매핑을 지정하세요."
        )


def _load_data_columns(data_source: str) -> list[str]:
    """Return raw dataset columns before recipe preprocessing/tokenization."""
    from datasets import load_dataset

    local_path = Path(data_source)
    if local_path.exists():
        if local_path.is_dir():
            return []
        ext = local_path.suffix.lower()
        ext_map = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}
        fmt = ext_map.get(ext)
        if fmt is None:
            return []
        ds = load_dataset(fmt, data_files=str(local_path), split="train")
    else:
        ds = load_dataset(data_source, split="train")
    return list(getattr(ds, "column_names", []) or [])


# ── MLflow artifact 조회 ──


def _resolve_baseline_path(model_path: Path) -> Path | None:
    """체크포인트 또는 artifact 디렉토리에서 baseline 파일을 찾는다."""
    candidates = [
        model_path / "baseline.json",
        model_path.parent / "baseline.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── Metric 생성 ──


def _create_metrics(
    metric_names: list[str] | None,
    recipe_eval: dict | None,
) -> list[Any]:
    """CLI --metrics 또는 recipe.evaluation.metrics에서 metric 인스턴스를 생성한다."""
    from mdp.settings.components import ComponentSpec
    from mdp.settings.resolver import ComponentResolver

    resolver = ComponentResolver()
    raw_list: list[dict[str, Any] | str] = []

    if metric_names:
        raw_list = [{"_component_": name} for name in metric_names]
    elif recipe_eval and recipe_eval.get("metrics"):
        raw_list = recipe_eval["metrics"]

    metrics = []
    for spec in raw_list:
        if isinstance(spec, str):
            spec = {"_component_": spec}
        try:
            metrics.append(
                resolver.resolve(ComponentSpec.from_yaml_dict(spec, path="metrics"))
            )
        except Exception as e:
            logger.warning("Metric 생성 실패: %s — %s", spec, e)
    return metrics


def _build_inference_dataset_spec(
    recipe_dataset: Any,
    *,
    data_source: str,
    cli_fields: list[str] | None,
):
    """Build the artifact-inference dataset override as a typed component spec."""
    from mdp.settings.components import ComponentSpec

    inference_ds_config = recipe_dataset.to_yaml_dict()
    inference_ds_config["source"] = data_source
    inference_ds_config["split"] = "train"
    if cli_fields:
        override_fields = {}
        for pair in cli_fields:
            if "=" in pair:
                role, col = pair.split("=", 1)
                override_fields[role] = col
        inference_ds_config["fields"] = override_fields
    return ComponentSpec.from_yaml_dict(
        inference_ds_config, path="inference.dataset"
    )


# ── Pretrained 전용 데이터 로딩 ──


def _load_pretrained_data(
    data_source: str,
    tokenizer: Any,
    cli_fields: list[str] | None,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[Any, list[dict] | None]:
    """pretrained 모드에서 데이터를 로드하고 토크나이즈하여 DataLoader와 메타데이터를 반환한다.

    --data가 HuggingFace hub 이름이면 load_dataset으로,
    로컬 파일(.jsonl, .csv, .parquet)이면 파일에서 로드한다.

    Returns
    -------
    tuple[DataLoader, list[dict] | None]
        DataLoader와 메타데이터 레코드 리스트. 메타데이터가 없으면 None.
        메타데이터는 원본 컬럼 중 모델 입력이 아닌 컬럼(label, topic 등)을
        샘플별 dict로 추출한 것이다. 콜백의 ``on_batch(metadata=...)`` 로 전달된다.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    local_path = Path(data_source)
    if local_path.exists():
        ext = local_path.suffix.lower()
        ext_map = {".jsonl": "json", ".json": "json", ".csv": "csv", ".parquet": "parquet"}
        fmt = ext_map.get(ext)
        if fmt is None:
            raise ValueError(f"지원하지 않는 파일 형식: {ext} (jsonl, csv, parquet 지원)")
        ds = load_dataset(fmt, data_files=str(local_path), split="train")
    else:
        ds = load_dataset(data_source, split="train")

    # --fields로 텍스트 컬럼 결정
    text_column = "text"
    if cli_fields:
        for pair in cli_fields:
            if "=" in pair:
                role, col = pair.split("=", 1)
                if role == "text":
                    text_column = col
                    break

    if text_column not in ds.column_names:
        # 첫 번째 string 컬럼을 자동 선택
        for col in ds.column_names:
            if ds.features[col].dtype == "string":
                text_column = col
                break
        else:
            raise ValueError(
                f"텍스트 컬럼을 찾을 수 없습니다. 컬럼: {ds.column_names}. "
                "--fields text=컬럼명 으로 지정하세요."
            )

    # tokenize 전에 metadata 컬럼을 결정한다.
    # 모델 입력이 될 수 있는 컬럼은 tokenizer가 생성하는 키들이다.
    # 원본 컬럼 중 text_column을 제외한 나머지가 metadata 후보다.
    original_columns = list(ds.column_names)
    metadata_columns = [c for c in original_columns if c != text_column]

    # metadata가 있으면 tokenize 전에 eager 추출한다.
    # 추론 데이터셋은 학습보다 작으므로 한 번 순회해도 충분하다.
    metadata_records: list[dict] | None = None
    if metadata_columns:
        metadata_records = [
            {col: ds[i][col] for col in metadata_columns}
            for i in range(len(ds))
        ]

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # remove_columns로 원본 컬럼을 제거한다.
    # DataLoader가 string 컬럼을 텐서로 변환하려다 실패하는 문제를 방지한다.
    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")

    return DataLoader(ds, batch_size=batch_size), metadata_records


# ── Public API ──


def _build_pretrained_kwargs(
    dtype: str | None,
    trust_remote_code: bool,
    attn_impl: str | None,
    device_map: str | None,
) -> dict[str, Any]:
    """pretrained 분기에서 from_pretrained에 전달할 kwargs를 구성한다."""
    from mdp.models.pretrained import PretrainedLoadSpec

    spec = PretrainedLoadSpec.from_options(
        "__placeholder__",
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        device_map=device_map,
    )
    return spec.to_loader_kwargs()


def run_inference(
    run_id: str | None,
    model_dir: str | None,
    data_source: str,
    cli_fields: list[str] | None = None,
    metric_names: list[str] | None = None,
    output_format: str = "parquet",
    output_dir: str = "./output",
    device_map: str | None = None,
    overrides: list[str] | None = None,
    pretrained: str | None = None,
    tokenizer_name: str | None = None,
    callbacks_file: str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool = False,
    attn_impl: str | None = None,
    save_output: bool = False,
    batch_size: int = 32,
    max_length: int = 512,
) -> None:
    """배치 추론 + (선택) 평가.

    --run-id, --model-dir, --pretrained 중 하나를 지정.

    DefaultOutputCallback 자동 주입 정책:
    - Recipe 경로 (--run-id, --model-dir): 항상 자동 추가
    - Pretrained + 사용자 콜백 없음: 자동 추가
    - Pretrained + 사용자 콜백 있음: 미추가 (콜백 전용 모드)
    - Pretrained + 사용자 콜백 + --save-output: 추가
    """
    from mdp.serving.inference import run_batch_inference

    try:
        source_plan = resolve_model_source_plan(
            run_id, model_dir, "inference", pretrained=pretrained,
        )
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="inference", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    is_pretrained = source_plan.is_pretrained

    # 콜백 로드 (pretrained/artifact 공통)
    loaded_callbacks: list = []
    if callbacks_file:
        from mdp.settings.resolver import ComponentResolver
        from mdp.training._common import create_callbacks, load_callbacks_from_file

        cb_configs = load_callbacks_from_file(callbacks_file)
        loaded_callbacks = create_callbacks(cb_configs, ComponentResolver())
        if not is_json_mode():
            typer.echo(f"Callbacks: {len(loaded_callbacks)}개 로드 ({callbacks_file})")

    if not is_json_mode():
        if is_pretrained:
            typer.echo(f"Pretrained: {pretrained}")
        elif run_id:
            typer.echo(f"MLflow run: {run_id}")
        else:
            typer.echo(f"모델 디렉토리: {model_dir}")
        typer.echo(f"데이터: {data_source}")

    try:
        if is_pretrained:
            # ── pretrained 분기: Recipe 없이 직접 로드 ──
            from mdp.cli.generate import _resolve_pretrained_tokenizer_name
            from mdp.models.pretrained import PretrainedResolver
            from transformers import AutoTokenizer

            from mdp.models.pretrained import PretrainedLoadSpec

            load_spec = PretrainedLoadSpec.from_options(
                source_plan.uri,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_impl,
                device_map=device_map,
            )
            model = PretrainedResolver.load(source_plan.uri, load_spec=load_spec)

            tok_name = tokenizer_name or _resolve_pretrained_tokenizer_name(source_plan.uri)
            from mdp.serving.model_loader import _resolve_padding_side
            tokenizer = AutoTokenizer.from_pretrained(tok_name, padding_side=_resolve_padding_side(model))
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            test_loader, metadata = _load_pretrained_data(
                data_source, tokenizer, cli_fields,
                batch_size=batch_size, max_length=max_length,
            )

            metrics = _create_metrics(metric_names, None)

            # pretrained 모델의 출력 이름을 기반으로 output path 생성
            model_label = tok_name.replace("/", "_")
            out_path = Path(output_dir) / f"{model_label}_predictions"

            if not is_json_mode():
                typer.echo("배치 추론 시작...")

            # pretrained 모델의 task는 모델 클래스명에서 추출한다.
            # config.architectures[0]로 로드했으므로 클래스명이 task 정보를 내포한다.
            pretrained_task = type(model).__name__

            # DefaultOutputCallback 자동 주입 정책 (pretrained 분기):
            # - 사용자 콜백 없음: 자동 추가
            # - 사용자 콜백 있음: 미추가 (콜백 전용 모드)
            # - 사용자 콜백 + --save-output: 추가
            inference_callbacks = list(loaded_callbacks) if loaded_callbacks else []
            should_inject = not loaded_callbacks or save_output
            if should_inject:
                from mdp.callbacks.inference import DefaultOutputCallback

                inference_callbacks.append(DefaultOutputCallback(
                    output_path=out_path,
                    output_format=output_format,
                    task=pretrained_task,
                ))

            result_path, eval_results = run_batch_inference(
                model=model,
                dataloader=test_loader,
                output_path=out_path,
                output_format=output_format,
                task=pretrained_task,
                metrics=metrics or None,
                callbacks=inference_callbacks or None,
                tokenizer=tokenizer,
                metadata=metadata,
            )

            if not is_json_mode():
                if result_path is not None:
                    typer.echo(f"추론 완료. 결과: {result_path}")
                else:
                    typer.echo("추론 완료. 콜백 전용 모드 (출력 파일 없음)")
                if eval_results:
                    typer.echo("평가 결과:")
                    for name, value in eval_results.items():
                        typer.echo(f"  {name}: {value}")

            if is_json_mode():
                from mdp.cli.schemas import InferenceResult

                result = InferenceResult(
                    output_path=str(result_path) if result_path else None,
                    task=pretrained_task,
                    evaluation_metrics=eval_results or None,
                )
                emit_result(build_result(
                    command="inference", pretrained=source_plan.uri,
                    **result.model_dump(exclude_none=True),
                ))

        else:
            # ── 기존 artifact 분기: Recipe 기반 ──
            from mdp.cli.schemas import InferenceResult
            from mdp.serving.model_loader import reconstruct_model
            from mdp.settings.resolver import ComponentResolver
            from torch.utils.data import DataLoader

            # 1. 모델 재구성 + 가중치 로드 (adapter면 merge)
            model, settings = reconstruct_model(
                source_plan.path, merge=True, device_map=device_map,
                overrides=overrides,
            )

            # 3. 데이터 로드 — inference는 --data CLI 인자로 별도 데이터를 받음
            # Recipe의 DataSpec.dataset 설정에서 source만 교체하여 동일 전처리 적용
            recipe_data = settings.recipe.data
            resolver = ComponentResolver()

            # inference용 dataset: Recipe dataset 설정을 복사하되 source를 CLI 인자로 교체
            inference_ds_spec = _build_inference_dataset_spec(
                recipe_data.dataset,
                data_source=data_source,
                cli_fields=cli_fields,
            )
            inferred_fields = inference_ds_spec.kwargs.get("fields")
            if inferred_fields:
                columns = _load_data_columns(data_source)
                _validate_data_interface(inferred_fields, columns)

            test_ds = resolver.resolve(inference_ds_spec)

            # DataLoader
            dl_config = recipe_data.dataloader.model_dump() if recipe_data.dataloader else {}
            batch_size = dl_config.get("batch_size", 32)
            num_workers = dl_config.get("num_workers", 0)
            test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

            # 4. Metric 생성
            recipe_eval = settings.recipe.evaluation.model_dump() if settings.recipe.evaluation else None
            metrics = _create_metrics(metric_names, recipe_eval)

            # 5. 추론 실행
            out_path = Path(output_dir) / f"{settings.recipe.name}_predictions"
            if not is_json_mode():
                typer.echo("배치 추론 시작...")

            # DefaultOutputCallback 자동 주입 (Recipe 경로: 항상 추가)
            from mdp.callbacks.inference import DefaultOutputCallback

            inference_callbacks = list(loaded_callbacks) if loaded_callbacks else []
            inference_callbacks.append(DefaultOutputCallback(
                output_path=out_path,
                output_format=output_format,
                task=settings.recipe.task,
            ))

            result_path, eval_results = run_batch_inference(
                model=model,
                dataloader=test_loader,
                output_path=out_path,
                output_format=output_format,
                task=settings.recipe.task,
                metrics=metrics or None,
                callbacks=inference_callbacks or None,
            )

            # 6. Drift detection (baseline이 존재하면)
            monitoring_result = None
            monitoring_cfg = getattr(settings.recipe, "monitoring", None)
            if monitoring_cfg and getattr(monitoring_cfg, "enabled", False):
                baseline_path = _resolve_baseline_path(source_plan.path)
                if baseline_path is not None:
                    try:
                        import json as _json
                        from mdp.monitoring.baseline import compute_baseline, compare_baselines

                        stored_baseline = _json.loads(baseline_path.read_text())
                        current = compute_baseline(
                            train_dataloader=test_loader, model=model, config=settings,
                        )
                        monitoring_result = compare_baselines(stored_baseline, current, settings)
                        if not is_json_mode() and monitoring_result.get("drift_detected"):
                            typer.echo(f"[drift] {monitoring_result.get('alerts', [])}")
                    except Exception as e:
                        logger.warning("Drift detection 실패: %s", e)

            if not is_json_mode():
                if result_path is not None:
                    typer.echo(f"추론 완료. 결과: {result_path}")
                else:
                    typer.echo("추론 완료. 콜백 전용 모드 (출력 파일 없음)")
                if eval_results:
                    typer.echo("평가 결과:")
                    for name, value in eval_results.items():
                        typer.echo(f"  {name}: {value}")

            if is_json_mode():
                result = InferenceResult(
                    output_path=str(result_path) if result_path else None,
                    task=settings.recipe.task,
                    monitoring=monitoring_result,
                    evaluation_metrics=eval_results or None,
                )
                emit_result(build_result(
                    command="inference", run_id=run_id,
                    **result.model_dump(exclude_none=True),
                ))

    except typer.Exit:
        raise
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="inference", error_type="RuntimeError", message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("추론 실패 상세")
        raise typer.Exit(code=1)
