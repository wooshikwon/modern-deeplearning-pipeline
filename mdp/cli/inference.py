"""mdp inference -- 배치 추론 + (선택) 평가."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode, resolve_model_source

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
            metrics.append(resolver.resolve(spec))
        except Exception as e:
            logger.warning("Metric 생성 실패: %s — %s", spec, e)
    return metrics


# ── Pretrained 전용 데이터 로딩 ──


def _load_pretrained_data(
    data_source: str,
    tokenizer: Any,
    cli_fields: list[str] | None,
    batch_size: int = 32,
) -> Any:
    """pretrained 모드에서 데이터를 로드하고 토크나이즈하여 DataLoader를 반환한다.

    --data가 HuggingFace hub 이름이면 load_dataset으로,
    로컬 파일(.jsonl, .csv, .parquet)이면 파일에서 로드한다.
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

    def tokenize_fn(examples: dict) -> dict:
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")

    return DataLoader(ds, batch_size=batch_size)


# ── Public API ──


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
) -> None:
    """배치 추론 + (선택) 평가.

    --run-id, --model-dir, --pretrained 중 하나를 지정.
    """
    from mdp.serving.inference import run_batch_inference

    try:
        model_path = resolve_model_source(run_id, model_dir, "inference", pretrained=pretrained)
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="inference", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    is_pretrained = model_path is None

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

            model = PretrainedResolver.load(pretrained)

            tok_name = tokenizer_name or _resolve_pretrained_tokenizer_name(pretrained)
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            test_loader = _load_pretrained_data(
                data_source, tokenizer, cli_fields,
            )

            metrics = _create_metrics(metric_names, None)

            # pretrained 모델의 출력 이름을 기반으로 output path 생성
            model_label = tok_name.replace("/", "_")
            out_path = Path(output_dir) / f"{model_label}_predictions"

            if not is_json_mode():
                typer.echo("배치 추론 시작...")

            result_path, eval_results = run_batch_inference(
                model=model,
                dataloader=test_loader,
                output_path=out_path,
                output_format=output_format,
                task="classification",
                metrics=metrics or None,
                callbacks=loaded_callbacks or None,
                tokenizer=tokenizer,
            )

            if not is_json_mode():
                typer.echo(f"추론 완료. 결과: {result_path}")
                if eval_results:
                    typer.echo("평가 결과:")
                    for name, value in eval_results.items():
                        typer.echo(f"  {name}: {value}")

            if is_json_mode():
                from mdp.cli.schemas import InferenceResult

                result = InferenceResult(
                    output_path=str(result_path),
                    task="classification",
                    evaluation_metrics=eval_results or None,
                )
                emit_result(build_result(
                    command="inference", pretrained=pretrained,
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
                model_path, merge=True, device_map=device_map,
                overrides=overrides,
            )

            # 3. 데이터 로드 — inference는 --data CLI 인자로 별도 데이터를 받음
            # Recipe의 DataSpec.dataset 설정에서 source만 교체하여 동일 전처리 적용
            recipe_data = settings.recipe.data
            resolver = ComponentResolver()

            # inference용 dataset: Recipe dataset 설정을 복사하되 source를 CLI 인자로 교체
            inference_ds_config = dict(recipe_data.dataset)
            inference_ds_config["source"] = data_source
            inference_ds_config["split"] = "train"  # inference 데이터는 단일 split
            # CLI fields override 적용
            if cli_fields:
                override_fields = {}
                for pair in cli_fields:
                    if "=" in pair:
                        role, col = pair.split("=", 1)
                        override_fields[role] = col
                inference_ds_config["fields"] = override_fields

            test_ds = resolver.resolve(inference_ds_config)

            # 필드 검증 (가능한 경우)
            columns = test_ds._ds.column_names if hasattr(test_ds, "_ds") and hasattr(test_ds._ds, "column_names") else []
            inferred_fields = inference_ds_config.get("fields")
            if inferred_fields:
                _validate_data_interface(inferred_fields, columns)

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

            result_path, eval_results = run_batch_inference(
                model=model,
                dataloader=test_loader,
                output_path=out_path,
                output_format=output_format,
                task=settings.recipe.task,
                metrics=metrics or None,
                callbacks=loaded_callbacks or None,
            )

            # 6. Drift detection (baseline이 존재하면)
            monitoring_result = None
            monitoring_cfg = getattr(settings.recipe, "monitoring", None)
            if monitoring_cfg and getattr(monitoring_cfg, "enabled", False):
                baseline_path = _resolve_baseline_path(model_path)
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
                typer.echo(f"추론 완료. 결과: {result_path}")
                if eval_results:
                    typer.echo("평가 결과:")
                    for name, value in eval_results.items():
                        typer.echo(f"  {name}: {value}")

            if is_json_mode():
                result = InferenceResult(
                    output_path=str(result_path),
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
