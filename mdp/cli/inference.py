"""mdp inference -- MLflow run_id 기반 배치 추론 + (선택) 평가."""

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
) -> None:
    """배치 추론 + (선택) 평가.

    --run-id 또는 --model-dir 중 하나를 지정.
    """
    from mdp.cli.schemas import InferenceResult
    from mdp.data import _load_source, _rename_columns
    from mdp.data.loader import load_data
    from mdp.data.tokenizer import build_tokenizer
    from mdp.data.transforms import build_transforms
    from mdp.serving.inference import run_batch_inference
    from mdp.serving.model_loader import reconstruct_model
    from torch.utils.data import DataLoader

    try:
        model_path = resolve_model_source(run_id, model_dir, "inference")
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="inference", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    if not is_json_mode():
        if run_id:
            typer.echo(f"MLflow run: {run_id}")
        else:
            typer.echo(f"모델 디렉토리: {model_dir}")
        typer.echo(f"데이터: {data_source}")

    try:
        # 1. 모델 소스 결정 (already resolved above)

        # 2. 모델 재구성 + 가중치 로드 (adapter면 merge)
        model, settings = reconstruct_model(
            model_path, merge=True, device_map=device_map,
        )

        # 3. 데이터 로드 + 필드 검증
        recipe_data = settings.recipe.data
        fields = _resolve_fields(recipe_data.fields, cli_fields)

        test_ds = _load_source(data_source, split="train")
        columns = test_ds.column_names if hasattr(test_ds, "column_names") else []
        _validate_data_interface(fields, columns)
        test_ds = _rename_columns(test_ds, fields)

        # 전처리
        label_strategy = recipe_data.label_strategy
        val_transform = None
        if recipe_data.augmentation:
            val_transform = build_transforms(recipe_data.augmentation.get("val"))
        tokenize_fn = build_tokenizer(recipe_data.tokenizer, label_strategy=label_strategy)
        raw_columns = list(fields.keys()) if fields else None
        test_ds = load_data(test_ds, transform=val_transform, tokenize_fn=tokenize_fn, raw_columns=raw_columns)

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
