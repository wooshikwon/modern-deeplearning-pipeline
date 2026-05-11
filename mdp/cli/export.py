"""mdp export -- adapter/checkpoint에서 서빙 가능한 전체 모델을 내보낸다."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode, resolve_model_source

logger = logging.getLogger(__name__)


def run_export(
    run_id: str | None = None,
    checkpoint: str | None = None,
    output: str = "./exported-model",
) -> None:
    """adapter artifact 또는 checkpoint에서 merge된 전체 모델을 내보낸다.

    --run-id: MLflow run의 model artifact에서 내보내기.
    --checkpoint: 로컬 checkpoint 디렉토리에서 내보내기.
    """
    try:
        source_dir = resolve_model_source(run_id, checkpoint, "export")
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="export", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        from mdp.serving.model_loader import reconstruct_model

        if not run_id and not source_dir.exists():
            raise FileNotFoundError(f"checkpoint 경로를 찾을 수 없습니다: {checkpoint}")

        if not is_json_mode() and run_id:
            typer.echo(f"MLflow run: {run_id}")

        if not is_json_mode():
            typer.echo(f"소스: {source_dir}")

        # 모델 재구성 + merge
        model, settings = reconstruct_model(source_dir, merge=True)
        target = getattr(model, "module", model)

        # 출력 디렉토리 생성
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)

        from mdp.artifacts.serving import ServingArtifactManager

        ServingArtifactManager().write(
            model,
            settings,
            output_dir,
            mode="deployment_export",
            recipe_source_dir=source_dir,
        )

        if not is_json_mode():
            typer.echo(f"내보내기 완료: {output_dir}")
        else:
            from mdp.cli.schemas import ExportResult
            result = ExportResult(
                output_dir=str(output_dir),
                model_class=type(target).__name__,
                merged=True,
            )
            emit_result(build_result(
                command="export", **result.model_dump(exclude_none=True),
            ))

    except typer.Exit:
        raise
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="export", error_type="RuntimeError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("Export 실패 상세")
        raise typer.Exit(code=1)
