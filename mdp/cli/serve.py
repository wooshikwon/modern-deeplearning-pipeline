"""mdp serve -- 모델 서빙 REST API 서버."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode

logger = logging.getLogger(__name__)


def run_serve(
    run_id: str | None = None,
    model_dir: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
) -> None:
    """모델을 REST API로 서빙한다.

    --run-id: MLflow run에서 모델 로딩 (adapter면 on-demand merge).
    --model-dir: 로컬 디렉토리에서 직접 로딩 (mdp export 결과).
    """
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "[error] 서빙에 fastapi와 uvicorn이 필요합니다: pip install mdp[serve]",
            err=True,
        )
        raise typer.Exit(code=1)

    if run_id and model_dir:
        msg = "--run-id와 --model-dir는 동시에 지정할 수 없습니다."
        if is_json_mode():
            emit_result(build_error(command="serve", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    if not run_id and not model_dir:
        msg = "--run-id 또는 --model-dir 중 하나를 지정하세요."
        if is_json_mode():
            emit_result(build_error(command="serve", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        # 모델 디렉토리 결정
        if run_id:
            import mlflow

            if not is_json_mode():
                typer.echo(f"MLflow run: {run_id}")
            source_dir = Path(mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="model",
            ))
        else:
            source_dir = Path(model_dir)

        from mdp.serving.server import create_handler, create_app
        from mdp.serving.model_loader import reconstruct_model

        # adapter면 merge, full이면 그대로
        model, settings = reconstruct_model(source_dir, merge=True)
        recipe = settings.recipe

        model.eval()
        handler = create_handler.__wrapped__(model, recipe) if hasattr(create_handler, "__wrapped__") else _build_handler(model, source_dir, recipe)
        app = create_app(handler, recipe)

        if not is_json_mode():
            typer.echo(f"서빙 시작: http://{host}:{port}")
            typer.echo(f"  모델: {recipe.name} (task: {recipe.task})")
            typer.echo(f"  /predict — 추론 엔드포인트")
            typer.echo(f"  /health  — 헬스 체크")

        if is_json_mode():
            emit_result(build_result(
                command="serve", status="starting",
                port=port,
            ))

        uvicorn.run(app, host=host, port=port)

    except typer.Exit:
        raise
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="serve", error_type="RuntimeError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("서빙 실패 상세")
        raise typer.Exit(code=1)


def _build_handler(model, source_dir: Path, recipe):
    """model + recipe에서 handler를 직접 생성한다."""
    from mdp.serving.handlers import StreamingHandler, BatchHandler
    from mdp.serving.server import _load_tokenizer, _load_transform

    tokenizer = _load_tokenizer(source_dir, recipe)
    transform = _load_transform(recipe)

    if recipe.task in ("text_generation", "seq2seq"):
        return StreamingHandler(model, tokenizer, recipe)
    else:
        return BatchHandler(model, tokenizer, transform, recipe)
