"""mdp serve -- MLflow run_id 기반 MDP 네이티브 서빙."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode

logger = logging.getLogger(__name__)


def run_serve(run_id: str, port: int, host: str = "0.0.0.0") -> None:
    """MLflow run의 model artifact를 FastAPI로 서빙한다."""
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "[error] 서빙에 fastapi와 uvicorn이 필요합니다: pip install mdp[serve]",
            err=True,
        )
        raise typer.Exit(code=1)

    import mlflow

    if not is_json_mode():
        typer.echo(f"MLflow run: {run_id}")

    try:
        model_dir = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model",
        ))
    except Exception as e:
        msg = f"MLflow run '{run_id}'에서 model artifact를 찾을 수 없습니다: {e}"
        if is_json_mode():
            emit_result(build_error(command="serve", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        from mdp.serving.server import create_handler, create_app

        handler = create_handler(model_dir)
        import yaml
        recipe_data = yaml.safe_load((model_dir / "recipe.yaml").read_text())

        # recipe 객체를 Settings에서 가져오기 위해 reconstruct
        from mdp.serving.model_loader import reconstruct_model
        _, settings = reconstruct_model(model_dir)
        recipe = settings.recipe

        app = create_app(handler, recipe)

        if not is_json_mode():
            typer.echo(f"서빙 시작: http://{host}:{port}")
            typer.echo(f"  모델: {recipe.name} (task: {recipe.task})")
            typer.echo(f"  /predict — 추론 엔드포인트")
            typer.echo(f"  /health  — 헬스 체크")

        if is_json_mode():
            emit_result(build_result(
                command="serve", status="starting",
                run_id=run_id, port=port,
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
