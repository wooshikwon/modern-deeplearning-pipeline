"""mdp serve -- 모델 서빙 REST API 서버."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode, resolve_model_source

logger = logging.getLogger(__name__)


def run_serve(
    run_id: str | None = None,
    model_dir: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
    device_map: str | None = None,
    max_memory: str | None = None,
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

    try:
        source_dir = resolve_model_source(run_id, model_dir, "serve")
    except typer.BadParameter as e:
        msg = str(e)
        if is_json_mode():
            emit_result(build_error(command="serve", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    try:
        if not is_json_mode() and run_id:
            typer.echo(f"MLflow run: {run_id}")

        from mdp.serving.server import create_handler, create_app
        from mdp.serving.model_loader import reconstruct_model

        # device_map 결정: CLI > config > None
        serving_config = None
        try:
            from mdp.settings.factory import SettingsFactory
            _settings = SettingsFactory().from_artifact(str(source_dir))
            serving_config = _settings.config.serving
        except Exception:
            pass

        effective_device_map = device_map
        if effective_device_map is None and serving_config is not None:
            effective_device_map = serving_config.device_map

        effective_max_memory = None
        if max_memory is not None:
            import json
            try:
                effective_max_memory = json.loads(max_memory)
            except json.JSONDecodeError as e:
                raise typer.BadParameter(f"--max-memory: 올바른 JSON이 아닙니다: {e}")
        elif serving_config is not None and serving_config.max_memory is not None:
            effective_max_memory = serving_config.max_memory

        # adapter면 merge, full이면 그대로
        model, settings = reconstruct_model(
            source_dir, merge=True,
            device_map=effective_device_map,
            max_memory=effective_max_memory,
        )
        recipe = settings.recipe

        model.eval()
        serving_config = settings.config.serving
        handler = create_handler(model, recipe, source_dir, serving_config=serving_config)
        app = create_app(handler, recipe)

        if not is_json_mode():
            typer.echo(f"서빙 시작: http://{host}:{port}")
            typer.echo(f"  모델: {recipe.name} (task: {recipe.task})")
            if effective_device_map is not None:
                typer.echo(f"  device_map: {effective_device_map}")
            typer.echo(f"  /predict — 추론 엔드포인트")
            typer.echo(f"  /health  — 헬스 체크")

        if is_json_mode():
            from mdp.cli.schemas import ServeResult
            result = ServeResult(run_id=run_id, port=port)
            emit_result(build_result(
                command="serve", **result.model_dump(exclude_none=True),
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


