"""mdp train -- 모델 학습을 실행한다."""

from __future__ import annotations

import logging

import typer

logger = logging.getLogger(__name__)


def run_train(recipe_path: str, config_path: str) -> None:
    """Recipe + Config YAML을 조립하여 학습을 실행한다.

    1. SettingsFactory로 YAML → Settings 변환
    2. Config.compute.target으로 Executor 결정
    3. MLflow run 내에서 executor.run(settings) 호출
    """
    from mdp.compute import get_executor
    from mdp.settings.factory import SettingsFactory

    typer.echo(f"Recipe: {recipe_path}")
    typer.echo(f"Config: {config_path}")

    try:
        settings = SettingsFactory().for_training(recipe_path, config_path)
    except Exception as e:
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    target = settings.config.compute.target
    try:
        executor = get_executor(target)
    except ValueError as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Executor: {target}")

    # MLflow tracking
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.config.mlflow.tracking_uri)
        mlflow.set_experiment(settings.config.mlflow.experiment_name)

        with mlflow.start_run(run_name=settings.recipe.name):
            typer.echo("학습 시작...")
            job_id = executor.run(settings)
            typer.echo(f"학습 완료. job_id={job_id}")
    except ImportError:
        logger.warning("MLflow가 설치되지 않아 tracking 없이 실행합니다.")
        typer.echo("학습 시작 (MLflow 없음)...")
        job_id = executor.run(settings)
        typer.echo(f"학습 완료. job_id={job_id}")
    except Exception as e:
        typer.echo(f"[error] 학습 실패: {e}", err=True)
        logger.exception("학습 실패 상세")
        raise typer.Exit(code=1)
