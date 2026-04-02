"""mdp rl-train -- RL alignment 학습을 실행한다."""

from __future__ import annotations

import logging

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode

logger = logging.getLogger(__name__)


def run_rl_train(recipe_path: str, config_path: str) -> None:
    """RL Recipe + Config YAML로 alignment 학습을 실행한다."""
    from mdp.factory.factory import Factory
    from mdp.settings.factory import SettingsFactory
    from mdp.training.rl_trainer import RLTrainer

    if not is_json_mode():
        typer.echo(f"Recipe: {recipe_path}")
        typer.echo(f"Config: {config_path}")

    try:
        settings = SettingsFactory().for_training(recipe_path, config_path)
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="rl-train", error_type="ValidationError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    if settings.recipe.algorithm is None:
        msg = "RL 학습에는 recipe에 algorithm 섹션이 필요합니다. SFT는 mdp train을 사용하세요."
        if is_json_mode():
            emit_result(build_error(command="rl-train", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    algo_name = settings.recipe.algorithm.get("_component_", "unknown")
    if not is_json_mode():
        typer.echo(f"알고리즘: {algo_name}")
        typer.echo(f"모델: {list(settings.recipe.models.keys())}")

    try:
        factory = Factory(settings)
        models = factory.create_models()
        dataloaders = factory.create_dataloaders()

        trainer = RLTrainer(
            settings=settings,
            models=models,
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
        )

        if not is_json_mode():
            typer.echo("RL 학습 시작...")

        result = trainer.train()

        if not is_json_mode():
            typer.echo(f"학습 완료. steps={result['total_steps']}, loss={result['metrics']['loss']:.4f}")

        if is_json_mode():
            emit_result(build_result(command="rl-train", **result))

    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="rl-train", error_type="RuntimeError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("RL 학습 실패 상세")
        raise typer.Exit(code=1)
