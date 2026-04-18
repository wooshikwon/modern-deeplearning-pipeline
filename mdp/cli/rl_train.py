"""mdp rl-train -- RL alignment 학습을 실행한다."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode
from mdp.cli.schemas import TrainResult

logger = logging.getLogger(__name__)


def _detect_gpu_count() -> int:
    """사용 가능한 GPU 수를 반환한다."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _run_single(settings, cb_configs: list[dict] | None = None) -> dict:
    """단일 GPU/CPU RL 학습을 실행한다."""
    from mdp.cli._torchrun_entry import run_training

    return run_training(settings, cb_configs=cb_configs)


def _run_distributed(settings, nproc: int, cb_configs: list[dict] | None = None) -> dict:
    """torchrun을 사용하여 분산 RL 학습을 실행한다."""
    settings_dict = settings.model_dump()
    if cb_configs:
        settings_dict["__cb_configs"] = cb_configs

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(settings_dict, f, ensure_ascii=False, default=str)
        settings_path = f.name

    result_path = str(Path(settings_path).with_suffix("")) + "_result.json"
    entry_script = Path(__file__).resolve().parent / "_torchrun_entry.py"

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        str(entry_script),
        "--settings-path", settings_path,
        "--result-path", result_path,
    ]
    logger.info("torchrun command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)

        # rank-0이 저장한 결과를 읽는다
        result_file = Path(result_path)
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    finally:
        Path(settings_path).unlink(missing_ok=True)
        Path(result_path).unlink(missing_ok=True)


def run_rl_train(
    recipe_path: str,
    config_path: str,
    overrides: list[str] | None = None,
    callbacks_file: str | None = None,
) -> None:
    """RL Recipe + Config YAML로 alignment 학습을 실행한다."""
    from mdp.settings.factory import SettingsFactory

    if not is_json_mode():
        typer.echo(f"Recipe: {recipe_path}")
        typer.echo(f"Config: {config_path}")
        if callbacks_file:
            typer.echo(f"Callbacks: {callbacks_file}")
        if overrides:
            typer.echo(f"Overrides: {overrides}")

    try:
        settings = SettingsFactory().for_training(recipe_path, config_path, overrides=overrides)

        cb_configs: list[dict] = []
        if callbacks_file:
            from mdp.training._common import load_callbacks_from_file
            cb_configs = load_callbacks_from_file(callbacks_file)
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="rl-train", error_type="ValidationError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    if settings.recipe.rl is None:
        msg = "RL 학습에는 recipe에 rl 섹션이 필요합니다. SFT는 mdp train을 사용하세요."
        if is_json_mode():
            emit_result(build_error(command="rl-train", error_type="ValidationError", message=msg))
        else:
            typer.echo(f"[error] {msg}", err=True)
        raise typer.Exit(code=1)

    algo_name = settings.recipe.rl.algorithm.get("_component_", "unknown")
    if not is_json_mode():
        typer.echo(f"알고리즘: {algo_name}")
        typer.echo(f"모델: {list(settings.recipe.rl.models.keys())}")

    nproc = _detect_gpu_count()
    if not is_json_mode():
        typer.echo(f"GPU count: {nproc}")

    try:
        if nproc > 1:
            if not is_json_mode():
                typer.echo(f"분산 RL 학습 시작 (nproc={nproc})...")
            train_result = _run_distributed(settings, nproc, cb_configs=cb_configs or None)
        else:
            if not is_json_mode():
                typer.echo("RL 학습 시작...")
            train_result = _run_single(settings, cb_configs=cb_configs or None)

        if not is_json_mode():
            typer.echo(f"학습 완료. steps={train_result.get('total_steps', 0)}, loss={train_result.get('metrics', {}).get('loss', 0):.4f}")

        if is_json_mode():
            result = TrainResult(
                checkpoint_dir=settings.config.storage.checkpoint_dir,
                output_dir=settings.config.storage.output_dir,
                metrics=train_result.get("metrics", {}),
                total_epochs=train_result.get("total_epochs"),
                total_steps=train_result.get("total_steps"),
                stopped_reason=train_result.get("stopped_reason"),
                duration_seconds=train_result.get("training_duration_seconds"),
                monitoring=train_result.get("monitoring"),
                algorithm=train_result.get("algorithm"),
                run_id=train_result.get("run_id"),
                checkpoints_saved=train_result.get("checkpoints_saved"),
            )
            emit_result(build_result(
                command="rl-train", **result.model_dump(exclude_none=True),
            ))

    except Exception as e:
        if is_json_mode():
            emit_result(build_error(command="rl-train", error_type="RuntimeError", message=str(e)))
            raise typer.Exit(code=1)
        typer.echo(f"[error] {e}", err=True)
        logger.exception("RL 학습 실패 상세")
        raise typer.Exit(code=1)
