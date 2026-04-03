"""mdp train -- 모델 학습을 실행한다."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import typer

from mdp.cli.output import build_error, build_result, emit_result, is_json_mode

logger = logging.getLogger(__name__)


def _detect_gpu_count() -> int:
    """사용 가능한 GPU 수를 반환한다."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _run_single(settings) -> dict:
    """단일 GPU/CPU 학습을 실행한다."""
    from mdp.cli._torchrun_entry import run_training

    return run_training(settings)


def _run_distributed(settings, nproc: int) -> dict:
    """torchrun을 사용하여 분산 학습을 실행한다."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(settings.model_dump(), f, ensure_ascii=False, default=str)
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


def run_train(recipe_path: str, config_path: str) -> None:
    """Recipe + Config YAML을 조립하여 학습을 실행한다."""
    from mdp.cli.schemas import TrainResult
    from mdp.settings.factory import SettingsFactory

    if not is_json_mode():
        typer.echo(f"Recipe: {recipe_path}")
        typer.echo(f"Config: {config_path}")

    try:
        settings = SettingsFactory().for_training(recipe_path, config_path)
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="train",
                error_type="ValidationError",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    nproc = _detect_gpu_count()
    if not is_json_mode():
        typer.echo(f"GPU count: {nproc}")

    try:
        if nproc > 1:
            if not is_json_mode():
                typer.echo(f"분산 학습 시작 (nproc={nproc})...")
            train_result = _run_distributed(settings, nproc)
        else:
            if not is_json_mode():
                typer.echo("단일 학습 시작...")
            train_result = _run_single(settings)

        if not is_json_mode():
            typer.echo("학습 완료.")

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
            )
            emit_result(build_result(
                command="train", **result.model_dump(exclude_none=True),
            ))

    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="train",
                error_type="RuntimeError",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] 학습 실패: {e}", err=True)
        logger.exception("학습 실패 상세")
        raise typer.Exit(code=1)
