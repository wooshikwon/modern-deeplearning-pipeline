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


def _run_single(settings) -> None:
    """단일 GPU/CPU 학습을 실행한다."""
    from mdp.cli._torchrun_entry import run_training

    return run_training(settings)


def _run_distributed(settings, nproc: int) -> None:
    """torchrun을 사용하여 분산 학습을 실행한다."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(settings.model_dump(), f, ensure_ascii=False, default=str)
        settings_path = f.name

    entry_script = Path(__file__).resolve().parent / "_torchrun_entry.py"

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        str(entry_script),
        "--settings-path", settings_path,
    ]
    logger.info("torchrun command: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=True)
    return result.returncode


def run_train(recipe_path: str, config_path: str) -> None:
    """Recipe + Config YAML을 조립하여 학습을 실행한다.

    1. SettingsFactory로 YAML -> Settings 변환
    2. GPU 수 감지하여 단일/분산 학습 결정
    3. JobManager로 작업 추적
    4. JSON 출력 지원
    """
    from mdp.settings.factory import SettingsFactory
    from mdp.utils.job_manager import JobManager

    typer.echo(f"Recipe: {recipe_path}")
    typer.echo(f"Config: {config_path}")

    try:
        settings = SettingsFactory().for_training(recipe_path, config_path)
    except Exception as e:
        if is_json_mode():
            emit_result(build_error(
                command="train",
                error_type="settings_error",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] Settings 로딩 실패: {e}", err=True)
        raise typer.Exit(code=1)

    nproc = _detect_gpu_count()
    typer.echo(f"GPU count: {nproc}")

    manager = JobManager()
    job_id = manager.create_job(executor="local")

    try:
        if nproc > 1:
            typer.echo(f"분산 학습 시작 (nproc={nproc})...")
            _run_distributed(settings, nproc)
        else:
            typer.echo("단일 학습 시작...")
            _run_single(settings)

        manager.update_status(job_id, "completed")
        typer.echo(f"학습 완료. job_id={job_id}")

        if is_json_mode():
            emit_result(build_result(
                command="train",
                job_id=job_id,
                nproc=nproc,
            ))

    except Exception as e:
        manager.update_status(job_id, "failed", error=str(e))
        if is_json_mode():
            emit_result(build_error(
                command="train",
                error_type="training_error",
                message=str(e),
            ))
            raise typer.Exit(code=1)
        typer.echo(f"[error] 학습 실패: {e}", err=True)
        logger.exception("학습 실패 상세")
        raise typer.Exit(code=1)
    finally:
        manager.close()
