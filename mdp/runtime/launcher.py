"""Parent-process training launch helpers."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mdp.settings.distributed import should_launch_distributed
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaunchPlan:
    """Resolved parent-process launch decision."""

    settings: Settings
    nproc: int
    distributed: bool
    cb_configs: tuple[dict[str, Any], ...] = ()


def detect_gpu_count() -> int:
    """Return the number of CUDA devices visible to the parent process."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def build_launch_plan(
    settings: Settings,
    *,
    nproc: int | None = None,
    cb_configs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> LaunchPlan:
    """Build the single-process vs torchrun launch decision."""
    detected = detect_gpu_count() if nproc is None else nproc
    return LaunchPlan(
        settings=settings,
        nproc=detected,
        distributed=should_launch_distributed(settings, detected),
        cb_configs=tuple(cb_configs or ()),
    )


def run_single(
    settings: Settings,
    cb_configs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict:
    """Run training in the current Python process."""
    from mdp.cli._torchrun_entry import run_training

    return run_training(settings, cb_configs=list(cb_configs) if cb_configs else None)


def run_distributed(
    settings: Settings,
    nproc: int,
    cb_configs: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict:
    """Run training through torchrun worker subprocesses."""
    settings_dict = settings.model_dump()
    if cb_configs:
        settings_dict["__cb_configs"] = list(cb_configs)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(settings_dict, f, ensure_ascii=False, default=str)
        settings_path = f.name

    result_path = str(Path(settings_path).with_suffix("")) + "_result.json"
    entry_script = Path(__file__).resolve().parents[1] / "cli" / "_torchrun_entry.py"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        str(entry_script),
        "--settings-path",
        settings_path,
        "--result-path",
        result_path,
    ]
    logger.info("torchrun command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)

        result_file = Path(result_path)
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        return {}
    finally:
        Path(settings_path).unlink(missing_ok=True)
        Path(result_path).unlink(missing_ok=True)


def launch(plan: LaunchPlan) -> dict:
    """Execute a resolved launch plan."""
    cb_configs = plan.cb_configs or None
    if plan.distributed:
        return run_distributed(plan.settings, plan.nproc, cb_configs=cb_configs)
    return run_single(plan.settings, cb_configs=cb_configs)
