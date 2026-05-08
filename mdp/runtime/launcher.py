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

from mdp.settings.run_plan import RunPlan
from mdp.runtime.payload import RunPlanPayload

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaunchPlan:
    """Resolved parent-process launch decision."""

    run_plan: RunPlan
    nproc: int
    distributed: bool


def detect_gpu_count() -> int:
    """Return the number of CUDA devices visible to the parent process."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def build_launch_plan(
    run_plan: RunPlan,
    *,
    nproc: int | None = None,
) -> LaunchPlan:
    """Build the single-process vs torchrun launch decision."""
    detected = detect_gpu_count() if nproc is None else nproc
    return LaunchPlan(
        run_plan=run_plan,
        nproc=detected,
        distributed=detected > 1 and run_plan.distributed_intent,
    )


def run_single(
    run_plan: RunPlan,
    *,
    callbacks_observer: Any = None,
) -> dict:
    """Run training in the current Python process."""
    from mdp.runtime.training import run_training

    return run_training(
        run_plan,
        callbacks_observer=callbacks_observer,
    )


def run_distributed(
    run_plan: RunPlan,
    nproc: int,
) -> dict:
    """Run training through torchrun worker subprocesses."""
    payload = RunPlanPayload.from_run_plan(run_plan)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload.to_json_dict(), f, ensure_ascii=False, default=str)
        run_plan_path = f.name

    result_path = str(Path(run_plan_path).with_suffix("")) + "_result.json"
    entry_script = Path(__file__).resolve().parents[1] / "cli" / "_torchrun_entry.py"

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        str(entry_script),
        "--run-plan-path",
        run_plan_path,
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
        Path(run_plan_path).unlink(missing_ok=True)
        Path(result_path).unlink(missing_ok=True)
