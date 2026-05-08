"""Runtime-owned training lifecycle helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from mdp.settings.run_plan import RunPlan

if TYPE_CHECKING:
    from mdp.training.callbacks.base import BaseCallback

CallbacksObserver = Callable[[list["BaseCallback"], Any], None]


def run_training(
    run_plan: RunPlan,
    *,
    callbacks_observer: CallbacksObserver | None = None,
) -> dict:
    """Run SFT or RL training from a validated RunPlan."""
    from mdp.runtime.engine import ExecutionEngine
    from mdp.runtime.worker import apply_liger_patches_for_training

    apply_liger_patches_for_training()
    return ExecutionEngine(callbacks_observer=callbacks_observer).run(run_plan)
