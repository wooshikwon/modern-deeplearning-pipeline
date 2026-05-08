"""Training execution engine."""

from __future__ import annotations

from typing import Any, Callable

from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.planner import AssemblyPlanner
from mdp.settings.run_plan import RunPlan
from mdp.training.callbacks.base import BaseCallback

CallbacksObserver = Callable[[list[BaseCallback], Any], None]


class ExecutionEngine:
    """Own RunPlan -> AssemblyPlan -> TrainingBundle -> trainer execution."""

    def __init__(
        self,
        *,
        assembly_planner: type[AssemblyPlanner] = AssemblyPlanner,
        materializer_cls: type[AssemblyMaterializer] = AssemblyMaterializer,
        callbacks_observer: CallbacksObserver | None = None,
    ) -> None:
        self._assembly_planner = assembly_planner
        self._materializer_cls = materializer_cls
        self._callbacks_observer = callbacks_observer

    def run(
        self,
        run_plan: RunPlan,
    ) -> dict:
        """Run SFT or RL training from a validated RunPlan."""
        assembly_plan = self._assembly_planner.from_run_plan(run_plan)
        materializer = self._materializer_cls(assembly_plan)
        callbacks = materializer.materialize_callbacks()

        if self._callbacks_observer is not None:
            self._callbacks_observer(callbacks, run_plan.settings)

        if assembly_plan.kind == "sft_training":
            from mdp.training.trainer import Trainer

            bundle = materializer.materialize_sft_training_bundle(callbacks=callbacks)
            return Trainer.from_bundle(bundle).train()

        if assembly_plan.kind == "rl_training":
            from mdp.training.rl_trainer import RLTrainer

            bundle = materializer.materialize_rl_training_bundle(callbacks=callbacks)
            return RLTrainer.from_bundle(bundle).train()

        raise ValueError(f"지원하지 않는 AssemblyPlan kind: {assembly_plan.kind!r}")
