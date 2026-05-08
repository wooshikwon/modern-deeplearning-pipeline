"""Training execution engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from mdp.factory.materializer import AssemblyMaterializer
from mdp.factory.planner import AssemblyPlanner
from mdp.settings.plan import SettingsPlan
from mdp.training.callbacks.base import BaseCallback

CallbacksObserver = Callable[[list[BaseCallback], Any], None]


class ExecutionEngine:
    """Own SettingsPlan -> AssemblyPlan -> TrainingBundle -> trainer execution."""

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
        settings_plan: SettingsPlan,
        cb_configs: Sequence[dict[str, Any]] | None = None,
    ) -> dict:
        """Run SFT or RL training from a validated SettingsPlan."""
        plan = self._with_callback_configs(settings_plan, cb_configs)
        assembly_plan = self._assembly_planner.from_settings_plan(plan)
        materializer = self._materializer_cls(assembly_plan)
        callbacks = materializer.materialize_callbacks()

        if self._callbacks_observer is not None:
            self._callbacks_observer(callbacks, plan.settings)

        if assembly_plan.kind == "sft_training":
            from mdp.training.trainer import Trainer

            bundle = materializer.materialize_sft_training_bundle(callbacks=callbacks)
            return Trainer.from_bundle(bundle).train()

        if assembly_plan.kind == "rl_training":
            from mdp.training.rl_trainer import RLTrainer

            bundle = materializer.materialize_rl_training_bundle(callbacks=callbacks)
            return RLTrainer.from_bundle(bundle).train()

        raise ValueError(f"지원하지 않는 AssemblyPlan kind: {assembly_plan.kind!r}")

    @staticmethod
    def _with_callback_configs(
        settings_plan: SettingsPlan,
        cb_configs: Sequence[dict[str, Any]] | None,
    ) -> SettingsPlan:
        if cb_configs is None:
            return settings_plan
        return replace(
            settings_plan,
            callback_configs=tuple(dict(config) for config in cb_configs),
        )
