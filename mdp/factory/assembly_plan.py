"""AssemblyPlan graph for training factory decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from mdp.factory.specs import CallbackNode, DataNode, ModelNode, StrategyNode, TrainerNode
from mdp.settings.plan import SettingsPlan


@dataclass(frozen=True)
class AssemblyPlan:
    """Serializable graph describing component assembly without instances."""

    kind: Literal["sft_training", "rl_training"]
    settings_plan: SettingsPlan
    models: tuple[ModelNode, ...]
    data: DataNode
    trainer: TrainerNode
    strategy: StrategyNode | None
    callbacks: tuple[CallbackNode, ...]

    def to_dict(self) -> dict:
        """Return a primitive-friendly representation for debugging/tests."""
        data = asdict(self)
        settings_plan = self.settings_plan
        data["settings_plan"] = {
            "command": settings_plan.command,
            "mode": settings_plan.mode,
            "recipe_path": str(settings_plan.recipe_path) if settings_plan.recipe_path else None,
            "config_path": str(settings_plan.config_path) if settings_plan.config_path else None,
            "artifact_dir": str(settings_plan.artifact_dir) if settings_plan.artifact_dir else None,
            "overrides": list(settings_plan.overrides),
            "callback_configs": list(settings_plan.callback_configs),
            "validation_scope": settings_plan.validation_scope,
            "distributed_intent": settings_plan.distributed_intent,
            "settings": settings_plan.settings.model_dump(mode="json"),
        }
        return data
