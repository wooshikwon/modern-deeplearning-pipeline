"""AssemblyPlan graph for component assembly decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from mdp.assembly.specs import CallbackNode, DataNode, ModelNode, StrategyNode, TrainerNode
from mdp.settings.run_plan import RunPlan


@dataclass(frozen=True)
class AssemblyPlan:
    """Serializable graph describing component assembly without instances."""

    kind: Literal["sft_training", "rl_training"]
    run_plan: RunPlan
    models: tuple[ModelNode, ...]
    data: DataNode
    trainer: TrainerNode
    strategy: StrategyNode | None
    callbacks: tuple[CallbackNode, ...]

    def to_dict(self) -> dict:
        """Return a primitive-friendly representation for debugging/tests."""
        data = asdict(self)
        run_plan = self.run_plan
        data["run_plan"] = {
            "command": run_plan.command,
            "mode": run_plan.mode,
            "sources": {
                "recipe_path": str(run_plan.sources.recipe_path)
                if run_plan.sources.recipe_path
                else None,
                "config_path": str(run_plan.sources.config_path)
                if run_plan.sources.config_path
                else None,
                "artifact_dir": str(run_plan.sources.artifact_dir)
                if run_plan.sources.artifact_dir
                else None,
            },
            "overrides": list(run_plan.overrides),
            "callback_configs": [
                config.to_yaml_dict() for config in run_plan.callback_configs
            ],
            "validation_scope": run_plan.validation_scope,
            "distributed_intent": run_plan.distributed_intent,
            "settings": run_plan.settings.model_dump(mode="json"),
        }
        return data
