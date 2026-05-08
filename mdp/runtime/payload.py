"""Process-boundary serialization for RunPlan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mdp.settings.components import ComponentSpec
from mdp.settings.run_plan import Command, Mode, RunPlan, RunSources, ValidationScope
from mdp.settings.schema import Settings


@dataclass(frozen=True)
class RunPlanPayload:
    """JSON-serializable RunPlan payload for torchrun workers."""

    command: Command
    mode: Mode
    settings: dict[str, Any]
    sources: dict[str, str | None]
    overrides: tuple[str, ...]
    callback_configs: tuple[dict[str, Any], ...]
    validation_scope: ValidationScope
    distributed_intent: bool

    @classmethod
    def from_run_plan(cls, plan: RunPlan) -> "RunPlanPayload":
        return cls(
            command=plan.command,
            mode=plan.mode,
            settings=plan.settings.model_dump(mode="json"),
            sources={
                "recipe_path": str(plan.sources.recipe_path)
                if plan.sources.recipe_path is not None
                else None,
                "config_path": str(plan.sources.config_path)
                if plan.sources.config_path is not None
                else None,
                "artifact_dir": str(plan.sources.artifact_dir)
                if plan.sources.artifact_dir is not None
                else None,
            },
            overrides=plan.overrides,
            callback_configs=tuple(
                config.to_yaml_dict() for config in plan.callback_configs
            ),
            validation_scope=plan.validation_scope,
            distributed_intent=plan.distributed_intent,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "mode": self.mode,
            "settings": self.settings,
            "sources": self.sources,
            "overrides": list(self.overrides),
            "callback_configs": list(self.callback_configs),
            "validation_scope": self.validation_scope,
            "distributed_intent": self.distributed_intent,
        }

    @classmethod
    def from_json_dict(cls, raw: dict[str, Any]) -> "RunPlanPayload":
        return cls(
            command=raw["command"],
            mode=raw["mode"],
            settings=raw["settings"],
            sources=raw.get("sources", {}),
            overrides=tuple(raw.get("overrides", ())),
            callback_configs=tuple(raw.get("callback_configs", ())),
            validation_scope=raw["validation_scope"],
            distributed_intent=bool(raw["distributed_intent"]),
        )

    def to_run_plan(self) -> RunPlan:
        sources = self.sources
        return RunPlan(
            command=self.command,
            mode=self.mode,
            settings=Settings(**self.settings),
            sources=RunSources(
                recipe_path=_optional_path(sources.get("recipe_path")),
                config_path=_optional_path(sources.get("config_path")),
                artifact_dir=_optional_path(sources.get("artifact_dir")),
            ),
            overrides=self.overrides,
            callback_configs=tuple(
                ComponentSpec.from_yaml_dict(config, path=f"callbacks[{index}]")
                for index, config in enumerate(self.callback_configs)
            ),
            validation_scope=self.validation_scope,
            distributed_intent=self.distributed_intent,
        )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None
