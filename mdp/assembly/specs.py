"""Serializable component specs for assembly planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from mdp.settings.components import ComponentSpec as SettingsComponentSpec
from mdp.settings.components import ModelComponentSpec


ModelRoute = Literal["component", "pretrained"]


@dataclass(frozen=True)
class ComponentSpec:
    """Assembly plan component with alias resolution made explicit."""

    component: str | None
    kwargs: dict[str, Any]
    path: str
    resolved_component: str | None = None
    model_route: ModelRoute | None = None
    pretrained: str | None = None

    @classmethod
    def from_component(
        cls,
        spec: SettingsComponentSpec,
        *,
        resolved_component: str | None,
        path: str | None = None,
    ) -> "ComponentSpec":
        return cls(
            component=spec.component,
            kwargs=dict(spec.kwargs),
            path=path or spec.path,
            resolved_component=resolved_component,
        )

    @classmethod
    def from_model(
        cls,
        spec: ModelComponentSpec,
        *,
        resolved_component: str | None,
        path: str | None = None,
    ) -> "ComponentSpec":
        route: ModelRoute = "component" if spec.component is not None else "pretrained"
        return cls(
            component=spec.component,
            kwargs=dict(spec.kwargs),
            path=path or spec.path,
            resolved_component=resolved_component,
            model_route=route,
            pretrained=spec.pretrained,
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | SettingsComponentSpec | ModelComponentSpec | None,
    ) -> "ComponentSpec | None":
        if config is None:
            return None
        if isinstance(config, SettingsComponentSpec):
            return cls.from_component(config, resolved_component=None)
        if isinstance(config, ModelComponentSpec):
            return cls.from_model(config, resolved_component=None)
        data = dict(config)
        component = data.pop("_component_", None)
        pretrained = data.pop("pretrained", None)
        return cls(
            component=component if isinstance(component, str) else None,
            kwargs=data,
            path="$",
            pretrained=pretrained if isinstance(pretrained, str) else None,
            model_route="component" if component is not None else "pretrained",
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.component is not None:
            data["_component_"] = self.component
        if self.pretrained is not None:
            data["pretrained"] = self.pretrained
        data.update(self.kwargs)
        return data


ModelRole = str


@dataclass(frozen=True)
class ModelNode:
    """Model assembly decision for one role."""

    role: ModelRole
    trainable: bool
    model: ComponentSpec
    head: ComponentSpec | None = None
    adapter: ComponentSpec | None = None
    optimizer: ComponentSpec | None = None
    scheduler: ComponentSpec | None = None
    loss: ComponentSpec | None = None


@dataclass(frozen=True)
class DataNode:
    """Data assembly decision for train/validation loaders."""

    dataset: ComponentSpec
    val_dataset: ComponentSpec | None
    collator: ComponentSpec
    sampler: ComponentSpec | None
    dataloader_config: dict[str, Any]
    distributed_intent: bool


@dataclass(frozen=True)
class StrategyNode:
    """Distributed strategy config plus the runtime capability boundary."""

    config: dict[str, Any]
    strategy: str | dict[str, Any] | None
    capability_boundary: tuple[str, ...]
    distributed_intent: bool


@dataclass(frozen=True)
class TrainerNode:
    """Trainer-side config ownership that is not a concrete trainer instance."""

    kind: Literal["sft", "rl"]
    training_config: dict[str, Any]
    algorithm: ComponentSpec | None = None
    generation_config: dict[str, Any] | None = None
    evaluation_config: dict[str, Any] | None = None
    monitoring_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class CallbackNode:
    """Callback config passed through from RunPlan."""

    config: ComponentSpec
