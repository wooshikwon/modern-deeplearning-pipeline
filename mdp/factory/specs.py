"""Serializable component specs for factory assembly planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class ComponentSpec:
    """Thin wrapper around a component config mapping.

    The wrapper intentionally does not validate or normalize keys. Some model
    sources are ``pretrained``-only, and future ``_component_`` envelopes must
    be able to carry arbitrary provider-specific fields.
    """

    config: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", dict(self.config))

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ComponentSpec | None":
        if config is None:
            return None
        return cls(dict(config))

    @property
    def component(self) -> str | None:
        value = self.config.get("_component_")
        return value if isinstance(value, str) else None

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.config)


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
    """Callback config passed through from SettingsPlan."""

    config: ComponentSpec
