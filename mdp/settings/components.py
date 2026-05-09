"""Typed open component envelopes for settings schema parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic_core import core_schema


ConfigValue = Any
_COMPONENT_KEY = "_component_"
_CONFLICT_PAIRS = (("target", "target_modules"), ("save", "modules_to_save"))


def _ensure_mapping(value: Any, *, kind: str) -> dict[str, Any]:
    if isinstance(value, (ComponentSpec, ModelComponentSpec, RoleModelSpec)):
        return value.to_yaml_dict()
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must be a mapping")
    return dict(value)


def _check_component_name(value: Any, *, required: bool) -> str | None:
    component = value.get(_COMPONENT_KEY)
    if component is None:
        if required:
            raise ValueError(f"{_COMPONENT_KEY} is required")
        return None
    if not isinstance(component, str) or not component:
        raise ValueError(f"{_COMPONENT_KEY} must be a non-empty string")
    return component


def _check_pretrained(value: Any) -> str | None:
    pretrained = value.get("pretrained")
    if pretrained is None:
        return None
    if not isinstance(pretrained, str) or not pretrained:
        raise ValueError("pretrained must be a non-empty string")
    return pretrained


def _check_semantic_conflicts(value: dict[str, Any]) -> None:
    for semantic_key, raw_key in _CONFLICT_PAIRS:
        if semantic_key in value and raw_key in value:
            raise ValueError(
                f"{semantic_key!r} and {raw_key!r} cannot be specified together"
            )


def _serialize_yaml_value(value: Any) -> dict[str, Any]:
    if isinstance(value, (ComponentSpec, ModelComponentSpec, RoleModelSpec)):
        return value.to_yaml_dict()
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"cannot serialize component envelope value: {value!r}")


def component_kwargs(
    spec: "ComponentSpec | ModelComponentSpec | Mapping[str, Any] | None",
) -> Mapping[str, Any]:
    """Return open kwargs from typed component specs or legacy YAML mappings."""
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        try:
            return ComponentSpec.from_yaml_dict(spec).kwargs
        except ValueError:
            try:
                return ModelComponentSpec.from_yaml_dict(spec).kwargs
            except ValueError:
                return {
                    key: value
                    for key, value in spec.items()
                    if key not in {_COMPONENT_KEY, "pretrained"}
                }
    return spec.kwargs


@dataclass(frozen=True)
class ComponentSpec:
    """Open envelope for a non-model ``_component_`` YAML block.

    ``kwargs`` intentionally remains open. This class deliberately does not
    expose dict-style ``get`` or item access, so runtime code has to migrate to
    explicit ``component`` and ``kwargs`` accessors instead of carrying raw dict
    habits forward.
    """

    component: str
    kwargs: dict[str, ConfigValue] = field(default_factory=dict)
    path: str = "$"

    @classmethod
    def from_yaml_dict(cls, value: Any, *, path: str = "$") -> "ComponentSpec":
        data = _ensure_mapping(value, kind=cls.__name__)
        component = _check_component_name(data, required=True)
        if "pretrained" in data:
            raise ValueError("pretrained is only allowed on model components")
        _check_semantic_conflicts(data)
        kwargs = {key: val for key, val in data.items() if key != _COMPONENT_KEY}
        return cls(component=component or "", kwargs=kwargs, path=path)

    @classmethod
    def _validate(cls, value: Any) -> "ComponentSpec":
        if isinstance(value, cls):
            return value
        return cls.from_yaml_dict(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: core_schema.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_yaml_value
            ),
        )

    def to_yaml_dict(self) -> dict[str, Any]:
        return {_COMPONENT_KEY: self.component, **self.kwargs}


@dataclass(frozen=True)
class ModelComponentSpec:
    """Open envelope for ``recipe.model`` and RL role model routes."""

    component: str | None = None
    pretrained: str | None = None
    kwargs: dict[str, ConfigValue] = field(default_factory=dict)
    path: str = "$"

    @classmethod
    def from_yaml_dict(cls, value: Any, *, path: str = "$") -> "ModelComponentSpec":
        data = _ensure_mapping(value, kind=cls.__name__)
        component = _check_component_name(data, required=False)
        pretrained = _check_pretrained(data)
        if component is None and pretrained is None:
            raise ValueError(f"model component requires {_COMPONENT_KEY} or pretrained")
        kwargs = {
            key: val
            for key, val in data.items()
            if key not in {_COMPONENT_KEY, "pretrained"}
        }
        return cls(component=component, pretrained=pretrained, kwargs=kwargs, path=path)

    @classmethod
    def _validate(cls, value: Any) -> "ModelComponentSpec":
        if isinstance(value, cls):
            return value
        return cls.from_yaml_dict(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: core_schema.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_yaml_value
            ),
        )

    def to_yaml_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.component is not None:
            data[_COMPONENT_KEY] = self.component
        if self.pretrained is not None:
            data["pretrained"] = self.pretrained
        data.update(self.kwargs)
        return data


@dataclass(frozen=True)
class RoleModelSpec:
    """RL role model envelope with explicit ownership of nested components."""

    model: ModelComponentSpec
    head: ComponentSpec | None = None
    adapter: ComponentSpec | None = None
    optimizer: ComponentSpec | None = None
    scheduler: ComponentSpec | None = None
    loss: ComponentSpec | None = None
    trainable: bool | None = None
    freeze: bool | None = None
    path: str = "$"
    explicit_model: bool = False

    _OWNED_COMPONENT_KEYS = frozenset(
        {"head", "adapter", "optimizer", "scheduler", "loss"}
    )
    _OWNED_KEYS = _OWNED_COMPONENT_KEYS | {"model", "trainable", "freeze"}

    @classmethod
    def from_yaml_dict(cls, value: Any, *, path: str = "$") -> "RoleModelSpec":
        data = _ensure_mapping(value, kind=cls.__name__)
        if "model" in data:
            unknown_keys = sorted(set(data) - cls._OWNED_KEYS)
            if unknown_keys:
                first = unknown_keys[0]
                raise ValueError(f"unknown role model field at {path}.{first}")
            model = ModelComponentSpec.from_yaml_dict(data["model"], path=f"{path}.model")
            explicit_model = True
        else:
            model_data = {key: val for key, val in data.items() if key not in cls._OWNED_KEYS}
            model = ModelComponentSpec.from_yaml_dict(model_data, path=path)
            explicit_model = False

        owned = {
            key: (
                ComponentSpec.from_yaml_dict(data[key], path=f"{path}.{key}")
                if data.get(key) is not None
                else None
            )
            for key in cls._OWNED_COMPONENT_KEYS
        }
        trainable = data.get("trainable")
        freeze = data.get("freeze")
        if trainable is not None and not isinstance(trainable, bool):
            raise ValueError("trainable must be a boolean")
        if freeze is not None and not isinstance(freeze, bool):
            raise ValueError("freeze must be a boolean")
        return cls(
            model=model,
            trainable=trainable,
            freeze=freeze,
            path=path,
            explicit_model=explicit_model,
            **owned,
        )

    @classmethod
    def _validate(cls, value: Any) -> "RoleModelSpec":
        if isinstance(value, cls):
            return value
        return cls.from_yaml_dict(value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: core_schema.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize_yaml_value
            ),
        )

    def to_yaml_dict(self) -> dict[str, Any]:
        data = (
            {"model": self.model.to_yaml_dict()}
            if self.explicit_model
            else self.model.to_yaml_dict()
        )
        for key in ("head", "adapter", "optimizer", "scheduler", "loss"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value.to_yaml_dict()
        if self.trainable is not None:
            data["trainable"] = self.trainable
        if self.freeze is not None:
            data["freeze"] = self.freeze
        return data


MetricSpec = str | ComponentSpec
