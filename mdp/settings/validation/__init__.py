"""validation 패키지 — Settings의 비즈니스·호환성 검증."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mdp.settings.components import ComponentSpec


@dataclass
class ValidationResult:
    """검증 결과. errors와 warnings를 분리하여 담는다."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def is_qlora(adapter: Any) -> bool:
    """adapter가 QLoRA인지 판별한다."""
    if isinstance(adapter, ComponentSpec):
        return adapter.component in ("QLoRA", "mdp.models.adapters.qlora.apply_qlora")
    if isinstance(adapter, dict):
        component = adapter.get("_component_", "")
        return component in ("QLoRA", "mdp.models.adapters.qlora.apply_qlora")
    return False
