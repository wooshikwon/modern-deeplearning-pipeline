"""validation 패키지 — Settings의 비즈니스·호환성 검증."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """검증 결과. errors와 warnings를 분리하여 담는다."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
