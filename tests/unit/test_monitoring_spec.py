"""MonitoringSpec 정식화 테스트 (spec-system-logging-cleanup §U4).

본 테스트는 Recipe.monitoring 필드에 할당되는 ``MonitoringSpec`` 의 시스템 로깅
관련 3 필드(``log_every_n_steps`` / ``memory_history`` / ``verbose``) 가 다음
계약을 지키는지 검증한다:

1. 기본값이 spec §U4 에 명시된 값과 동일하다.
2. ``log_every_n_steps`` 는 ``>= 1`` 범위 제약을 가진다 — pydantic 이 0·음수를
   거부해야 한다.
3. Recipe 에 ``monitoring`` 블록이 없어도 default_factory 가 ``MonitoringSpec()``
   을 생성해 3 필드가 기본값으로 접근 가능해야 한다.
4. MonitoringSpec 은 ``extra="forbid"`` 로 오타 필드(예: ``log_every_n_step``)
   가 들어오면 ValidationError 로 거부한다.
5. 기존 ``enabled`` / ``baseline`` / ``drift`` 필드는 system logging 필드와
   독립이며 그대로 접근 가능해야 한다 (하위 호환).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mdp.settings.schema import MonitoringSpec, Recipe


# ──────────────────────────────────────────────────────────────────────────
# 기본값
# ──────────────────────────────────────────────────────────────────────────


class TestMonitoringSpecDefaults:
    """spec §U4 가 기대하는 3 필드 기본값."""

    def test_defaults_match_spec(self) -> None:
        m = MonitoringSpec()
        assert m.log_every_n_steps == 10
        assert m.memory_history is False
        assert m.verbose is False

    def test_legacy_fields_default(self) -> None:
        """기존 baseline/drift 경로는 그대로 기본값 유지."""
        m = MonitoringSpec()
        assert m.enabled is False
        assert m.baseline == {}
        assert m.drift == {}


# ──────────────────────────────────────────────────────────────────────────
# 타입 · 범위
# ──────────────────────────────────────────────────────────────────────────


class TestMonitoringSpecValidation:
    """``log_every_n_steps >= 1`` 제약 + extra=forbid."""

    def test_log_every_n_steps_positive(self) -> None:
        """5 처럼 양의 정수는 허용."""
        m = MonitoringSpec(log_every_n_steps=5)
        assert m.log_every_n_steps == 5

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_log_every_n_steps_non_positive_rejected(self, bad: int) -> None:
        """0 또는 음수는 pydantic 이 거부해야 한다 — step % 0 은 ZeroDivisionError."""
        with pytest.raises(ValidationError):
            MonitoringSpec(log_every_n_steps=bad)

    def test_extra_field_forbidden(self) -> None:
        """오타 필드(``log_every_n_step``) 를 조용히 받아들이면 운영 실수 원인이
        된다. pydantic extra='forbid' 로 즉시 거부."""
        with pytest.raises(ValidationError):
            # log_every_n_step (단수) — 실제 필드는 steps (복수)
            MonitoringSpec.model_validate({"log_every_n_step": 5})


# ──────────────────────────────────────────────────────────────────────────
# Recipe 통합
# ──────────────────────────────────────────────────────────────────────────


def _minimal_recipe_kwargs(monitoring: MonitoringSpec | dict | None = None) -> dict:
    """Recipe 생성에 필요한 최소 kwargs. monitoring 은 caller 가 지정."""
    kwargs: dict = {
        "name": "test",
        "task": "causal_lm",
        "data": {
            "dataset": {"_component_": "mdp.data.datasets.HuggingFaceDataset",
                        "source": "wikitext", "split": "train"},
            "collator": {"_component_": "mdp.data.collators.CausalLMCollator",
                         "tokenizer": "gpt2"},
        },
        "training": {"epochs": 1},
        "metadata": {"author": "test", "description": "monitoring spec smoke"},
    }
    if monitoring is not None:
        kwargs["monitoring"] = monitoring
    return kwargs


class TestRecipeMonitoringField:
    """Recipe.monitoring 가 default_factory 로 MonitoringSpec 을 만든다."""

    def test_recipe_without_monitoring_defaults_to_spec(self) -> None:
        """Recipe 에 monitoring 을 안 적어도 기본 MonitoringSpec 이 꽂혀 있어야 한다."""
        recipe = Recipe(**_minimal_recipe_kwargs())
        assert isinstance(recipe.monitoring, MonitoringSpec)
        assert recipe.monitoring.log_every_n_steps == 10
        assert recipe.monitoring.memory_history is False
        assert recipe.monitoring.verbose is False

    def test_recipe_with_partial_monitoring_merges_defaults(self) -> None:
        """monitoring 블록이 1 필드만 있어도 나머지는 기본값으로 채워진다."""
        recipe = Recipe(**_minimal_recipe_kwargs(monitoring={"log_every_n_steps": 25}))
        assert recipe.monitoring.log_every_n_steps == 25
        assert recipe.monitoring.memory_history is False
        assert recipe.monitoring.verbose is False

    def test_recipe_with_full_monitoring_override(self) -> None:
        """3 필드 + legacy 필드 전부 override 가능."""
        recipe = Recipe(
            **_minimal_recipe_kwargs(
                monitoring={
                    "log_every_n_steps": 1,
                    "memory_history": True,
                    "verbose": True,
                    "enabled": True,
                }
            )
        )
        assert recipe.monitoring.log_every_n_steps == 1
        assert recipe.monitoring.memory_history is True
        assert recipe.monitoring.verbose is True
        assert recipe.monitoring.enabled is True

    def test_recipe_rejects_invalid_log_every_n_steps(self) -> None:
        """Recipe 경로로도 invalid 값이 거부된다 (nested validation)."""
        with pytest.raises(ValidationError):
            Recipe(**_minimal_recipe_kwargs(monitoring={"log_every_n_steps": 0}))
