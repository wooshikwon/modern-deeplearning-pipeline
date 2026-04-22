"""BaseAlgorithm.needs_logits 계약 검증 테스트.

U1 — BaseAlgorithm에 ``needs_logits: ClassVar[bool] = True`` 필드가 선언되어
있으며, 기존 서브클래스(DPO/GRPO/PPO)가 이 기본값을 상속해 byte-identical
동작을 유지하는지 확인한다.
"""

from __future__ import annotations

import typing

from mdp.training.losses.base import BaseAlgorithm
from mdp.training.losses.rl import DPOLoss, GRPOLoss, PPOLoss


class TestBaseAlgorithmNeedsLogits:
    """BaseAlgorithm의 needs_logits ClassVar 기본값·타입 계약."""

    def test_base_default_is_true(self) -> None:
        """BaseAlgorithm.needs_logits 기본값은 True."""
        assert BaseAlgorithm.needs_logits is True

    def test_base_instance_inherits_default(self) -> None:
        """BaseAlgorithm 인스턴스에서도 needs_logits는 True."""
        algo = BaseAlgorithm()
        assert algo.needs_logits is True

    def test_needs_logits_is_classvar_typed(self) -> None:
        """needs_logits는 ``ClassVar[bool]`` 로 타이핑되어 있다."""
        hints = typing.get_type_hints(BaseAlgorithm, include_extras=True)
        # Python은 ClassVar를 get_type_hints에서 ``ClassVar[bool]`` 형태로 유지한다.
        raw_annotation = BaseAlgorithm.__annotations__.get("needs_logits")
        assert raw_annotation is not None, "needs_logits annotation이 없습니다."
        # annotation 문자열 표현에 ClassVar와 bool이 포함되어야 한다.
        annotation_repr = repr(raw_annotation)
        assert "ClassVar" in annotation_repr, (
            f"needs_logits는 ClassVar로 선언되어야 합니다. 실제: {annotation_repr}"
        )
        assert "bool" in annotation_repr, (
            f"needs_logits는 bool 타입이어야 합니다. 실제: {annotation_repr}"
        )
        # include_extras로 get_type_hints를 호출하면 ClassVar도 포함된다.
        resolved = hints.get("needs_logits")
        assert resolved is not None, "get_type_hints가 needs_logits를 반환하지 않았습니다."


class TestExistingAlgorithmsInheritDefault:
    """기존 DPO/GRPO/PPO는 needs_logits를 선언하지 않고 기본값 True를 상속한다."""

    def test_dpo_inherits_needs_logits_true(self) -> None:
        """DPOLoss는 BaseAlgorithm을 상속하며 needs_logits=True."""
        assert issubclass(DPOLoss, BaseAlgorithm)
        assert DPOLoss.needs_logits is True
        assert getattr(DPOLoss(), "needs_logits", None) is True

    def test_grpo_inherits_needs_logits_true(self) -> None:
        """GRPOLoss는 BaseAlgorithm을 상속하며 needs_logits=True."""
        assert issubclass(GRPOLoss, BaseAlgorithm)
        assert GRPOLoss.needs_logits is True
        assert getattr(GRPOLoss(), "needs_logits", None) is True

    def test_ppo_inherits_needs_logits_true(self) -> None:
        """PPOLoss는 BaseAlgorithm을 상속하며 needs_logits=True."""
        assert issubclass(PPOLoss, BaseAlgorithm)
        assert PPOLoss.needs_logits is True
        assert getattr(PPOLoss(), "needs_logits", None) is True

    def test_subclasses_do_not_redeclare_needs_logits(self) -> None:
        """서브클래스는 needs_logits를 자체 선언하지 않고 기본값을 상속한다.

        delegation 원칙: "기존 알고리즘 서브클래스에 needs_logits=True 명시 금지 —
        기본값 True로 상속받으면 충분".
        """
        for cls in (DPOLoss, GRPOLoss, PPOLoss):
            assert "needs_logits" not in cls.__dict__, (
                f"{cls.__name__}이 needs_logits를 자체 선언했습니다. "
                "기본값 True를 BaseAlgorithm에서 상속하세요."
            )

    def test_getattr_default_pattern_returns_true(self) -> None:
        """``getattr(algorithm, "needs_logits", True)`` 패턴이 True를 반환한다.

        Trainer (U2 소비자)의 호출 패턴과 동일.
        """
        for cls in (DPOLoss, GRPOLoss, PPOLoss):
            instance = cls()
            assert getattr(instance, "needs_logits", True) is True


class TestNewSubclassCanOverride:
    """새 알고리즘은 ``needs_logits = False``로 override 가능하다 (U3 consumer 패턴)."""

    def test_subclass_can_override_to_false(self) -> None:
        """``needs_logits: ClassVar[bool] = False``를 override하면 False."""
        from typing import ClassVar

        class FusedLossAlgorithm(BaseAlgorithm):
            needs_logits: ClassVar[bool] = False

        assert FusedLossAlgorithm.needs_logits is False
        assert getattr(FusedLossAlgorithm(), "needs_logits", True) is False
        # base는 여전히 True
        assert BaseAlgorithm.needs_logits is True


class TestBaseAlgorithmAdditionalFlags:
    """BaseAlgorithm은 needs_hidden_states / needs_generation 도 ClassVar로 선언한다 (2-1).

    계약 flag 3종을 단일 진실 원천(BaseAlgorithm)에 통합해 타입 체커·IDE가
    추적 가능하도록 한다. 기본값은 모두 기존 duck typing default와 동일하므로
    byte-identical 동작 유지.
    """

    def test_base_needs_hidden_states_default_is_false(self) -> None:
        """BaseAlgorithm.needs_hidden_states 기본값은 False."""
        assert BaseAlgorithm.needs_hidden_states is False
        assert BaseAlgorithm().needs_hidden_states is False

    def test_base_needs_generation_default_is_false(self) -> None:
        """BaseAlgorithm.needs_generation 기본값은 False."""
        assert BaseAlgorithm.needs_generation is False
        assert BaseAlgorithm().needs_generation is False

    def test_additional_flags_are_classvar_typed(self) -> None:
        """needs_hidden_states / needs_generation 은 ``ClassVar[bool]`` 로 타이핑."""
        annotations = BaseAlgorithm.__annotations__
        for flag_name in ("needs_hidden_states", "needs_generation"):
            raw = annotations.get(flag_name)
            assert raw is not None, f"{flag_name} annotation이 없습니다."
            annotation_repr = repr(raw)
            assert "ClassVar" in annotation_repr, (
                f"{flag_name}는 ClassVar로 선언되어야 합니다. 실제: {annotation_repr}"
            )
            assert "bool" in annotation_repr, (
                f"{flag_name}는 bool 타입이어야 합니다. 실제: {annotation_repr}"
            )

    def test_dpo_inherits_all_three_defaults(self) -> None:
        """DPOLoss는 선언 없이 3개 flag default 값을 모두 상속한다 (byte-identical)."""
        for flag_name in ("needs_logits", "needs_hidden_states", "needs_generation"):
            assert flag_name not in DPOLoss.__dict__, (
                f"DPOLoss가 {flag_name}를 자체 선언했습니다. 기본값 상속으로 충분."
            )
        algo = DPOLoss()
        assert algo.needs_logits is True
        assert algo.needs_hidden_states is False
        assert algo.needs_generation is False

    def test_grpo_ppo_needs_generation_is_classvar(self) -> None:
        """GRPOLoss / PPOLoss는 needs_generation=True를 ClassVar로 선언한다.

        plain class attribute가 아닌 ClassVar 타입 승격 — 2-1 수정 본체.
        """
        for cls in (GRPOLoss, PPOLoss):
            raw = cls.__annotations__.get("needs_generation")
            assert raw is not None, (
                f"{cls.__name__}.needs_generation annotation이 없습니다. "
                "plain class attribute를 ClassVar[bool]로 승격하세요."
            )
            annotation_repr = repr(raw)
            assert "ClassVar" in annotation_repr, (
                f"{cls.__name__}.needs_generation는 ClassVar로 선언되어야 합니다. "
                f"실제: {annotation_repr}"
            )
            assert cls.needs_generation is True
            assert getattr(cls(), "needs_generation", False) is True
