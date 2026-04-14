"""FSDPStrategy wrap policy 결정 로직 테스트.

_resolve_wrap_policy 메서드가 모델의 선언에 따라 올바른 wrap policy를
선택하는지 검증한다. FSDP 자체를 초기화하지 않고 policy 결정 로직만 테스트.
"""

from __future__ import annotations

import functools
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor, nn

from mdp.models.base import BaseModel
from mdp.training.strategies.fsdp import FSDPStrategy


# ── 테스트용 모델 ──


class _BlockDeclaredModel(BaseModel):
    """_block_classes가 선언된 BaseModel 서브클래스."""

    _block_classes = {"FakeDecoderLayer"}

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.tensor(0.0)

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        return {}


class _NoneBlockModel(BaseModel):
    """_block_classes = None인 BaseModel 서브클래스."""

    _block_classes = None

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.tensor(0.0)

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        return {}


class _MultiBlockModel(BaseModel):
    """복수 _block_classes (VLM 패턴)."""

    _block_classes = {"FakeDecoderLayer", "FakeEncoderLayer"}

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        return torch.tensor(0.0)

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        return {}


# ── 테스트 ──


class TestFSDPWrapPolicyResolution:
    """_resolve_wrap_policy의 4단계 우선순위를 검증한다."""

    def _make_strategy(self, **kwargs) -> FSDPStrategy:
        return FSDPStrategy(**kwargs)

    def _fake_size_policy(self):
        """size_based_auto_wrap_policy 대역."""
        return MagicMock(name="size_based_auto_wrap_policy")

    # ---- 1) 사용자 명시 auto_wrap_cls (최우선) ----

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_user_explicit_auto_wrap_cls_takes_priority(self, mock_resolve) -> None:
        """auto_wrap_cls가 지정되면 모델의 _block_classes보다 우선."""
        strategy = self._make_strategy(auto_wrap_cls="UserSpecifiedLayer")
        model = _BlockDeclaredModel()

        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())

        # transformer_auto_wrap_policy의 functools.partial이 반환됨
        assert isinstance(policy, functools.partial)
        mock_resolve.assert_called_once_with("UserSpecifiedLayer")

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_user_explicit_auto_wrap_cls_list(self, mock_resolve) -> None:
        """auto_wrap_cls가 리스트일 때도 동작."""
        strategy = self._make_strategy()
        strategy.auto_wrap_cls = ["LayerA", "LayerB"]
        model = _NoneBlockModel()

        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())

        assert isinstance(policy, functools.partial)
        assert mock_resolve.call_count == 2

    # ---- 2) MDP 계약: _block_classes ----

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_block_classes_declared_uses_transformer_policy(self, mock_resolve) -> None:
        """_block_classes가 선언된 BaseModel은 transformer_auto_wrap_policy."""
        strategy = self._make_strategy()
        model = _BlockDeclaredModel()

        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())

        assert isinstance(policy, functools.partial)
        mock_resolve.assert_called_once_with("FakeDecoderLayer")

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_block_classes_multiple_resolves_all(self, mock_resolve) -> None:
        """복수 _block_classes는 모두 resolve되어 wrap set에 포함."""
        strategy = self._make_strategy()
        model = _MultiBlockModel()

        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())

        assert isinstance(policy, functools.partial)
        assert mock_resolve.call_count == 2

    # ---- 3) HF 호환: _no_split_modules ----

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_hf_no_split_modules_fallback(self, mock_resolve) -> None:
        """BaseModel이 아닌 HF 모델이 _no_split_modules를 가지면 그것을 사용."""
        strategy = self._make_strategy()
        model = nn.Linear(8, 8)  # 일반 nn.Module
        model._no_split_modules = ["LlamaDecoderLayer"]

        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())

        assert isinstance(policy, functools.partial)
        mock_resolve.assert_called_once_with("LlamaDecoderLayer")

    # ---- 4) size 기반 폴백 ----

    def test_no_declaration_uses_size_based(self) -> None:
        """선언이 없는 일반 모델은 size_based_auto_wrap_policy."""
        strategy = self._make_strategy(min_num_params=500_000)
        model = nn.Linear(8, 8)  # _block_classes도 _no_split_modules도 없음

        size_policy = self._fake_size_policy()
        policy = strategy._resolve_wrap_policy(model, size_policy)

        # size_based의 partial이 반환됨
        assert isinstance(policy, functools.partial)

    def test_block_classes_none_uses_size_based(self) -> None:
        """_block_classes = None인 모델은 size_based_auto_wrap_policy."""
        strategy = self._make_strategy()
        model = _NoneBlockModel()

        size_policy = self._fake_size_policy()
        policy = strategy._resolve_wrap_policy(model, size_policy)

        assert isinstance(policy, functools.partial)

    # ---- 위험 조합 차단: LoRA + FSDP + 선언 없음 ----

    def test_peft_without_declaration_raises_value_error(self) -> None:
        """LoRA(PEFT) + FSDP + block 미선언 → ValueError."""
        strategy = self._make_strategy()
        model = nn.Linear(8, 8)
        model.peft_config = {"some": "config"}  # PEFT 모델 시뮬레이션

        with pytest.raises(ValueError, match="LoRA.*FSDP.*필수"):
            strategy._resolve_wrap_policy(model, self._fake_size_policy())

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_peft_with_block_classes_is_safe(self, mock_resolve) -> None:
        """LoRA(PEFT) + FSDP이지만 _block_classes가 있으면 정상 동작."""
        strategy = self._make_strategy()
        model = _BlockDeclaredModel()
        model.peft_config = {"some": "config"}

        # _block_classes가 있으므로 transformer_auto_wrap_policy 사용
        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())
        assert isinstance(policy, functools.partial)

    @patch.object(FSDPStrategy, "_resolve_layer_class", return_value=nn.Linear)
    def test_peft_with_no_split_modules_is_safe(self, mock_resolve) -> None:
        """LoRA(PEFT) + FSDP이지만 HF _no_split_modules가 있으면 정상 동작."""
        strategy = self._make_strategy()
        model = nn.Linear(8, 8)
        model.peft_config = {"some": "config"}
        model._no_split_modules = ["LlamaDecoderLayer"]

        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())
        assert isinstance(policy, functools.partial)

    def test_non_peft_without_declaration_is_safe(self) -> None:
        """LoRA 없이 FSDP만 사용하면 size_based도 안전 (에러 없음)."""
        strategy = self._make_strategy()
        model = nn.Linear(8, 8)  # peft_config 없음

        # ValueError 발생하지 않음
        policy = strategy._resolve_wrap_policy(model, self._fake_size_policy())
        assert isinstance(policy, functools.partial)
