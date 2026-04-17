"""Unit tests for BaseStrategy.unwrap() / invoke_custom() contract.

이 두 메서드는 MDP의 선언적 계약("model이 training_step / validation_step을 선언하면
trainer가 그대로 부른다")을 DDP/FSDP 래핑 이후에도 유지시키는 브리지다.
래퍼가 없는 경우, DDP, FSDP 세 경로 모두 대칭적으로 동작해야 한다.

단일 프로세스에서 실제 DDP/FSDP 객체를 만들려면 torch.distributed init이 필요하므로,
대신 DDP의 동작(``.module`` 속성으로 내부 모델 노출)을 흉내 내는 얇은 wrapper를
쓴다. FSDP 쪽은 실제 ``torch.distributed.fsdp.FullyShardedDataParallel``이 아닐 때
base 구현으로 fallback하는 경로만 검증한다 (실제 all-gather swap trick은 분산
환경이 필요한 integration 테스트 영역).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mdp.training.strategies.base import BaseStrategy
from mdp.training.strategies.ddp import DDPStrategy
from mdp.training.strategies.fsdp import FSDPStrategy


# ---------------------------------------------------------------------------
# Fixtures: minimal model with custom methods + fake DDP wrapper
# ---------------------------------------------------------------------------


class _ModelWithCustom(nn.Module):
    """training_step / validation_step을 선언하는 최소 모델."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.train_calls = 0
        self.val_calls = 0

    def forward(self, batch: dict) -> dict:  # noqa: ARG002
        # forward는 표준 경로만 쓰는 테스트 대조군
        return {"logits": torch.zeros(1)}

    def training_step(self, batch: dict) -> torch.Tensor:
        self.train_calls += 1
        # Bradley-Terry 식을 흉내 낸 스칼라 loss
        return self.linear(batch["x"]).sum()

    def validation_step(self, batch: dict) -> dict:
        self.val_calls += 1
        return {"val_loss": float(self.linear(batch["x"]).sum().item())}


class _FakeDDPWrapper(nn.Module):
    """DDP의 구조적 본질(``.module``로 inner에 접근)만 흉내.

    실제 DDP 생성에는 ``dist.init_process_group``이 필요하지만, 우리가 검증하려는
    것은 ``DDPStrategy.unwrap``/``invoke_custom``이 ``.module`` 경로를 올바르게
    타느냐다. 따라서 duck typing으로 충분하다.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner


# ---------------------------------------------------------------------------
# BaseStrategy: no-op defaults (단일 GPU 또는 base 추상 동작)
# ---------------------------------------------------------------------------


class _ConcreteBase(BaseStrategy):
    """BaseStrategy의 추상 메서드만 최소 구현한 테스트용 구체 클래스."""

    def setup(self, model, device, optimizer=None):  # noqa: ARG002
        return model

    def save_checkpoint(self, model, path):  # noqa: ARG002
        pass

    def load_checkpoint(self, model, path):  # noqa: ARG002
        return model


class TestBaseStrategyDefaults:
    def test_unwrap_is_identity_when_not_wrapped(self):
        model = _ModelWithCustom()
        strat = _ConcreteBase()
        assert strat.unwrap(model) is model

    def test_invoke_custom_calls_method_directly(self):
        model = _ModelWithCustom()
        strat = _ConcreteBase()

        batch = {"x": torch.zeros(4)}
        result = strat.invoke_custom(model, "training_step", batch)

        assert model.train_calls == 1
        assert isinstance(result, torch.Tensor)

    def test_invoke_custom_works_for_validation_step(self):
        model = _ModelWithCustom()
        strat = _ConcreteBase()

        result = strat.invoke_custom(model, "validation_step", {"x": torch.zeros(4)})
        assert "val_loss" in result
        assert model.val_calls == 1


# ---------------------------------------------------------------------------
# DDPStrategy: ``.module`` unwrap + base invoke_custom 상속
# ---------------------------------------------------------------------------


class TestDDPStrategyCustomMethods:
    def test_unwrap_returns_inner_module(self):
        inner = _ModelWithCustom()
        wrapped = _FakeDDPWrapper(inner)
        strat = DDPStrategy()
        assert strat.unwrap(wrapped) is inner

    def test_unwrap_passes_through_when_not_wrapped(self):
        """Frozen model처럼 wrap 안 된 객체가 들어와도 안전해야 한다."""
        model = _ModelWithCustom()
        strat = DDPStrategy()
        assert strat.unwrap(model) is model

    def test_invoke_custom_reaches_inner_training_step(self):
        """base 구현(unwrap + getattr)을 통해 inner의 training_step이 호출된다."""
        inner = _ModelWithCustom()
        wrapped = _FakeDDPWrapper(inner)
        strat = DDPStrategy()

        result = strat.invoke_custom(wrapped, "training_step", {"x": torch.zeros(4)})

        assert inner.train_calls == 1
        assert isinstance(result, torch.Tensor)

    def test_invoke_custom_does_not_rely_on_wrapper_getattr(self):
        """회귀 방지: DDP는 ``__getattr__``이 custom 메서드를 forward하지 않는다.
        ``_FakeDDPWrapper``도 의도적으로 custom 메서드를 정의하지 않아,
        구현이 ``wrapped.training_step()``을 직접 시도하면 AttributeError로
        실패한다. 반드시 ``.module``을 통해 접근해야 한다.
        """
        inner = _ModelWithCustom()
        wrapped = _FakeDDPWrapper(inner)
        # wrapped 자체에는 training_step이 없다 (실제 DDP 동작과 동일).
        assert not hasattr(type(wrapped), "training_step")

        strat = DDPStrategy()
        strat.invoke_custom(wrapped, "training_step", {"x": torch.zeros(4)})
        assert inner.train_calls == 1


# ---------------------------------------------------------------------------
# FSDPStrategy: isinstance(FSDP) 아닐 때 base fallback
# ---------------------------------------------------------------------------


class TestFSDPStrategyFallback:
    def test_unwrap_returns_inner_module(self):
        inner = _ModelWithCustom()
        wrapped = _FakeDDPWrapper(inner)
        strat = FSDPStrategy(sharding_strategy="FULL_SHARD")
        # ``.module`` 속성이 있으면 unwrap. 실제 FSDP 아니어도 동일 계약.
        assert strat.unwrap(wrapped) is inner

    def test_invoke_custom_falls_back_to_base_when_not_fsdp(self):
        """FSDPStrategy를 쓰더라도 실제 인스턴스가 FSDP가 아니면
        (예: frozen model이 NO_SHARD로도 래핑되지 않은 경우, 테스트 환경 등)
        base 구현의 unwrap+getattr 경로로 안전하게 처리되어야 한다.
        """
        inner = _ModelWithCustom()
        wrapped = _FakeDDPWrapper(inner)
        strat = FSDPStrategy(sharding_strategy="FULL_SHARD")

        result = strat.invoke_custom(wrapped, "training_step", {"x": torch.zeros(4)})
        assert inner.train_calls == 1
        assert isinstance(result, torch.Tensor)

    def test_invoke_custom_on_bare_model_works(self):
        """wrapping 없이 model이 그대로 들어와도 동작."""
        model = _ModelWithCustom()
        strat = FSDPStrategy(sharding_strategy="FULL_SHARD")
        result = strat.invoke_custom(model, "validation_step", {"x": torch.zeros(4)})
        assert "val_loss" in result


# ---------------------------------------------------------------------------
# Cross-strategy symmetry: 동일 입력에 동일 효과
# ---------------------------------------------------------------------------


class TestCrossStrategySymmetry:
    @pytest.mark.parametrize(
        "strategy_factory",
        [
            lambda: _ConcreteBase(),
            lambda: DDPStrategy(),
            lambda: FSDPStrategy(sharding_strategy="FULL_SHARD"),
        ],
    )
    def test_bare_model_training_step_parity(self, strategy_factory):
        """Wrapping이 없는 model에 대해선 세 전략 모두 동일하게 작동해야 한다."""
        model = _ModelWithCustom()
        strat = strategy_factory()

        batch = {"x": torch.ones(4)}
        result = strat.invoke_custom(model, "training_step", batch)

        assert model.train_calls == 1
        assert isinstance(result, torch.Tensor)
