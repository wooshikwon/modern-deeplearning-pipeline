"""Trainer 통합 경로 테스트: precision, validation 주기, warmup, callback 시점.

7 tests:
- test_bf16_precision: bf16 autocast 학습 완료
- test_step_based_validation: val_check_unit="step" 에폭 내 검증
- test_fractional_epoch_validation: val_check_interval=0.5 에폭당 2회 검증
- test_on_batch_end_per_accumulation_step: grad_accum 기준 콜백 호출
- test_warmup_ratio_creates_sequential_scheduler: warmup_ratio → SequentialLR
- test_warmup_mutual_exclusion: warmup_steps + warmup_ratio → ValueError
- test_baseline_info_safe_on_exception: 학습 예외 시 NameError 없이 전파
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mdp.training.callbacks.base import BaseCallback
from mdp.training.trainer import Trainer
from tests.e2e.conftest import make_test_settings
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


class _ValidationCounter(BaseCallback):
    """검증 횟수를 세는 콜백."""

    def __init__(self) -> None:
        self.count = 0

    def on_validation_end(self, **kwargs) -> None:
        self.count += 1


class _BatchEndCounter(BaseCallback):
    """on_batch_end 호출 횟수를 세는 콜백."""

    def __init__(self) -> None:
        self.count = 0

    def on_batch_end(self, **kwargs) -> None:
        self.count += 1


def test_bf16_precision() -> None:
    """bf16 autocast로 학습이 완료되는지 확인."""
    settings = make_test_settings(epochs=2, precision="bf16")
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    train_loader = ListDataLoader(make_vision_batches(3, 4, 2, 8))

    trainer = Trainer(settings=settings, model=model, train_loader=train_loader)
    trainer.device = torch.device("cpu")

    result = trainer.train()
    assert result["total_epochs"] == 2
    assert result["total_steps"] > 0


def test_step_based_validation() -> None:
    """val_check_unit='step', interval=3일 때 에폭 내 검증이 실행되는지."""
    counter = _ValidationCounter()
    settings = make_test_settings(
        epochs=1, val_check_interval=3, val_check_unit="step",
    )
    model = TinyVisionModel(num_classes=2, hidden_dim=16)

    # 10 배치 → step 3, 6, 9에서 mid-epoch 검증 + 에폭 끝 = 4회
    train_batches = make_vision_batches(10, 4, 2, 8)
    val_batches = make_vision_batches(2, 4, 2, 8)
    trainer = Trainer(
        settings=settings, model=model,
        train_loader=ListDataLoader(train_batches),
        val_loader=ListDataLoader(val_batches),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    trainer.callbacks.append(counter)

    trainer.train()
    assert counter.count == 4, f"Expected 4 validations, got {counter.count}"


def test_fractional_epoch_validation() -> None:
    """val_check_interval=0.5일 때 에폭당 2회 검증."""
    counter = _ValidationCounter()
    settings = make_test_settings(epochs=2, val_check_interval=0.5)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)

    # 10 배치, 0.5 에폭 = 5배치마다 → mid + end = 에폭당 2회, 2에폭 = 4회
    train_batches = make_vision_batches(10, 4, 2, 8)
    val_batches = make_vision_batches(2, 4, 2, 8)
    trainer = Trainer(
        settings=settings, model=model,
        train_loader=ListDataLoader(train_batches),
        val_loader=ListDataLoader(val_batches),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    trainer.callbacks.append(counter)

    trainer.train()
    assert counter.count == 4, f"Expected 4 validations (2/epoch × 2 epochs), got {counter.count}"


def test_on_batch_end_per_accumulation_step() -> None:
    """grad_accum=4일 때 on_batch_end가 accumulation step 기준으로 호출되는지."""
    counter = _BatchEndCounter()
    settings = make_test_settings(epochs=1, gradient_accumulation_steps=4)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)

    # 8 배치, accum=4 → 2 accumulation steps per epoch
    train_batches = make_vision_batches(8, 4, 2, 8)
    trainer = Trainer(
        settings=settings, model=model,
        train_loader=ListDataLoader(train_batches),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    trainer.callbacks.append(counter)

    trainer.train()
    assert counter.count == 2, (
        f"Expected 2 on_batch_end calls (8 batches / 4 accum), got {counter.count}"
    )


def test_warmup_ratio_creates_sequential_scheduler() -> None:
    """warmup_ratio가 SequentialLR을 생성하는지."""
    settings = make_test_settings(
        epochs=2,
        scheduler={
            "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 2,
            "warmup_ratio": 0.5,
        },
    )
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    train_loader = ListDataLoader(make_vision_batches(4, 4, 2, 8))

    trainer = Trainer(settings=settings, model=model, train_loader=train_loader)
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    assert trainer.scheduler is not None
    assert "SequentialLR" in type(trainer.scheduler).__name__

    result = trainer.train()
    assert result["total_epochs"] == 2


def test_warmup_mutual_exclusion() -> None:
    """warmup_steps와 warmup_ratio 동시 지정 → ValueError."""
    settings = make_test_settings(
        epochs=2,
        scheduler={
            "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 2,
            "warmup_steps": 10,
            "warmup_ratio": 0.5,
        },
    )
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    train_loader = ListDataLoader(make_vision_batches(4, 4, 2, 8))

    with pytest.raises(ValueError, match="warmup_steps.*warmup_ratio"):
        Trainer(settings=settings, model=model, train_loader=train_loader)


def test_baseline_info_safe_on_exception() -> None:
    """학습 중 예외가 발생해도 NameError 없이 예외가 전파되는지."""

    class _CrashingModel(nn.Module):
        """2번째 배치에서 RuntimeError를 발생시키는 모델."""
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3 * 8 * 8, 2)
            self._call_count = 0

        def forward(self, batch):
            x = batch["pixel_values"].flatten(1)
            return {"logits": self.linear(x)}

        def training_step(self, batch):
            self._call_count += 1
            if self._call_count >= 2:
                raise RuntimeError("Intentional crash for testing")
            x = batch["pixel_values"].flatten(1)
            logits = self.linear(x)
            return torch.nn.functional.cross_entropy(logits, batch["labels"])

        def validation_step(self, batch):
            return {"loss": 0.0}

    settings = make_test_settings(epochs=5)
    model = _CrashingModel()
    train_loader = ListDataLoader(make_vision_batches(5, 4, 2, 8))

    trainer = Trainer(settings=settings, model=model, train_loader=train_loader)
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    # 예외가 RuntimeError로 전파되어야 하고, NameError가 아니어야 한다
    with pytest.raises(RuntimeError, match="Intentional crash"):
        trainer.train()
