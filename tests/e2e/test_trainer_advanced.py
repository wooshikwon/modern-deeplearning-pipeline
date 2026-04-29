"""Trainer 통합 경로 테스트: precision, validation 주기, warmup, callback 시점.

Tests:
- test_bf16_precision: bf16 autocast 학습 완료
- test_step_based_validation: val_check_unit="step" 에폭 내 검증
- test_fractional_epoch_validation: val_check_interval=0.5 에폭당 2회 검증
- test_on_batch_end_per_accumulation_step: grad_accum 기준 콜백 호출
- test_warmup_ratio_creates_sequential_scheduler: warmup_ratio → SequentialLR
- test_warmup_mutual_exclusion: warmup_steps + warmup_ratio → ValueError
- test_warmup_default_factors_unchanged: factor 생략 시 1e-8/1.0 기본값 유지
- test_warmup_custom_start_factor_from_recipe: Recipe 필드가 LinearLR 인자에 도달
- test_warmup_invalid_factor_range_rejected: 유효 범위 위반 3케이스 ValueError
- test_baseline_info_safe_on_exception: 학습 예외 시 NameError 없이 전파
- test_device_map_model_rejected: device_map 분산 모델 학습 진입 차단
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR

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


# ─────────────────────────────────────────────────────────────────────
# spec-lr-warmup-configurable (U4): warmup factor Recipe opt-in e2e 검증.
# 핵심 약속: (1) 기본값 유지, (2) Recipe 필드가 LinearLR까지 도달, (3) 유효
# 범위 위반 차단. 내부 접근 경로는 `trainer.scheduler` → SequentialLR →
# `._schedulers[0]`(= LinearLR). 가능하면 `.start_factor`·`.end_factor`
# `.total_iters` public 속성만 읽어 PyTorch 내부 구조 변경 노출을 최소화.
# ─────────────────────────────────────────────────────────────────────


def _linear_warmup_from(trainer: Trainer) -> LinearLR:
    """trainer.scheduler가 SequentialLR 래핑임을 확인하고 LinearLR 추출."""
    sched = trainer.scheduler
    assert isinstance(sched, SequentialLR), (
        f"warmup 활성화 시 SequentialLR이어야 합니다. got={type(sched).__name__}"
    )
    linear = sched._schedulers[0]
    assert isinstance(linear, LinearLR), (
        f"SequentialLR 첫 번째 스케줄러는 LinearLR이어야 합니다. "
        f"got={type(linear).__name__}"
    )
    return linear


def test_warmup_default_factors_unchanged() -> None:
    """factor 필드를 recipe에 적지 않으면 MDP 기본값 1e-8 / 1.0 유지.

    backward compatibility 보장용. 기존 fixture·외부 consumer가 새 필드를
    도입하지 않아도 동일 warmup dynamics를 얻는다.
    """
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
    linear = _linear_warmup_from(trainer)

    assert math.isclose(linear.start_factor, 1e-8, rel_tol=1e-12)
    assert linear.end_factor == 1.0


def test_warmup_custom_start_factor_from_recipe() -> None:
    """warmup_start_factor=0.1이 LinearLR에 도달하고 step 0 lr이 정확히 `base_lr × 0.1`.

    파이프라인 끝단 검증: Recipe YAML → schema → _create_scheduler →
    parse_warmup_config → create_scheduler_with_warmup → LinearLR(start_factor=0.1).
    """
    base_lr = 1e-3
    settings = make_test_settings(
        epochs=2,
        optimizer={"_component_": "AdamW", "lr": base_lr},
        scheduler={
            "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 2,
            "warmup_ratio": 0.5,
            "warmup_start_factor": 0.1,
            "warmup_end_factor": 1.0,
        },
    )
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    train_loader = ListDataLoader(make_vision_batches(4, 4, 2, 8))

    trainer = Trainer(settings=settings, model=model, train_loader=train_loader)
    linear = _linear_warmup_from(trainer)

    assert math.isclose(linear.start_factor, 0.1, rel_tol=1e-12)
    assert math.isclose(linear.end_factor, 1.0, rel_tol=1e-12)

    # Step 0 직후: LinearLR의 첫 lr = base_lr × start_factor.
    # PyTorch LRScheduler는 생성 시 get_lr()이 한 번 호출되므로 param_groups
    # 에 이미 반영돼 있다.
    current_lr = trainer.optimizer.param_groups[0]["lr"]
    assert current_lr == pytest.approx(base_lr * 0.1, rel=1e-9)


@pytest.mark.parametrize(
    "factors",
    [
        {"warmup_start_factor": -0.1, "warmup_end_factor": 1.0},   # 음수
        {"warmup_start_factor": 1.5, "warmup_end_factor": 1.5},    # 1.0 초과
        {"warmup_start_factor": 0.5, "warmup_end_factor": 0.3},    # start > end
    ],
)
def test_warmup_invalid_factor_range_rejected(factors: dict) -> None:
    """유효 범위(0 < start <= end <= 1.0) 위반 시 Trainer 초기화가 ValueError.

    `parse_warmup_config`의 범위 검증이 헬퍼 경유로 propagate된다. start_factor=0
    ZeroDivisionError도 차단(양수 강제).
    """
    settings = make_test_settings(
        epochs=2,
        scheduler={
            "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 2,
            "warmup_ratio": 0.5,
            **factors,
        },
    )
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    train_loader = ListDataLoader(make_vision_batches(4, 4, 2, 8))

    with pytest.raises(ValueError, match="warmup factor 유효 범위 위반"):
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


def test_device_map_model_rejected() -> None:
    """hf_device_map이 있는 모델로 train() 호출 시 RuntimeError."""
    settings = make_test_settings(epochs=1)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    model.hf_device_map = {"": 0}  # accelerate dispatch 흔적 모사
    train_loader = ListDataLoader(make_vision_batches(2, 4, 2, 8))

    trainer = Trainer(settings=settings, model=model, train_loader=train_loader)
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    with pytest.raises(RuntimeError, match="device_map"):
        trainer.train()


# ─────────────────────────────────────────────────────────────────────
# spec-logging-consistency (U5): Trainer의 step-level LR metric 기록 검증.
#
# U2에서 `log_step_metrics`를 `_train_one_epoch`의 grad_accum 경계 + residual
# flush 두 지점에 삽입했다. 본 테스트는 warmup scheduler를 사용할 때 step 0
# 시점의 optimizer LR이 `recipe.optimizer.lr × warmup_start_factor`로 시작하며,
# 그 값이 MLflow `log_metrics`에 그대로 흐르는지 검증한다. 과거 `policy_lr`
# 스냅샷이 param에 박히던 weighted-ntp Phase 3 혼란을 반대쪽(metric 경로)에서
# 회귀 방어한다.
# ─────────────────────────────────────────────────────────────────────


def test_trainer_logs_step_level_lr() -> None:
    """Trainer가 step마다 `learning_rate` metric을 기록하고, warmup step 0에서
    `base_lr × start_factor` 값이 관측된다.

    실제 `mdp.training._mlflow_logging` 모듈 + 실제 optimizer 인스턴스를 거치되,
    MLflow 자체는 `unittest.mock.patch`로 캡처한다(네트워크 없음). `log_metrics`
    호출 중 step=1(첫 optimizer step 이후)의 인자에 `learning_rate` 키가 존재하고,
    값이 `base_lr × warmup_start_factor` 근처임을 확인한다.
    """
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch

    base_lr = 1e-4
    warmup_start_factor = 1e-8  # MDP 기본값 (LinearLR ZeroDivisionError 회피)

    settings = make_test_settings(
        epochs=1,
        optimizer={"_component_": "AdamW", "lr": base_lr},
        scheduler={
            "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 5,
            "warmup_ratio": 0.03,
            "warmup_start_factor": warmup_start_factor,
            "warmup_end_factor": 1.0,
        },
    )
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    # 5개 이상 배치를 둬 최소 2회 step-level 로깅이 발생하도록 한다.
    train_loader = ListDataLoader(make_vision_batches(5, 4, 2, 8))

    trainer = Trainer(settings=settings, model=model, train_loader=train_loader)
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    # MLflow run이 시작된 것처럼 보이게 `active_run`이 truthy를 반환. `_start_mlflow_run`
    # 은 nullcontext로 대체하여 파일시스템 tracking backend의 실제 디렉토리 생성도
    # 없앤다.
    with patch.object(trainer, "_start_mlflow_run", return_value=nullcontext()), patch(
        "mlflow.active_run", return_value=MagicMock()
    ), patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.log_params"
    ), patch("mlflow.set_tag"), patch("mlflow.log_dict"), patch(
        "mlflow.log_artifacts"
    ):
        trainer.train()

    # ─── 검증 ───────────────────────────────────────────────────────
    # `log_step_metrics`가 호출될 때마다 `mlflow.log_metrics(merged, step=...)` 형태.
    # step 인자가 있는 호출 중 `learning_rate` 키를 포함한 것만 추린다.
    step_calls = [
        c for c in mock_log_metrics.call_args_list
        if c.kwargs.get("step") is not None and "learning_rate" in c.args[0]
    ]
    assert len(step_calls) > 0, (
        f"step-level learning_rate 로그가 없습니다. "
        f"log_metrics calls: {mock_log_metrics.call_args_list}"
    )

    # 첫 step(=1) 호출에서 관측되는 learning_rate 값.
    first_step_call = step_calls[0]
    logged_lr = first_step_call.args[0]["learning_rate"]

    # Warmup step 1의 LR: LinearLR(start_factor=1e-8, end_factor=1.0)는 step 0에서
    # 이미 `base_lr × start_factor`를 param_groups에 반영한 상태로 시작한다.
    # 이후 매 step마다 선형 증가하지만, total_iters가 충분히 크면 첫 step 값은
    # 여전히 `start_factor × base_lr` 수준(≤ end_factor × base_lr 상한).
    expected_initial = base_lr * warmup_start_factor
    # end_factor × base_lr 이하가 LinearLR 동작의 상한 (Cosine 단계 이전).
    assert logged_lr <= base_lr, (
        f"Warmup 적용 후 첫 step LR이 base_lr({base_lr})보다 커서는 안 됨: {logged_lr}"
    )
    # start_factor × base_lr 이상(첫 step에서 이미 최소 1 interval 진행).
    assert logged_lr >= expected_initial * 0.999, (
        f"첫 step LR({logged_lr})이 start_factor×base_lr({expected_initial})에 미달. "
        f"warmup LinearLR 경로가 깨진 것"
    )

    # `train_loss`도 같은 step 호출에 병합되어 있어야 한다(단일 round-trip).
    assert "train_loss" in first_step_call.args[0], (
        f"step-level 호출의 extra에 train_loss가 있어야 합니다. "
        f"keys={list(first_step_call.args[0].keys())}"
    )
