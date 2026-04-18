"""`mdp/training/_schedulers.py` 공용 warmup 헬퍼 단위 테스트.

spec: dev-cycle/spec/spec-lr-warmup-configurable.md (Unit 1 verify 대상).

`parse_warmup_config` + `create_scheduler_with_warmup` 두 함수를 고립 검증한다.
Trainer / RLTrainer 전체를 띄우지 않고 optimizer + dummy base scheduler만으로
factor 적용·범위 검증·SequentialLR 래핑 로직을 검증할 수 있다 (공용 헬퍼 추출의
테스트 가속 효과).
"""

from __future__ import annotations

import pytest
import torch
from torch.optim.lr_scheduler import ConstantLR, SequentialLR

from mdp.training._schedulers import (
    DEFAULT_WARMUP_END_FACTOR,
    DEFAULT_WARMUP_START_FACTOR,
    WarmupConfig,
    create_scheduler_with_warmup,
    parse_warmup_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def optimizer() -> torch.optim.Optimizer:
    """단일 param_group optimizer. warmup 적용 시 첫 step lr 검증용."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=1.0)


@pytest.fixture
def base_scheduler(optimizer: torch.optim.Optimizer):
    """Identity base scheduler. ConstantLR(factor=1.0)로 base_lr 그대로 유지."""
    return ConstantLR(optimizer, factor=1.0, total_iters=1)


# ── parse_warmup_config ───────────────────────────────────────────────


def test_parse_warmup_config_defaults() -> None:
    """빈 dict → 모든 필드 기본값. total_steps는 ratio 미지정시 영향 없음."""
    config: dict = {}
    warmup = parse_warmup_config(config, total_steps=1000)
    assert warmup == WarmupConfig(
        warmup_steps=0,
        interval="step",
        start_factor=DEFAULT_WARMUP_START_FACTOR,
        end_factor=DEFAULT_WARMUP_END_FACTOR,
    )
    assert config == {}  # 모든 키 pop 확인


def test_parse_warmup_config_ratio_converts_to_steps() -> None:
    """weighted-ntp Baseline 재현: ratio=0.03, total=986 → steps=29."""
    config: dict = {"warmup_ratio": 0.03}
    warmup = parse_warmup_config(config, total_steps=986)
    assert warmup.warmup_steps == 29
    assert "warmup_ratio" not in config


def test_parse_warmup_config_steps_and_ratio_rejected() -> None:
    """warmup_steps와 warmup_ratio 동시 양수 → ValueError."""
    config: dict = {"warmup_steps": 10, "warmup_ratio": 0.05}
    with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
        parse_warmup_config(config, total_steps=1000)


def test_parse_warmup_config_custom_factors() -> None:
    """Recipe 옵트인 factor 지정이 WarmupConfig에 보존되는지."""
    config: dict = {
        "warmup_steps": 50,
        "warmup_start_factor": 0.1,
        "warmup_end_factor": 1.0,
        "interval": "epoch",
    }
    warmup = parse_warmup_config(config, total_steps=1000)
    assert warmup.start_factor == 0.1
    assert warmup.end_factor == 1.0
    assert warmup.warmup_steps == 50
    assert warmup.interval == "epoch"
    # 5개 pop 키가 모두 제거되어 resolver 경로에 누설되지 않음
    assert config == {}


def test_parse_warmup_config_invalid_factor_range() -> None:
    """factor 범위 위반 3가지 시나리오 모두 ValueError 보장."""
    # start_factor > end_factor
    with pytest.raises(ValueError, match="warmup factor 유효 범위 위반"):
        parse_warmup_config(
            {"warmup_start_factor": 0.5, "warmup_end_factor": 0.3},
            total_steps=1000,
        )

    # end_factor > 1.0
    with pytest.raises(ValueError, match="warmup factor 유효 범위 위반"):
        parse_warmup_config(
            {"warmup_start_factor": 0.1, "warmup_end_factor": 1.5},
            total_steps=1000,
        )

    # start_factor == 0 → PyTorch LinearLR ZeroDivisionError 선제 차단
    with pytest.raises(ValueError, match="warmup factor 유효 범위 위반"):
        parse_warmup_config(
            {"warmup_start_factor": 0.0},
            total_steps=1000,
        )


# ── create_scheduler_with_warmup ──────────────────────────────────────


def test_create_scheduler_with_warmup_no_warmup_passthrough(
    optimizer: torch.optim.Optimizer, base_scheduler
) -> None:
    """warmup_steps=0 → base_scheduler 그대로 (래핑 없음)."""
    warmup = WarmupConfig(
        warmup_steps=0,
        interval="step",
        start_factor=DEFAULT_WARMUP_START_FACTOR,
        end_factor=DEFAULT_WARMUP_END_FACTOR,
    )
    result = create_scheduler_with_warmup(optimizer, base_scheduler, warmup)
    assert result is base_scheduler


def test_create_scheduler_with_warmup_wraps_in_sequential(
    optimizer: torch.optim.Optimizer, base_scheduler
) -> None:
    """warmup_steps>0 → SequentialLR로 래핑."""
    warmup = WarmupConfig(
        warmup_steps=10,
        interval="step",
        start_factor=0.1,
        end_factor=1.0,
    )
    result = create_scheduler_with_warmup(optimizer, base_scheduler, warmup)
    assert isinstance(result, SequentialLR)


def test_create_scheduler_with_warmup_linear_first_step(
    optimizer: torch.optim.Optimizer, base_scheduler
) -> None:
    """step 0 시점 optimizer lr == base_lr × start_factor 정확 일치.

    SequentialLR은 내부적으로 첫 schedule(LinearLR)을 즉시 적용하므로
    생성 직후 param_groups[0]["lr"]을 확인하면 start_factor 반영이 검증된다.
    """
    base_lr = optimizer.param_groups[0]["lr"]
    start_factor = 0.1
    warmup = WarmupConfig(
        warmup_steps=10,
        interval="step",
        start_factor=start_factor,
        end_factor=1.0,
    )
    create_scheduler_with_warmup(optimizer, base_scheduler, warmup)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        base_lr * start_factor
    )
