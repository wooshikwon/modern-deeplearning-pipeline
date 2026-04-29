"""Warmup scheduler 조립 공용 헬퍼.

Trainer / RLTrainer가 중복 구현하던 warmup 래핑 로직을 단일 지점으로 추출한다.
spec: dev-cycle/spec/spec-lr-warmup-configurable.md (Unit 1).

사용 패턴 (Trainer / RLTrainer 공통 3단계):

    config = dict(spec["scheduler"])          # copy — in-place pop 안전
    warmup = parse_warmup_config(config, total_steps)
    klass, kwargs = resolver.resolve_partial(config)
    base_scheduler = klass(optimizer, **kwargs)
    scheduler = create_scheduler_with_warmup(optimizer, base_scheduler, warmup)

본 모듈은 `torch.optim.lr_scheduler`만 의존하며, Recipe / Settings / resolver를
직접 참조하지 않는다. 검증 책임(factor 유효 범위, steps/ratio 상호배제)은
`parse_warmup_config` 내부에서 완결되며 위반 시 `ValueError`를 던진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.optim.lr_scheduler import LinearLR, LRScheduler, SequentialLR

DEFAULT_WARMUP_START_FACTOR = 1e-8
DEFAULT_WARMUP_END_FACTOR = 1.0


@dataclass(frozen=True)
class WarmupConfig:
    """Scheduler config dict에서 추출한 warmup 메타데이터의 정규형.

    Attributes
    ----------
    warmup_steps:
        절대 step 수. `warmup_ratio`가 주어진 경우 `total_steps × ratio`로
        변환된 값. `0` 이하이면 warmup 비활성.
    interval:
        ``"step"`` 또는 ``"epoch"``. SequentialLR step 축 기준.
    start_factor:
        LinearLR의 첫 step 멀티플라이어. 기본값 ``1e-8`` (HF 관례와 수치 동등).
    end_factor:
        LinearLR의 ``total_iters=warmup_steps`` 시점 멀티플라이어. 기본값 ``1.0``.
    """

    warmup_steps: int
    interval: str
    start_factor: float
    end_factor: float


def parse_warmup_config(
    config: dict[str, Any],
    total_steps: int,
) -> WarmupConfig:
    """Scheduler config dict에서 warmup 관련 필드를 pop하여 WarmupConfig 반환.

    config dict는 in-place로 수정되며, pop된 5개 키
    (``interval``, ``warmup_steps``, ``warmup_ratio``, ``warmup_start_factor``,
    ``warmup_end_factor``)는 이후 resolver.resolve_partial에 전달되지 않는다.
    caller는 반드시 ``dict(original_config)``로 사본을 만들어 전달해야 한다
    (원본 Recipe dict 오염 방지).

    Parameters
    ----------
    config:
        Pop 대상 scheduler config (in-place mutation).
    total_steps:
        ``warmup_ratio → warmup_steps`` 변환에 사용.

    Raises
    ------
    ValueError
        ``warmup_steps``와 ``warmup_ratio``가 동시에 양수이거나, factor가
        ``0 < start_factor <= end_factor <= 1.0`` 범위를 벗어날 때.
    """
    interval = config.pop("interval", "step")
    warmup_steps = config.pop("warmup_steps", 0)
    warmup_ratio = config.pop("warmup_ratio", 0.0)
    start_factor = config.pop(
        "warmup_start_factor", DEFAULT_WARMUP_START_FACTOR
    )
    end_factor = config.pop("warmup_end_factor", DEFAULT_WARMUP_END_FACTOR)

    if warmup_steps > 0 and warmup_ratio > 0:
        raise ValueError(
            "warmup_steps와 warmup_ratio를 동시에 지정할 수 없습니다. "
            f"warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio}"
        )

    if warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)

    if not (0.0 < start_factor <= end_factor <= 1.0):
        raise ValueError(
            "warmup factor 유효 범위 위반: "
            "0 < warmup_start_factor <= warmup_end_factor <= 1.0. "
            f"start_factor={start_factor}, end_factor={end_factor}"
        )

    return WarmupConfig(
        warmup_steps=int(warmup_steps),
        interval=interval,
        start_factor=float(start_factor),
        end_factor=float(end_factor),
    )


def create_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    base_scheduler: LRScheduler,
    warmup: WarmupConfig,
) -> LRScheduler:
    """base_scheduler를 LinearLR warmup으로 래핑하여 SequentialLR 반환.

    ``warmup.warmup_steps <= 0``이면 warmup 비활성으로 간주하고 base_scheduler를
    그대로 반환한다. caller가 별도 분기 없이 호출할 수 있도록 내부에서 처리.

    Parameters
    ----------
    optimizer:
        base_scheduler와 동일한 optimizer 인스턴스. LinearLR·SequentialLR에
        동일 optimizer로 바인딩된다.
    base_scheduler:
        resolver로 이미 인스턴스화된 torch scheduler.
    warmup:
        ``parse_warmup_config`` 결과.
    """
    if warmup.warmup_steps <= 0:
        return base_scheduler

    linear = LinearLR(
        optimizer,
        start_factor=warmup.start_factor,
        end_factor=warmup.end_factor,
        total_iters=warmup.warmup_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[linear, base_scheduler],
        milestones=[warmup.warmup_steps],
    )
