"""Shared utilities for SFT Trainer and RLTrainer.

Device detection, distributed strategy creation, expert parallelism setup,
and backward/optimizer step logic.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)


def setup_amp(
    precision: str, device: torch.device,
) -> tuple[bool, torch.dtype, GradScaler]:
    """AMP/GradScaler를 설정한다. Trainer/RLTrainer 공용."""
    scaler_device = device.type
    if precision == "fp16":
        amp_enabled = True
        amp_dtype = torch.float16
        if device.type == "mps":
            logger.warning("MPS에서 fp16은 GradScaler를 지원하지 않습니다. bf16을 권장합니다.")
            scaler = GradScaler(scaler_device, enabled=False)
        else:
            scaler = GradScaler(scaler_device, enabled=True)
    elif precision == "bf16":
        amp_enabled = True
        amp_dtype = torch.bfloat16
        scaler = GradScaler(scaler_device, enabled=False)
    else:  # fp32
        amp_enabled = False
        amp_dtype = torch.float32
        scaler = GradScaler(scaler_device, enabled=False)
    return amp_enabled, amp_dtype, scaler


def create_callbacks(configs: list[dict[str, Any]], resolver: Any) -> list:
    """Recipe의 callbacks 설정에서 콜백 리스트를 생성한다."""
    callbacks = []
    for cfg in configs:
        try:
            callbacks.append(resolver.resolve(cfg))
        except Exception as e:
            logger.warning("콜백 생성 실패: %s", e)
    return callbacks


def detect_device() -> torch.device:
    """Detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_strategy(**kwargs: Any) -> Any:
    """GPU 수에 따라 적절한 분산 전략을 자동 선택한다.

    멀티 GPU → DDPStrategy, 단일 GPU/CPU → None.
    aliases.yaml에서 ``auto`` 로 등록되어 ``_component_: auto``로 사용한다.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return None
    from mdp.training.strategies.ddp import DDPStrategy

    return DDPStrategy(**kwargs)


def create_strategy(settings: Settings, resolver: Any) -> Any:
    """Config.compute.distributed에서 분산 전략을 생성한다. None이면 전략 없음."""
    dist_config = settings.config.compute.distributed
    if dist_config is None:
        return None
    if not isinstance(dist_config, dict):
        return None

    strategy_name = dist_config.get("strategy", "auto")
    if strategy_name == "none":
        return None

    # strategy 값이 이미 _component_ dict이면 직접 resolve
    if isinstance(strategy_name, dict):
        return resolver.resolve(strategy_name)

    # 문자열이면 aliases.yaml에서 조회
    strategy_kwargs = {
        k: v for k, v in dist_config.items()
        if k not in ("strategy", "moe")
    }
    return resolver.resolve({"_component_": strategy_name, **strategy_kwargs})


def create_expert_parallel(settings: Settings) -> Any:
    """Create ExpertParallel from distributed.moe config, or return None."""
    dist_config = settings.config.compute.distributed
    if dist_config is None or not isinstance(dist_config, dict):
        return None
    moe_config = dist_config.get("moe")
    if moe_config is None or not moe_config.get("enabled", False):
        return None

    from mdp.training.strategies.moe import ExpertParallel

    return ExpertParallel(
        ep_size=moe_config.get("ep_size", moe_config.get("expert_parallel_size", 1)),
        expert_module_pattern=moe_config.get("expert_module_pattern", "experts"),
    )


def backward_and_step(
    losses: dict[str, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, Any | None],
    scaler: GradScaler,
    trainable_models: dict[str, nn.Module],
    grad_accum_steps: int,
    at_accum_boundary: bool,
    grad_clip_norm: float | None = None,
    force_step: bool = False,
) -> bool | None:
    """Shared backward + optimizer step.

    Returns:
        True: optimizer step executed.
        False: backward done, not at accumulation boundary.
        None: NaN/Inf detected, gradients cleared, caller should skip.
    """
    # NaN/Inf guard
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            logger.warning("NaN/Inf loss detected in '%s', skipping step", name)
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            return None

    # Backward with accumulation scaling
    accum = 1 if force_step else grad_accum_steps
    for loss in losses.values():
        scaler.scale(loss / accum).backward()

    # Optimizer step at accumulation boundary or force
    if force_step or at_accum_boundary:
        for name, opt in optimizers.items():
            scaler.unscale_(opt)
            if grad_clip_norm is not None and name in trainable_models:
                clip_grad_norm_(trainable_models[name].parameters(), grad_clip_norm)
            scaler.step(opt)
            sched = schedulers.get(name)
            if sched is not None:
                sched.step()
        scaler.update()
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        return True

    return False
