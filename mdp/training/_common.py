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


STRATEGY_MAP: dict[str, str] = {
    "ddp": "mdp.training.strategies.ddp.DDPStrategy",
    "fsdp": "mdp.training.strategies.fsdp.FSDPStrategy",
    "deepspeed_zero2": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "deepspeed_zero3": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    # deprecated alias — 다음 major 버전에서 제거
    "deepspeed": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
}


def detect_device() -> torch.device:
    """Detect the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_strategy(settings: Settings, resolver: Any) -> Any:
    """Create a distributed strategy from settings, or return None."""
    dist_config = settings.config.compute.distributed
    if dist_config is None:
        return None
    strategy_name = dist_config.get("strategy", "auto") if isinstance(dist_config, dict) else "auto"
    if strategy_name in ("none", "auto"):
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
            return None
        strategy_name = "ddp"
    class_path = STRATEGY_MAP.get(strategy_name)
    if class_path is None:
        raise ValueError(f"알 수 없는 분산 전략: {strategy_name}")
    strategy_kwargs = {
        k: v for k, v in (dist_config if isinstance(dist_config, dict) else {}).items()
        if k not in ("strategy", "moe")
    }
    return resolver.resolve({"_component_": class_path, **strategy_kwargs})


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
