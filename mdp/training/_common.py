"""Shared utilities for SFT Trainer and RLTrainer.

Device detection, distributed strategy creation, and expert parallelism setup.
"""

from __future__ import annotations

from typing import Any

import torch

from mdp.settings.schema import Settings

STRATEGY_MAP: dict[str, str] = {
    "ddp": "mdp.training.strategies.ddp.DDPStrategy",
    "fsdp": "mdp.training.strategies.fsdp.FSDPStrategy",
    "deepspeed": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "deepspeed_zero3": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
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
