"""Distributed execution intent helpers."""

from __future__ import annotations

from typing import Mapping

from mdp.settings.schema import Settings


_DEEPSPEED_ALIASES = {
    "deepspeed",
    "deepspeed_zero2",
    "deepspeed_zero3",
    "deepspeedstrategy",
    "deepspeedzero2",
    "deepspeedzero3",
    "mdp.training.strategies.deepspeed.deepspeedstrategy",
}


def get_strategy_name(settings: Settings) -> object:
    """Return the raw distributed strategy value, applying the runtime default."""
    dist_config = settings.config.compute.distributed
    if not isinstance(dist_config, Mapping):
        return None
    strategy = dist_config.get("strategy", "auto")
    if isinstance(strategy, Mapping):
        return strategy.get("_component_", "")
    return strategy


def is_deepspeed_strategy(strategy: object) -> bool:
    """Return whether a strategy value resolves to the current DeepSpeed path."""
    if not isinstance(strategy, str):
        return False
    normalized = strategy.replace("-", "_").lower()
    return normalized in _DEEPSPEED_ALIASES or normalized.endswith(".deepspeedstrategy")


def has_distributed_intent(settings: Settings) -> bool:
    """Return whether config explicitly asks for distributed execution."""
    strategy = get_strategy_name(settings)
    return strategy is not None and strategy != "none"


def should_launch_distributed(settings: Settings, detected_gpu_count: int) -> bool:
    """Decide whether the CLI should enter the torchrun path."""
    return detected_gpu_count > 1 and has_distributed_intent(settings)
