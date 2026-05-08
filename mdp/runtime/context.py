"""Runtime context and SettingsPlan helpers for training workers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mdp.settings.distributed import has_distributed_intent
from mdp.settings.plan import Command, Mode, SettingsPlan
from mdp.settings.schema import Settings


@dataclass(frozen=True)
class RuntimeContext:
    """Environment-derived worker context."""

    rank: int
    local_rank: int
    world_size: int
    device: str
    result_path: Path | None = None

    @classmethod
    def from_env(cls, result_path: str | Path | None = None) -> "RuntimeContext":
        """Build a context from torchrun-style environment variables."""
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=_device_label(local_rank),
            result_path=Path(result_path) if result_path else None,
        )

    @property
    def is_torchrun(self) -> bool:
        return self.world_size > 1 or "RANK" in os.environ or "WORLD_SIZE" in os.environ

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def training_settings_plan_from_settings(
    settings: Settings,
    *,
    command: Command | None = None,
    cb_configs: Sequence[dict[str, Any]] | None = None,
) -> SettingsPlan:
    """Create a training SettingsPlan from an already validated Settings object."""
    resolved_command = command or _training_command(settings)
    return SettingsPlan(
        command=resolved_command,
        mode=_training_mode(settings, resolved_command),
        settings=settings,
        recipe_path=None,
        config_path=None,
        artifact_dir=None,
        overrides=(),
        callback_configs=tuple(dict(config) for config in (cb_configs or ())),
        validation_scope="training",
        distributed_intent=has_distributed_intent(settings),
    )


def _training_command(settings: Settings) -> Command:
    if settings.recipe.rl is not None:
        return "rl-train"
    return "train"


def _training_mode(settings: Settings, command: Command) -> Mode:
    if command == "rl-train" or settings.recipe.rl is not None:
        return "rl"
    return "sft"


def _device_label(local_rank: int) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{local_rank}"
    except Exception:
        pass
    return "cpu"
