"""Runtime context and RunPlan helpers for training workers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


def _device_label(local_rank: int) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{local_rank}"
    except Exception:
        pass
    return "cpu"
