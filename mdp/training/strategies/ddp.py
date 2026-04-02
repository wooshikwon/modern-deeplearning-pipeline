"""DistributedDataParallel (DDP) training strategy."""

from __future__ import annotations

import os

import torch
from torch import nn

from mdp.training.strategies.base import BaseStrategy


class DDPStrategy(BaseStrategy):
    """Wraps a model with :class:`~torch.nn.parallel.DistributedDataParallel`.

    The local rank is read from the ``LOCAL_RANK`` environment variable
    (set by ``torchrun``).  The NCCL backend is used by default.
    """

    def __init__(self, backend: str = "nccl") -> None:
        self.backend = backend
        self._local_rank: int | None = None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:  # noqa: ARG002
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        cuda_device = torch.device(f"cuda:{self._local_rank}")
        model = model.to(cuda_device)
        return DDP(model, device_ids=[self._local_rank])

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        import torch.distributed as dist

        if dist.get_rank() == 0:
            from safetensors.torch import save_file

            save_file(model.module.state_dict(), path)

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        from safetensors.torch import load_file

        state_dict = load_file(path)
        model.module.load_state_dict(state_dict)
        return model

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
