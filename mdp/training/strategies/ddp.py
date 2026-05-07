"""DistributedDataParallel (DDP) training strategy."""

from __future__ import annotations

import os

import torch
from torch import nn

from mdp.training.strategies.base import BaseStrategy, StrategyCheckpointCapability


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

    @property
    def checkpoint_capability(self) -> StrategyCheckpointCapability:
        return StrategyCheckpointCapability(
            supports_managed_checkpoint=True,
            weight_format="safetensors",
        )

    def setup(self, model: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> nn.Module:  # noqa: ARG002
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        if device.type == "cuda":
            target_device = torch.device(f"cuda:{self._local_rank}")
            model = model.to(target_device)
            return DDP(model, device_ids=[self._local_rank])
        else:
            model = model.to(device)
            return DDP(model)

    def unwrap(self, wrapped_model: nn.Module) -> nn.Module:
        """DDP는 단순 래퍼이므로 ``.module``이 실제 model이다.

        ``invoke_custom``은 base 구현(``unwrap`` + ``getattr``)이면 충분하다:
        DDP의 gradient 동기화는 parameter autograd hook으로 이루어지므로,
        ``.module.custom_method(batch)``처럼 wrapper forward를 우회해 호출해도
        backward 시 정상적으로 all-reduce가 발생한다.
        """
        return getattr(wrapped_model, "module", wrapped_model)

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        import torch.distributed as dist

        if dist.get_rank() == 0:
            from safetensors.torch import save_file

            save_file(self.unwrap(model).state_dict(), path)

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        from safetensors.torch import load_file

        state_dict = load_file(path)
        self.unwrap(model).load_state_dict(state_dict, strict=False)
        return model

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
