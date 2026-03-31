"""FullyShardedDataParallel (FSDP) training strategy."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from mdp.training.strategies.base import BaseStrategy


class FSDPStrategy(BaseStrategy):
    """Wraps a model with :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    Parameters
    ----------
    sharding_strategy:
        One of ``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``, ``"NO_SHARD"``,
        or ``"HYBRID_SHARD"``.  Defaults to ``"FULL_SHARD"``.
    mixed_precision:
        When *True*, enables ``bfloat16`` param / reduce / buffer
        mixed-precision policy.
    """

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
        backend: str = "nccl",
    ) -> None:
        self.sharding_strategy_name = sharding_strategy
        self.mixed_precision = mixed_precision
        self.backend = backend
        self._local_rank: int | None = None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:  # noqa: ARG002
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        cuda_device = torch.device(f"cuda:{self._local_rank}")
        sharding = getattr(ShardingStrategy, self.sharding_strategy_name)

        fsdp_kwargs: dict[str, Any] = {
            "sharding_strategy": sharding,
            "device_id": cuda_device,
        }

        if self.mixed_precision:
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            )

        return FSDP(model, **fsdp_kwargs)

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                torch.save(state_dict, path)

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
