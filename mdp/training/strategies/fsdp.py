"""FullyShardedDataParallel (FSDP) training strategy."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

import torch
from torch import nn

from mdp.training.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class FSDPStrategy(BaseStrategy):
    """Wraps a model with :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    Parameters
    ----------
    sharding_strategy:
        One of ``"FULL_SHARD"``, ``"SHARD_GRAD_OP"``, ``"NO_SHARD"``,
        or ``"HYBRID_SHARD"``.  Defaults to ``"FULL_SHARD"``.
    mixed_precision:
        When *True*, enables mixed-precision policy matching the given *precision*.
    precision:
        ``"bf16"`` or ``"fp16"``.  Determines the MixedPrecision dtype.
    min_num_params:
        Minimum parameter count for auto-wrapping.  Layers with at least
        this many parameters become individual FSDP units.
    """

    def __init__(
        self,
        sharding_strategy: str = "FULL_SHARD",
        mixed_precision: bool = True,
        backend: str = "nccl",
        cpu_offload: bool = False,
        precision: str = "bf16",
        min_num_params: int = 1_000_000,
        auto_wrap_cls: str | None = None,
    ) -> None:
        self.sharding_strategy_name = sharding_strategy
        self.mixed_precision = mixed_precision
        self.backend = backend
        self.cpu_offload = cpu_offload
        self.precision = precision
        self.min_num_params = min_num_params
        self.auto_wrap_cls = auto_wrap_cls
        self._local_rank: int | None = None

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> nn.Module:  # noqa: ARG002
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        if device.type == "cuda":
            target_device = torch.device(f"cuda:{self._local_rank}")
        else:
            target_device = device

        sharding = getattr(ShardingStrategy, self.sharding_strategy_name)

        # Auto-wrap policy: transformer layer class 또는 size 기반
        if self.auto_wrap_cls is not None:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            layer_cls = self._resolve_layer_class(self.auto_wrap_cls)
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={layer_cls},
            )
        else:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=self.min_num_params,
            )

        fsdp_kwargs: dict[str, Any] = {
            "sharding_strategy": sharding,
            "auto_wrap_policy": auto_wrap_policy,
        }

        if device.type == "cuda":
            fsdp_kwargs["device_id"] = target_device

        if self.mixed_precision and device.type == "cuda":
            dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
            fsdp_kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=dtype,
                buffer_dtype=dtype,
                cast_forward_inputs=True,
            )

        if self.cpu_offload:
            from torch.distributed.fsdp import CPUOffload
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        return FSDP(model, **fsdp_kwargs)

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )
        from safetensors.torch import save_file

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(state_dict, path)

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )
        from safetensors.torch import load_file

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = load_file(path)
            model.load_state_dict(state_dict)
        return model

    def setup_models(
        self, models: dict[str, nn.Module], device: torch.device,
        trainable_names: set[str] | None = None,
    ) -> dict[str, nn.Module]:
        import torch.distributed as dist
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not dist.is_initialized():
            dist.init_process_group(backend=self.backend)

        trainable_names = trainable_names or set()
        wrapped = {}
        for name, model in models.items():
            if name in trainable_names:
                wrapped[name] = self.setup(model, device)
            else:
                # frozen → NO_SHARD (forward only, no gradient communication)
                if device.type == "cuda":
                    model = model.to(torch.device(f"cuda:{self._local_rank}"))
                wrapped[name] = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD)
        return wrapped

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    @staticmethod
    def _resolve_layer_class(cls_name: str) -> type:
        """클래스 이름 또는 전체 경로에서 transformer layer 클래스를 resolve한다."""
        import importlib

        if "." in cls_name:
            module_path, _, class_name = cls_name.rpartition(".")
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        # 단축명: transformers 패키지에서 탐색
        try:
            import transformers
            cls = getattr(transformers, cls_name, None)
            if cls is not None:
                return cls
        except ImportError:
            pass
        raise ValueError(
            f"transformer layer class '{cls_name}'를 찾을 수 없습니다. "
            "전체 경로를 사용하세요."
        )
