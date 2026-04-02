"""DeepSpeed ZeRO training strategy."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn
import torch

from mdp.training.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_DEFAULT_DS_CONFIG: dict[str, Any] = {
    "zero_optimization": {
        "stage": 2,
    },
    "bf16": {
        "enabled": True,
    },
    "gradient_clipping": 1.0,
}


class DeepSpeedStrategy(BaseStrategy):
    """Initialises a model via :func:`deepspeed.initialize`.

    Parameters
    ----------
    ds_config:
        A DeepSpeed JSON-compatible config dict.  When *None*, a
        sensible ZeRO Stage-2 / bf16 default is used.
    batch_size:
        Micro batch size per GPU. Required by DeepSpeed.
    """

    def __init__(
        self,
        ds_config: dict[str, Any] | None = None,
        batch_size: int = 32,
        moe: dict | None = None,
    ) -> None:
        self.ds_config = dict(ds_config) if ds_config is not None else dict(_DEFAULT_DS_CONFIG)
        self.batch_size = batch_size
        self._engine = None

        if moe:
            if "moe" in self.ds_config:
                logger.warning(
                    "ds_config에 이미 'moe' 키가 존재합니다. "
                    "moe 파라미터로 덮어씁니다."
                )
            self.ds_config["moe"] = {
                "enabled": True,
                "ep_size": moe.get("expert_parallel_size", 1),
                "moe_param_group": True,
            }
            if "num_experts" in moe:
                self.ds_config["moe"]["num_experts"] = moe["num_experts"]

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def setup(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> nn.Module:  # noqa: ARG002
        import deepspeed  # lazy import

        # batch size를 config에 주입
        self.ds_config["train_micro_batch_size_per_gpu"] = self.batch_size

        init_kwargs: dict[str, Any] = {
            "model": model,
            "model_parameters": model.parameters(),
            "config": self.ds_config,
        }
        if optimizer is not None:
            init_kwargs["optimizer"] = optimizer

        model_engine, *_ = deepspeed.initialize(**init_kwargs)
        self._engine = model_engine
        return model_engine

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        model.save_checkpoint(path)  # type: ignore[attr-defined]

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        model.load_checkpoint(path)  # type: ignore[attr-defined]
        return model

    def setup_models(
        self, models: dict[str, nn.Module], device: torch.device,
        trainable_names: set[str] | None = None,
    ) -> dict[str, nn.Module]:
        import deepspeed

        trainable_names = trainable_names or set()
        wrapped = {}
        for name, model in models.items():
            if name in trainable_names:
                self.ds_config["train_micro_batch_size_per_gpu"] = self.batch_size
                engine, *_ = deepspeed.initialize(
                    model=model, model_parameters=model.parameters(),
                    config=dict(self.ds_config),
                )
                wrapped[name] = engine
            else:
                wrapped[name] = model.to(device)
        return wrapped

    def cleanup(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
