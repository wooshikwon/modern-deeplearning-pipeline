"""DeepSpeed ZeRO training strategy."""

from __future__ import annotations

from typing import Any

from torch import nn
import torch

from mdp.training.strategies.base import BaseStrategy

_DEFAULT_DS_CONFIG: dict[str, Any] = {
    "zero_optimization": {
        "stage": 2,
    },
    "bf16": {
        "enabled": True,
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": "auto",
}


class DeepSpeedStrategy(BaseStrategy):
    """Initialises a model via :func:`deepspeed.initialize`.

    Parameters
    ----------
    ds_config:
        A DeepSpeed JSON-compatible config dict.  When *None*, a
        sensible ZeRO Stage-2 / bf16 default is used.
    """

    def __init__(self, ds_config: dict[str, Any] | None = None) -> None:
        self.ds_config = ds_config if ds_config is not None else _DEFAULT_DS_CONFIG

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:  # noqa: ARG002
        import deepspeed  # lazy import

        model_engine, *_ = deepspeed.initialize(model=model, config=self.ds_config)
        return model_engine

    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        model.save_checkpoint(path)  # type: ignore[attr-defined]

    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        model.load_checkpoint(path)  # type: ignore[attr-defined]
        return model
