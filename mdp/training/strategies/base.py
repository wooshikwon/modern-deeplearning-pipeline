"""Base strategy interface for distributed training."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseStrategy(ABC):
    """Abstract base class for distributed training strategies.

    Every concrete strategy must implement :meth:`setup`,
    :meth:`save_checkpoint`, and :meth:`load_checkpoint`.  The optional
    :meth:`cleanup` hook is called when the training loop finishes and
    should release any distributed-process-group resources.
    """

    @abstractmethod
    def setup(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> nn.Module:
        """Prepare *model* for distributed training on *device*.

        *optimizer* is consumed by DeepSpeed (which wraps it internally).
        DDP/FSDP ignore it.

        Returns the wrapped model (e.g. ``DistributedDataParallel``).
        """

    @abstractmethod
    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        """Persist a training checkpoint to *path*."""

    @abstractmethod
    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        """Restore a checkpoint from *path* into *model* and return it."""

    def setup_models(
        self,
        models: dict[str, nn.Module],
        device: torch.device,
        trainable_names: set[str] | None = None,
        optimizers: dict[str, torch.optim.Optimizer] | None = None,
    ) -> dict[str, nn.Module]:
        """Prepare multiple models for distributed RL training.

        *trainable_names* indicates which models receive gradient updates.
        Frozen models may use a lighter sharding strategy.
        *optimizers* maps model names to their optimizers (used by DeepSpeed).
        Default: wraps each model via :meth:`setup`.
        """
        optimizers = optimizers or {}
        wrapped = {}
        for name, model in models.items():
            if trainable_names is not None and name not in trainable_names:
                # frozen model: 분산 래핑 없이 device 이동만
                wrapped[name] = model.to(device)
            else:
                wrapped[name] = self.setup(model, device, optimizer=optimizers.get(name))
        return wrapped

    def cleanup(self) -> None:
        """Release distributed resources.  No-op by default."""
