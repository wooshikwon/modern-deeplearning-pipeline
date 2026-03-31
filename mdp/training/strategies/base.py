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
    def setup(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Prepare *model* for distributed training on *device*.

        Returns the wrapped model (e.g. ``DistributedDataParallel``).
        """

    @abstractmethod
    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        """Persist a training checkpoint to *path*."""

    @abstractmethod
    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        """Restore a checkpoint from *path* into *model* and return it."""

    def cleanup(self) -> None:
        """Release distributed resources.  No-op by default."""
