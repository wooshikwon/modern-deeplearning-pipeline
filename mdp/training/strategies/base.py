"""Base strategy interface for distributed training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class StrategyCheckpointCapability:
    """Checkpoint behavior advertised by a distributed strategy."""

    supports_managed_checkpoint: bool = False
    requires_all_ranks_for_save: bool = False
    weight_format: str = "unsupported"
    unsupported_reason: str | None = None


class BaseStrategy(ABC):
    """Abstract base class for distributed training strategies.

    Every concrete strategy must implement :meth:`setup`,
    :meth:`save_checkpoint`, and :meth:`load_checkpoint`.  The optional
    :meth:`cleanup` hook is called when the training loop finishes and
    should release any distributed-process-group resources.

    ``checkpoint_capability`` is consumed by ``CheckpointManager`` before it
    calls strategy-owned weight I/O.  The default is deliberately unsupported:
    new strategies must opt in once their checkpoint semantics are compatible
    with the manifest-based manager.  This keeps DeepSpeed ZeRO checkpoints out
    of the DDP/FSDP restore path until a separate engine-contract spec owns
    engine state, optimizer shards, and resume semantics.

    ``unwrap`` and ``invoke_custom`` bridge the gap between MDP's
    declarative contract ("a model may define ``training_step`` /
    ``validation_step`` to own its own loss logic") and the runtime
    reality that those methods are hidden behind distributed wrappers
    (DDP's ``__getattr__`` does not forward custom methods; FSDP's
    all-gather hooks only fire through the wrapper's ``forward``).
    Trainers call these two helpers so model authors can keep writing
    plain methods without special-casing DDP/FSDP themselves.
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

    @property
    def checkpoint_capability(self) -> StrategyCheckpointCapability:
        """Return this strategy's checkpoint-manager compatibility contract."""
        return StrategyCheckpointCapability(
            unsupported_reason=(
                f"{type(self).__name__} does not declare manifest checkpoint "
                "compatibility"
            )
        )

    # ------------------------------------------------------------------
    # Declarative-contract bridges
    # ------------------------------------------------------------------

    def unwrap(self, wrapped_model: nn.Module) -> nn.Module:
        """Return the underlying model, stripping any distributed wrapper.

        Use this for **read-only** attribute access — ``hasattr``, ``getattr``,
        reading ``.config``, listing parameters, and the like.  For **calling**
        a custom method whose semantics depend on wrapper hooks (gradient
        synchronization, parameter all-gather), use :meth:`invoke_custom`
        instead.

        Default: no-op.  Subclasses that wrap the model (DDP, FSDP) override
        to return ``wrapped_model.module``.
        """
        return wrapped_model

    def invoke_custom(
        self,
        wrapped_model: nn.Module,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke a model's custom method while preserving distributed semantics.

        Models may define ``training_step``, ``validation_step``, and similar
        methods to encapsulate non-standard loss / evaluation logic (e.g.
        Bradley-Terry pairwise ranking in a value model).  Those methods are
        not forwarded through DDP's ``__getattr__`` and must not short-circuit
        FSDP's all-gather hooks — hence this strategy-specific dispatch.

        Default implementation: resolve the method on the unwrapped model and
        call it directly.  This is correct for:

        - Unwrapped models (``strategy is None`` path where the trainer
          performs plain calls).
        - DDP: its gradient synchronization is driven by autograd hooks on
          parameters, so calling through ``.module`` leaves that machinery
          intact.  ``DDPStrategy`` therefore relies on this default and only
          overrides :meth:`unwrap`.

        FSDP and any other strategy whose forward path carries essential
        hooks must override this method.
        """
        return getattr(self.unwrap(wrapped_model), method_name)(*args, **kwargs)

    # ------------------------------------------------------------------
    # Multi-model (RL) setup
    # ------------------------------------------------------------------

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
