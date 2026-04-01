"""Base callback interface for training hooks."""

from __future__ import annotations


class BaseCallback:
    """Base class for all training callbacks.

    Subclasses override specific hooks to inject behavior at each
    stage of the training loop.  All hooks accept arbitrary keyword
    arguments so that the trainer can forward extra context without
    breaking existing callbacks.
    """

    should_stop: bool = False

    def on_train_start(self, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_epoch_start(self, epoch: int, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_batch_start(self, step: int, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass

    def on_validation_start(self, epoch: int, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_validation_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass

    def on_train_end(
        self,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass
