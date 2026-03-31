"""Early stopping callback to halt training when a metric stops improving."""

from __future__ import annotations

import logging
import math

from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class EarlyStopping(BaseCallback):
    """Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    monitor:
        Name of the metric to monitor (e.g. ``"val_loss"``).
    patience:
        Number of validation checks with no improvement after which
        training is stopped.
    mode:
        ``"min"`` expects the metric to decrease; ``"max"`` expects it
        to increase.
    min_delta:
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in ("min", "max"):
            msg = f"mode must be 'min' or 'max', got '{mode}'"
            raise ValueError(msg)

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.best_value: float = math.inf if mode == "min" else -math.inf
        self.counter: int = 0
        self.should_stop: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_validation_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if metrics is None or self.monitor not in metrics:
            logger.warning(
                "EarlyStopping: metric '%s' not found in metrics, skipping.",
                self.monitor,
            )
            return

        current = metrics[self.monitor]

        if self._is_improvement(current):
            self.best_value = current
            self.counter = 0
            logger.debug(
                "EarlyStopping: %s improved to %.6f",
                self.monitor,
                current,
            )
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: %s did not improve (%.6f vs best %.6f), "
                "counter %d/%d",
                self.monitor,
                current,
                self.best_value,
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping: patience exhausted (%d). "
                    "Requesting training stop.",
                    self.patience,
                )
