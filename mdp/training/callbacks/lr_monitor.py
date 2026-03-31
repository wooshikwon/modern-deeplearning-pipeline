"""Learning rate monitor callback."""

from __future__ import annotations

import logging

from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class LearningRateMonitor(BaseCallback):
    """Log current learning rate(s) at a configurable interval.

    Parameters
    ----------
    logging_interval:
        ``"epoch"`` (default) or ``"step"``.
    """

    def __init__(self, logging_interval: str = "epoch") -> None:
        if logging_interval not in ("epoch", "step"):
            msg = f"logging_interval must be 'epoch' or 'step', got '{logging_interval}'"
            raise ValueError(msg)
        self.logging_interval = logging_interval

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_lrs(optimizer) -> dict[str, float]:
        """Return a dict mapping param-group index to its learning rate."""
        lrs: dict[str, float] = {}
        for idx, group in enumerate(optimizer.param_groups):
            key = f"lr/group_{idx}"
            lrs[key] = group["lr"]
        return lrs

    @staticmethod
    def _log_to_mlflow(lrs: dict[str, float], step: int) -> None:
        try:
            import mlflow  # noqa: PLC0415

            mlflow.log_metrics(lrs, step=step)
        except ImportError:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("MLflow logging failed, skipping.", exc_info=True)

    def _log_lrs(self, optimizer, step: int) -> None:
        if optimizer is None:
            return
        lrs = self._extract_lrs(optimizer)
        for key, value in lrs.items():
            logger.info("Step %d — %s = %.8f", step, key, value)
        self._log_to_mlflow(lrs, step)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if self.logging_interval == "step":
            self._log_lrs(kwargs.get("optimizer"), step)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if self.logging_interval == "epoch":
            self._log_lrs(kwargs.get("optimizer"), epoch)
