"""Rich-based progress bar callback."""

from __future__ import annotations

import logging
from typing import Any

from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class ProgressBar(BaseCallback):
    """Display a Rich progress bar during training.

    Rich is imported lazily so the callback can be instantiated in
    environments where Rich is not installed — it will simply be a
    no-op in that case.
    """

    def __init__(self) -> None:
        self._progress: Any | None = None
        self._task_id: Any | None = None
        self._rich_available: bool = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_progress(self):
        """Lazily create and return a Rich Progress instance."""
        if not self._rich_available:
            return None
        if self._progress is not None:
            return self._progress
        try:
            from rich.progress import (  # noqa: PLC0415
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[metrics]}"),
            )
        except ImportError:
            self._rich_available = False
            logger.debug("Rich not installed; ProgressBar is a no-op.")
            return None
        return self._progress

    @staticmethod
    def _format_metrics(metrics: dict[str, float] | None) -> str:
        if not metrics:
            return ""
        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_train_start(self, **kwargs) -> None:
        progress = self._get_progress()
        if progress is None:
            return
        total_steps = kwargs.get("total_steps")
        self._task_id = progress.add_task(
            "Training",
            total=total_steps,
            metrics="",
        )
        progress.start()

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        progress = self._get_progress()
        if progress is None or self._task_id is None:
            return
        progress.update(
            self._task_id,
            advance=1,
            metrics=self._format_metrics(metrics),
        )

    def on_train_end(
        self,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
