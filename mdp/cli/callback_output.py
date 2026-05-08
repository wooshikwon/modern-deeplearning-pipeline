"""Callback observer output helpers for CLI training commands."""

from __future__ import annotations

from typing import Any


def print_callbacks_log(
    cb_instances: list,
    settings: Any,
) -> None:
    """Print applied callbacks and auto-promoted training callbacks."""
    from mdp.cli.output import is_json_mode
    from mdp.runtime.worker import is_main_process

    if is_json_mode() or not is_main_process():
        return

    from mdp.callbacks.base import BaseInterventionCallback

    lines = []
    for cb in cb_instances:
        tag = "[Int]" if isinstance(cb, BaseInterventionCallback) else "[Obs]"
        cb_name = type(cb).__name__
        attrs = []
        if hasattr(cb, "monitor"):
            attrs.append(f"monitor={cb.monitor}")
        detail = f" ({', '.join(attrs)})" if attrs else ""
        lines.append(f"  {tag} {cb_name}{detail}")

    auto_lines = []
    training = settings.recipe.training
    if training.early_stopping is not None:
        es = training.early_stopping
        auto_lines.append(
            f"  EarlyStopping (monitor={es.monitor}, patience={es.patience})"
        )
    if training.ema is not None:
        ema = training.ema
        auto_lines.append(f"  EMA (decay={ema.decay})")

    if lines:
        print("Applied callbacks:")
        for line in lines:
            print(line)
    if auto_lines:
        print("Auto-promoted from training.*:")
        for line in auto_lines:
            print(line)
