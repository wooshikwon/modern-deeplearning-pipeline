"""Training public API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdp.training.trainer import Trainer


def __getattr__(name: str):
    if name == "Trainer":
        from mdp.training.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Trainer"]
