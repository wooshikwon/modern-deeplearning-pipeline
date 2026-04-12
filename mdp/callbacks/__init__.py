"""Shared callback base classes for training and inference."""

from mdp.callbacks.base import BaseCallback, BaseInferenceCallback
from mdp.callbacks.inference import DefaultOutputCallback

__all__ = ["BaseCallback", "BaseInferenceCallback", "DefaultOutputCallback"]
