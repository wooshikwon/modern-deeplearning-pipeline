"""Forward-call normalization for MDP and HuggingFace-style models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from mdp.models.base import BaseModel


def make_forward_fn(model: nn.Module) -> Callable[[dict[str, Tensor]], dict[str, Tensor]]:
    """Build a normalized ``forward(batch)`` callable for model consumers."""
    if isinstance(model, BaseModel):
        def _base_forward(batch: dict[str, Tensor]) -> dict[str, Tensor]:
            return model(batch)
        return _base_forward

    def _kwarg_forward(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = model(**batch)
        return normalize_model_output(outputs)

    return _kwarg_forward


def normalize_model_output(outputs: object) -> dict[str, Tensor]:
    """Normalize common PyTorch/HuggingFace outputs to a tensor dict."""
    if isinstance(outputs, dict):
        return outputs
    if isinstance(outputs, Tensor):
        return {"logits": outputs}
    if hasattr(outputs, "logits") and outputs.logits is not None:
        result: dict[str, Tensor] = {"logits": outputs.logits}
        if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
            result["boxes"] = outputs.pred_boxes
        return result
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return {"last_hidden_state": outputs.last_hidden_state}
    if hasattr(outputs, "keys"):
        return {k: v for k, v in outputs.items() if isinstance(v, Tensor)}
    return {"output": outputs}
