"""Forward-call normalization for MDP and HuggingFace-style models."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from mdp.models.base import BaseModel


def make_forward_fn(model: nn.Module) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a normalized ``forward(batch)`` callable for model consumers."""
    signature_model = getattr(model, "module", model)
    if isinstance(signature_model, BaseModel):
        def _base_forward(batch: dict[str, Any]) -> dict[str, Any]:
            outputs = model(batch)
            return normalize_model_output(outputs)
        return _base_forward

    single_arg = _single_required_arg_name(signature_model)
    if single_arg == "batch":
        def _batch_forward(batch: dict[str, Any]) -> dict[str, Any]:
            outputs = model(batch)
            return normalize_model_output(outputs)
        return _batch_forward
    if single_arg is not None:
        def _single_tensor_forward(batch: dict[str, Any]) -> dict[str, Any]:
            inputs = _extract_single_tensor_input(batch, preferred_key=single_arg)
            outputs = model(inputs)
            return normalize_model_output(outputs)
        return _single_tensor_forward

    def _kwarg_forward(batch: dict[str, Any]) -> dict[str, Any]:
        outputs = model(**batch)
        return normalize_model_output(outputs)

    return _kwarg_forward


def _single_required_arg_name(model: nn.Module) -> str | None:
    """Return the sole required forward arg name, if the model has one."""
    try:
        parameters = list(inspect.signature(model.forward).parameters.values())
    except (TypeError, ValueError):
        return None
    required = [
        p for p in parameters
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
    if len(required) == 1 and not has_var_kwargs:
        return required[0].name
    return None


def _extract_single_tensor_input(
    batch: dict[str, Any],
    *,
    preferred_key: str,
) -> Any:
    """Select the tensor input for raw single-arg vision models."""
    if preferred_key in batch:
        return batch[preferred_key]
    for key in ("pixel_values", "input_ids", "features", "x"):
        if key in batch:
            return batch[key]
    raise ValueError(
        "single-argument model forward requires one of: "
        f"{preferred_key!r}, 'pixel_values', 'input_ids', 'features', or 'x'"
    )


def normalize_model_output(outputs: object) -> dict[str, Any]:
    """Normalize common PyTorch/HuggingFace outputs to a tensor dict."""
    if isinstance(outputs, dict):
        return outputs
    if isinstance(outputs, Tensor):
        return {"logits": outputs}
    result: dict[str, Any] = {}
    if hasattr(outputs, "logits") and outputs.logits is not None:
        result["logits"] = outputs.logits
        if hasattr(outputs, "loss") and isinstance(outputs.loss, Tensor):
            result["loss"] = outputs.loss
        if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
            result["boxes"] = outputs.pred_boxes
        return result
    if hasattr(outputs, "loss") and isinstance(outputs.loss, Tensor):
        result["loss"] = outputs.loss
        return result
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return {"last_hidden_state": outputs.last_hidden_state}
    if hasattr(outputs, "keys"):
        return {k: v for k, v in outputs.items() if isinstance(v, Tensor)}
    return {"output": outputs}
