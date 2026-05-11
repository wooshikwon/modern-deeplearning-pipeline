"""Small safetensors helpers shared by checkpoint save/load paths."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def save_module(model: nn.Module, path: str | Path) -> None:
    """Save a module while preserving safetensors tied-weight metadata."""
    from safetensors.torch import save_model

    save_model(model, str(path))


def load_module(model: nn.Module, path: str | Path) -> None:
    """Load a safetensors module checkpoint, including tied-weight metadata."""
    from safetensors.torch import load_model

    load_model(model, str(path))


def save_state_dict(state_dict: dict[str, torch.Tensor], path: str | Path) -> None:
    """Save a full state dict, cloning only when safetensors rejects shared tensors."""
    from safetensors.torch import save_file

    try:
        save_file(state_dict, str(path))
    except RuntimeError as exc:
        if "share memory" not in str(exc):
            raise
        cloned = {
            name: tensor.detach().clone()
            for name, tensor in state_dict.items()
        }
        save_file(cloned, str(path))
