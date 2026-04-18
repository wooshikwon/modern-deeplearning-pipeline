"""ResidualAdd -- 특정 transformer layer의 residual stream에 벡터를 더하는 개입 콜백."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mdp.callbacks.base import BaseInterventionCallback


class ResidualAdd(BaseInterventionCallback):
    """특정 transformer layer의 residual stream에 벡터를 더한다.

    Parameters
    ----------
    target_layers:
        hook을 등록할 layer 인덱스 리스트. ``model.model.layers[i]`` 에 해당.
    vector_path:
        .pt 파일 경로. shape는 ``(hidden_dim,)`` 이거나 ``(num_layers, hidden_dim)``.
        shape가 ``(num_layers, hidden_dim)`` 이면 ``target_layers[i]`` 번째 슬라이스를
        각 layer에 적용한다.
    strength:
        벡터 배율 (스칼라).
    """

    def __init__(
        self,
        target_layers: list[int],
        vector_path: str | Path,
        strength: float = 1.0,
    ) -> None:
        self.target_layers = list(target_layers)
        self.vector_path = Path(vector_path)
        self.strength = float(strength)

        self._vector: torch.Tensor | None = None
        self._vector_sha256: str = ""
        self._handles: list[Any] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        """target_layers 각각에 forward pre-hook을 등록한다."""
        raw = torch.load(self.vector_path, map_location="cpu", weights_only=True)
        self._vector_sha256 = _sha256_tensor(raw)
        self._vector = raw  # keep on CPU; hook moves per-call

        layers = _get_layers(model)

        for rank, layer_idx in enumerate(self.target_layers):
            layer = layers[layer_idx]

            if self._vector.dim() == 2:
                # shape: (num_layers, hidden_dim) — each target gets its own slice
                vec_slice = self._vector[rank].clone()
            else:
                # shape: (hidden_dim,) — same vector for all targets
                vec_slice = self._vector.clone()

            handle = layer.register_forward_pre_hook(
                _make_steer_hook(vec_slice, self.strength)
            )
            self._handles.append(handle)

    def teardown(self, **kwargs) -> None:
        """등록된 모든 hook handle을 제거한다."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        if not self._vector_sha256:
            raise RuntimeError(
                "ResidualAdd.metadata accessed before setup(). "
                "Call setup(model) first to load the vector and register hooks."
            )
        return {
            "type": "ResidualAdd",
            "target_layers": self.target_layers,
            "vector_sha256": self._vector_sha256,
            "strength": self.strength,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_steer_hook(vec: torch.Tensor, strength: float):
    """pre-hook closure: hidden state에 strength * vec을 더한다."""

    def _steer(module, args):  # noqa: ARG001
        hidden = args[0]
        v = vec.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + strength * v
        return (steered,) + args[1:]

    return _steer


def _get_layers(model: nn.Module) -> list[nn.Module]:
    """모델에서 transformer layer 리스트를 추출한다."""
    # HuggingFace 모델의 공통 경로를 순서대로 시도
    for attr_path in ("model.layers", "transformer.h", "transformer.blocks", "layers"):
        obj = model
        found = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        if found and hasattr(obj, "__len__"):
            return list(obj)
    raise AttributeError(
        f"Cannot locate transformer layers on {type(model).__name__}. "
        "Expected model.model.layers, model.transformer.h, or similar."
    )


def _sha256_tensor(t: torch.Tensor) -> str:
    """텐서의 SHA-256 hex digest를 반환한다."""
    data = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()
