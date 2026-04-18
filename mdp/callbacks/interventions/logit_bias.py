"""LogitBias -- 마지막 layer output(logits)에 토큰별 bias를 더하는 개입 콜백."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mdp.callbacks.base import BaseInterventionCallback


class LogitBias(BaseInterventionCallback):
    """마지막 layer output(logits)에 토큰별 bias를 더한다.

    Parameters
    ----------
    token_biases:
        ``{token_id: bias_value}`` dict. token_id는 어휘 인덱스, bias_value는 float.
    """

    def __init__(self, token_biases: dict[int, float]) -> None:
        self.token_biases = dict(token_biases)
        self._handle: Any = None
        self._bias_tensor: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        """모델의 LM head(또는 최종 선형 레이어)에 forward hook을 등록한다."""
        lm_head = _get_lm_head(model)

        # bias_tensor는 sparse 표현으로 CPU에 보관; hook 내에서 이동
        # vocab_size는 hook 실행 시 logits shape에서 자동 결정
        ids = torch.tensor(list(self.token_biases.keys()), dtype=torch.long)
        vals = torch.tensor(list(self.token_biases.values()), dtype=torch.float32)
        self._bias_ids = ids
        self._bias_vals = vals

        self._handle = lm_head.register_forward_hook(self._apply_bias)

    def teardown(self, **kwargs) -> None:
        """등록된 hook handle을 제거한다."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def _apply_bias(self, module, input, output):  # noqa: ARG002
        """logit output에 token_biases를 더한다."""
        logits = output[0] if isinstance(output, tuple) else output

        ids = self._bias_ids.to(device=logits.device)
        vals = self._bias_vals.to(device=logits.device, dtype=logits.dtype)

        # logits shape: (batch, seq_len, vocab) or (batch, vocab)
        # Clone before in-place mutation to avoid modifying the original tensor.
        # ResidualAdd uses hidden + strength * v (creates new tensor) — symmetric.
        logits = logits.clone()
        logits[..., ids] = logits[..., ids] + vals

        if isinstance(output, tuple):
            return (logits,) + output[1:]
        return logits

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "type": "LogitBias",
            "num_biased_tokens": len(self.token_biases),
            "bias_sum": float(sum(self.token_biases.values())),
        }


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _get_lm_head(model: nn.Module) -> nn.Module:
    """모델에서 LM head 모듈을 찾는다."""
    for attr in ("lm_head", "cls", "head", "output_projection"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        f"Cannot locate LM head on {type(model).__name__}. "
        "Expected model.lm_head or similar."
    )
