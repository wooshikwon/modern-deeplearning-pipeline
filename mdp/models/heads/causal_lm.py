"""CausalLMHead — 자기회귀 언어 모델 헤드."""

from __future__ import annotations

from torch import Tensor, nn

from mdp.models.heads.base import BaseHead


class CausalLMHead(BaseHead):
    """자기회귀 언어 모델 헤드.

    hidden_dim → vocab_size 선형 변환 (bias 없음).

    Args:
        hidden_dim: 입력 hidden 차원.
        vocab_size: 어휘 크기.
    """

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        """언어 모델 logits를 반환한다.

        Args:
            features: (batch, seq_len, hidden_dim).

        Returns:
            logits: (batch, seq_len, vocab_size).
        """
        return self.lm_head(features)
