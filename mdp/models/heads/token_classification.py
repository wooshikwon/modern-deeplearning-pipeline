"""TokenClassificationHead -- 토큰 분류 태스크 헤드."""

from __future__ import annotations

from torch import Tensor, nn

from mdp.models.heads.base import BaseHead


class TokenClassificationHead(BaseHead):
    """토큰 분류 헤드.

    모든 토큰 위치에 대해 독립적으로 분류를 수행한다.
    ClassificationHead와 달리 pooling 없이 시퀀스 차원을 유지한다.

    Args:
        hidden_dim: 입력 hidden 차원.
        num_classes: 분류 클래스 수 (NER 태그 수 등).
        dropout: 드롭아웃 비율.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """토큰별 분류 logits를 반환한다.

        Args:
            features: (batch, seq_len, hidden_dim).

        Returns:
            logits: (batch, seq_len, num_classes).
        """
        features = self.dropout(features)
        return self.classifier(features)
