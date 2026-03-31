"""ClassificationHead — 분류 태스크 헤드."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mdp.models.heads.base import BaseHead


class ClassificationHead(BaseHead):
    """분류 헤드.

    3D 입력 (batch, seq_len, hidden_dim)이면 pooling을 적용하고,
    2D 입력 (batch, hidden_dim)이면 pooling을 생략한다.

    Args:
        num_classes: 분류 클래스 수.
        hidden_dim: 입력 feature 차원.
        dropout: 드롭아웃 비율.
        pooling: 풀링 방식 — "cls", "mean", "max".
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean", "max"):
            raise ValueError(f"지원하지 않는 pooling 방식: {pooling!r}")

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """분류 logits를 반환한다.

        Args:
            features: (batch, hidden_dim) 또는 (batch, seq_len, hidden_dim).

        Returns:
            logits: (batch, num_classes).
        """
        if features.ndim == 3:
            features = self._pool(features)

        features = self.dropout(features)
        return self.classifier(features)

    def _pool(self, features: Tensor) -> Tensor:
        """3D 텐서를 2D로 축소한다."""
        if self.pooling == "cls":
            return features[:, 0]
        if self.pooling == "mean":
            return features.mean(dim=1)
        # max
        return torch.max(features, dim=1).values
