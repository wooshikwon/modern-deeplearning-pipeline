"""DualEncoderHead -- 이중 인코더 (CLIP/SigLIP) 태스크 헤드."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from mdp.models.heads.base import BaseHead


class DualEncoderHead(BaseHead):
    """이중 인코더 헤드.

    이미지와 텍스트 각각의 feature를 공유 임베딩 공간으로 투영한다.
    CLIP, SigLIP 같은 contrastive learning 모델에서 사용한다.

    Args:
        embed_dim: 입력 임베딩 차원.
        projection_dim: 투영 공간 차원.
    """

    def __init__(self, embed_dim: int, projection_dim: int = 512) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.image_projection = nn.Linear(embed_dim, projection_dim)
        self.text_projection = nn.Linear(embed_dim, projection_dim)

    def forward(self, features: Tensor) -> Tensor:
        """단일 feature를 image projection으로 투영한다.

        Args:
            features: (batch, embed_dim).

        Returns:
            projected: (batch, projection_dim) L2 정규화된 임베딩.
        """
        projected = self.image_projection(features)
        return F.normalize(projected, dim=-1)

    def forward_pair(
        self,
        image_features: Tensor,
        text_features: Tensor,
    ) -> dict[str, Tensor]:
        """이미지-텍스트 feature 쌍을 투영한다.

        Args:
            image_features: (batch, embed_dim).
            text_features: (batch, embed_dim).

        Returns:
            L2 정규화된 임베딩 쌍:
            {"image_embeds": (batch, projection_dim),
             "text_embeds": (batch, projection_dim)}.
        """
        image_embeds = F.normalize(self.image_projection(image_features), dim=-1)
        text_embeds = F.normalize(self.text_projection(text_features), dim=-1)
        return {"image_embeds": image_embeds, "text_embeds": text_embeds}
