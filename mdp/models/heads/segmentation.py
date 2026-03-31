"""SegmentationHead -- semantic segmentation 태스크 헤드."""

from __future__ import annotations

from torch import Tensor, nn

from mdp.models.heads.base import BaseHead


class SegmentationHead(BaseHead):
    """시맨틱 세그멘테이션 헤드.

    1x1 Conv2d로 feature map의 채널을 클래스 수로 변환한다.
    pixel-wise classification을 수행하므로 공간 해상도를 유지한다.

    Args:
        in_channels: 입력 feature map 채널 수.
        num_classes: 세그멘테이션 클래스 수.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        """세그멘테이션 logits를 반환한다.

        Args:
            features: (batch, in_channels, H, W).

        Returns:
            logits: (batch, num_classes, H, W).
        """
        return self.conv(features)
