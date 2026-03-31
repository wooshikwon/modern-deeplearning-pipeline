"""DetectionHead — 객체 탐지 태스크 헤드."""

from __future__ import annotations

from torch import Tensor, nn

from mdp.models.heads.base import BaseHead


class DetectionHead(BaseHead):
    """객체 탐지 헤드.

    Conv2d 기반. 앵커 수 * (5 + num_classes) 채널을 출력한다.
    5 = 4 (bbox coordinates) + 1 (objectness).

    Args:
        in_channels: 입력 feature map 채널 수.
        num_classes: 탐지 클래스 수.
        num_anchors: 앵커 수.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.out_channels = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """탐지 출력을 반환한다.

        Args:
            features: (batch, in_channels, H, W).

        Returns:
            output: (batch, out_channels, H, W).
        """
        return self.conv(features)
