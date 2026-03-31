"""BaseHead ABC — 모든 태스크 헤드의 추상 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseHead(nn.Module, ABC):
    """태스크 헤드 추상 기반 클래스.

    백본의 feature를 태스크별 출력으로 변환한다.
    """

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """features를 받아 태스크별 출력 텐서를 반환한다."""
        ...
