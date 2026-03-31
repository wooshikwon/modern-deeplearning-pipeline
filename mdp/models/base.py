"""BaseModel ABC — 모든 MDP 모델의 추상 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """MDP 모델의 추상 기반 클래스.

    모든 모델은 forward, training_step, validation_step을 구현해야 한다.
    generate()는 자기회귀 모델만 오버라이드한다.
    """

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """순전파. batch dict를 받아 출력 dict를 반환한다."""
        ...

    def generate(self, batch: dict[str, Tensor], **kwargs: Any) -> dict[str, Any]:
        """자기회귀 생성. 지원하지 않는 모델은 NotImplementedError를 발생시킨다."""
        raise NotImplementedError("이 모델은 generate()를 지원하지 않습니다")

    @abstractmethod
    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        """학습 스텝. 스칼라 loss를 반환한다."""
        ...

    @abstractmethod
    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """검증 스텝. 메트릭 이름-값 dict를 반환한다."""
        ...

    def configure_optimizers(self) -> dict[str, Any] | None:
        """모델 전용 옵티마이저 설정. None이면 Recipe의 optimizer를 사용한다."""
        return None
