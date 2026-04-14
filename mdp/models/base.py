"""BaseModel ABC — 모든 MDP 모델의 추상 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """MDP 모델의 추상 기반 클래스.

    모든 모델은 forward, training_step, validation_step을 구현해야 한다.
    generate()는 자기회귀 모델만 오버라이드한다.

    ``_block_classes`` 는 필수 선언이다. 모델의 반복 블록 클래스 이름의
    ``set[str]`` 을 지정하거나, 반복 블록이 없으면 ``None`` 을 선언한다.
    FSDP wrap policy, gradient checkpointing 등에서 사용된다.
    """

    _block_classes: ClassVar[set[str] | None]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_block_classes" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}은 _block_classes를 선언해야 합니다. "
                "모델의 반복 블록 클래스 이름의 set을 지정하거나, "
                "반복 블록이 없으면 None을 선언하세요."
            )

    def _inherit_block_classes(self) -> None:
        """자식 모듈의 _no_split_modules(HF) 또는 _block_classes(MDP)에서 상속.

        HF backbone을 감싸는 커스텀 모델에서 ``__init__`` 마지막에 호출하면
        backbone의 블록 클래스 정보를 자동으로 가져온다.
        """
        for child in self.children():
            # MDP 자체 계약 우선
            bc = getattr(child, "_block_classes", None)
            if bc:
                self._block_classes = set(bc)
                return
            # HF 호환: _no_split_modules → _block_classes로 변환
            ns = getattr(child, "_no_split_modules", None)
            if ns:
                self._block_classes = set(ns)
                return

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
