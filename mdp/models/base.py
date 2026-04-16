"""BaseModel ABC — 모든 MDP 모델의 추상 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
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
            # _no_split_modules는 단축명("LlamaDecoderLayer")만 제공하므로
            # named_modules()를 스캔하여 전체 경로로 resolve한다.
            ns = getattr(child, "_no_split_modules", None)
            if ns:
                full_paths: set[str] = set()
                for short_name in ns:
                    for _, module in child.named_modules():
                        if type(module).__name__ == short_name:
                            cls = type(module)
                            full_paths.add(f"{cls.__module__}.{cls.__qualname__}")
                            break
                if full_paths:
                    self._block_classes = full_paths
                    return

    # ------------------------------------------------------------------ #
    #  Gradient Checkpointing — HF 백본 자동 위임                        #
    #  직계 자식 중 gradient_checkpointing_enable을 가진 첫 번째 모듈에   #
    #  위임한다. _inherit_block_classes()와 동일한 탐색 패턴.             #
    #  덕분에 어떤 HF 백본을 감싸더라도 Trainer GC 설정이 자동 적용된다.  #
    # ------------------------------------------------------------------ #

    def enable_input_require_grads(self) -> None:
        """LoRA+GC 조합에서 input이 requires_grad=False여도 grad가 흐르게 한다.

        Trainer가 FSDP wrap 전에 호출. HF 백본에 위임한다.
        직계 자식에서 해당 인터페이스를 찾아 최초 1회만 호출한다.
        """
        for child in self.children():
            if hasattr(child, "enable_input_require_grads"):
                child.enable_input_require_grads()
                return

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict | None = None,
    ) -> None:
        """Trainer가 FSDP wrap 전에 호출하는 GC 활성화 인터페이스.

        직계 자식 중 HF PreTrainedModel을 찾아 위임한다.
        커스텀 BaseModel이 어떤 HF 백본을 감싸든 자동으로 동작한다.
        subclass에서 오버라이드할 필요 없음.
        """
        for child in self.children():
            if hasattr(child, "gradient_checkpointing_enable"):
                if gradient_checkpointing_kwargs is not None:
                    child.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                    )
                else:
                    child.gradient_checkpointing_enable()
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

    def export(self, output_dir: Path) -> None:
        """모델을 output_dir에 export한다.

        ``mdp export`` 가 호출한다. 기본 구현은 safetensors 단일 파일로 저장한다.

        **커스텀 구조(backbone + head 등)는 반드시 오버라이드해야 한다.**

        오버라이드 예시 (backbone + value_head 분리 저장)::

            def export(self, output_dir: Path) -> None:
                import torch
                self.backbone.save_pretrained(output_dir / "backbone")
                torch.save(self.value_head.state_dict(), output_dir / "value_head.pt")

        로딩 측 인터페이스(``CriticValueModel.__init__``의
        ``pretrained``/``value_head_checkpoint`` 등)는 모델 설계에 따라
        대응되어야 한다.
        """
        if hasattr(self, "save_pretrained"):
            self.save_pretrained(output_dir)
        else:
            from safetensors.torch import save_file as _save_file
            _save_file(self.state_dict(), Path(output_dir) / "model.safetensors")
