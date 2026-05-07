"""BaseModel ABC — 모든 MDP 모델의 추상 기반 클래스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """MDP 모델의 추상 기반 클래스.

    모든 모델은 forward를 구현해야 한다. SFT loss는 recipe.loss external
    criterion 또는 forward output의 ``loss``로 제공한다.
    validation_step()은 선택적 검증 hook으로만 오버라이드한다.
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
                self._block_classes = set(bc)  # type: ignore[misc]
                return
            # HF 호환: _no_split_modules → _block_classes로 변환
            # _no_split_modules는 단축명("LlamaDecoderLayer")만 제공하므로
            # named_modules()를 스캔하여 전체 경로로 resolve한다.
            ns = getattr(child, "_no_split_modules", None)
            if ns:
                full_paths: set[str] = set()
                for short_name in ns:
                    resolved = False
                    for _, module in child.named_modules():
                        if type(module).__name__ == short_name:
                            cls = type(module)
                            full_paths.add(f"{cls.__module__}.{cls.__qualname__}")
                            resolved = True
                            break
                    if not resolved:
                        # named_modules() 스캔에서 해당 타입을 찾지 못한 경우
                        # (예: PEFT 래핑, lazy init, 합성 stub 등)
                        # short_name 자체를 fallback으로 사용한다.
                        # FSDP는 string 기반 match이므로 short_name만으로도 작동한다.
                        full_paths.add(short_name)
                if full_paths:
                    self._block_classes = full_paths  # type: ignore[misc]
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

    def extract_features_and_head(
        self,
        batch: dict[str, Tensor],
        layer_idx: int = -1,
    ) -> tuple[Tensor, Tensor]:
        """Algorithm이 ``needs_hidden_states=True``일 때 Trainer가 호출.

        Fused Linear Cross-Entropy 같은 memory-efficient loss 경로를 위해
        logits을 materialize하지 않고 hidden representation과 output head weight을
        반환한다.

        Returns:
            (hidden_states, output_head_weight) 튜플.
            - hidden_states: 지정된 representation layer의 출력.
              NLP causal LM: ``(B, S, H)``.
              Vision transformer: ``(B, N_patch [+CLS], H)``.
              Vision CNN: flatten된 feature ``(B, H)``.
            - output_head_weight: output projection의 weight matrix.
              NLP: ``lm_head.weight`` ``(V, H)``.
              Vision: classifier ``weight`` ``(C, H)``.

        기본 구현은 ``NotImplementedError``. BaseModel 서브클래스가
        이 메서드를 override하거나, HF/timm/torchvision 모델은
        RLTrainer의 framework dispatcher가 기본 구현을 제공한다.

        Args:
            batch: forward가 받는 것과 동일 형식의 입력 dict.
            layer_idx: 어느 layer의 hidden을 뽑을지. -1은 마지막 representation
                layer. HF 계열은 embedding layer가 index 0, 마지막 transformer
                block이 index -1이다.

        Raises:
            NotImplementedError: 서브클래스가 override하지 않은 경우.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}은 extract_features_and_head를 "
            "override하지 않았습니다. algorithm.needs_hidden_states=True와 "
            "함께 사용하려면 이 메서드를 구현하세요."
        )

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """선택적 검증 hook. 없으면 Trainer validation fallback을 사용한다."""
        raise NotImplementedError(
            f"{self.__class__.__name__}은 validation_step을 override하지 않았습니다."
        )

    def configure_optimizers(self) -> dict[str, Any] | None:
        """모델 전용 옵티마이저 설정. None이면 Recipe의 optimizer를 사용한다."""
        return None

    def load_from_export(self, artifact_dir: Path) -> None:
        """export() 역방향 — export 디렉토리에서 가중치를 로드한다.

        ``reconstruct_model()``이 ``export_info.json``을 감지하면 호출한다.
        기본 구현은 ``model.safetensors``를 ``load_state_dict``로 로드한다.

        **커스텀 export 구조(backbone+head 분리 등)는 반드시 오버라이드해야 한다.**

        오버라이드 예시 (backbone + value_head 분리 로드)::

            def load_from_export(self, artifact_dir: Path) -> None:
                from safetensors.torch import load_file
                import torch
                self.backbone.load_state_dict(
                    load_file(artifact_dir / "backbone" / "model.safetensors")
                )
                self.value_head.load_state_dict(
                    torch.load(artifact_dir / "value_head.pt",
                               map_location="cpu", weights_only=True)
                )
        """
        safetensors_path = Path(artifact_dir) / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file as _load_file
            self.load_state_dict(_load_file(safetensors_path))

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
