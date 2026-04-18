"""Base callback interface for training and inference hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn


class BaseCallback:
    """Base class for all training callbacks.

    Subclasses override specific hooks to inject behavior at each
    stage of the training loop.  All hooks accept arbitrary keyword
    arguments so that the trainer can forward extra context without
    breaking existing callbacks.
    """

    should_stop: bool = False
    critical: bool = False

    def on_train_start(self, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_epoch_start(self, epoch: int, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_batch_start(self, step: int, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass

    def on_epoch_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass

    def on_validation_start(self, epoch: int, **kwargs) -> None:  # noqa: ARG002
        pass

    def on_validation_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass

    def on_train_end(
        self,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:  # noqa: ARG002
        pass


class BaseInferenceCallback(BaseCallback):
    """추론 콜백 베이스. setup에서 model에 forward hook을 등록하고, on_batch에서 결과를 수집한다.

    학습 콜백(BaseCallback)과 동일한 critical 플래그를 상속한다.
    추론 루프는 hidden state나 attention을 직접 다루지 않는다 — 모든 내부 접근은
    콜백이 setup에서 등록하는 hook을 통해 이루어진다.

    Hook 종류
    ---------
    - ``register_forward_hook``: layer 출력을 **읽는다** (hidden state 추출, 활성화 분석).
    - ``register_forward_pre_hook``: layer 입력을 **수정한다** (activation steering, 벡터 주입).

    device_map 모델 (multi-GPU)
    ---------------------------
    ``device_map="auto"`` 등으로 여러 GPU에 분산된 모델에서도 hook이 정상 동작한다.
    accelerate는 모듈의 ``forward`` 메서드를 교체하여 디바이스 간 전송을 처리하지만,
    PyTorch의 사용자 hook은 ``__call__`` 레벨에서 실행되므로 간섭 없이 공존한다.

    실행 순서::

        module()
        ├─ PyTorch pre-hooks          ← 입력이 원래 디바이스에 있는 상태
        ├─ accelerate pre_forward     ← 입력을 실행 디바이스로 이동
        ├─ _old_forward               ← 실제 계산
        ├─ accelerate post_forward    ← 출력을 입력 디바이스로 복원
        └─ PyTorch post-hooks         ← 출력이 복원된 상태

    주의사항:

    1. **디바이스 매칭**: hook 안에서 텐서 간 연산 시 ``.to(target.device)`` 로 디바이스를
       맞춘다. 읽기 hook에서는 ``.detach().cpu()`` 로 CPU에 복사하면 안전하다.
    2. **dtype 매칭**: mixed precision 추론 시 모델 내부 텐서가 bf16/fp16일 수 있다.
       외부에서 주입하는 벡터도 ``.to(device=target.device, dtype=target.dtype)`` 로
       디바이스와 dtype을 함께 맞춰야 한다.
    3. **teardown 필수**: ``teardown()`` 에서 hook 핸들을 반드시 ``.remove()`` 한다.

    읽기 예시 (forward hook)::

        def setup(self, model, tokenizer=None, **kwargs):
            target = dict(model.named_modules())[self.layer_name]
            self._handle = target.register_forward_hook(self._capture)

        def _capture(self, module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self._latest = h.detach().cpu()

    쓰기 예시 (forward pre-hook, activation steering)::

        def setup(self, model, tokenizer=None, **kwargs):
            target = model.model.layers[self.layer_idx]
            self._handle = target.register_forward_pre_hook(self._steer)

        def _steer(self, module, args):
            hidden = args[0]
            vec = self.vector.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden + self.strength * vec,) + args[1:]
    """

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:  # noqa: ARG002
        """추론 시작 전 호출. 모델에 forward hook 등록, 내부 버퍼 준비.

        Parameters
        ----------
        model:
            추론에 사용되는 모델. device_map 모델이면 layer가 여러 GPU에 분산되어 있을 수
            있으나, ``model.named_modules()`` 로 모든 layer에 접근 가능하다.
        tokenizer:
            토크나이저. None 가능.
        """

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:  # noqa: ARG002
        """매 배치 forward 후 호출. hook이 캡처한 활성화를 처리한다.

        Parameters
        ----------
        batch_idx:
            현재 배치의 0-based 인덱스.
        batch:
            모델에 전달된 입력 배치 dict.
        outputs:
            모델의 정규화된 출력 dict.
        **kwargs:
            추가 컨텍스트. 현재 지원되는 키:

            - ``metadata`` (``list[dict] | None``): pretrained 분기에서 토큰화 전에
              추출된 원본 컬럼의 현재 배치 슬라이스. 각 dict는 한 샘플의 메타데이터
              (label, topic 등)를 담고 있다. artifact 분기이거나 메타데이터가 없으면 None.
        """

    is_intervention: bool = False

    def teardown(self, **kwargs) -> None:  # noqa: ARG002
        """추론 완료 후 호출. 누적 결과 저장, hook 핸들 해제."""


class BaseInterventionCallback(BaseInferenceCallback):
    """출력을 수정하는 callback. 이것이 없으면 inference 결과가 달라진다는 계약.

    - `is_intervention = True`가 서브클래스에서 자동 전파된다.
    - `metadata` 프로퍼티는 MLflow tag 적재용 정보를 반환한다. 구현 필수.
    - 관측 callback(BaseInferenceCallback 직계 서브클래스)은 `is_intervention = False`.
    """

    is_intervention: bool = True

    @property
    def metadata(self) -> dict[str, Any]:
        """MLflow tag로 적재될 개입 metadata. 최소 `type` 키 필수.

        예:
            {"type": "ResidualAdd", "target_layers": [20, 21],
             "vector_sha256": "abc123...", "strength": 1.0}

        MLflow tag 키는 호출자(U6)가 `intervention.{i}.{k}`로 프리픽스를 붙인다.
        값은 str/int/float/list 중 하나여야 하며, list는 JSON 직렬화된다.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.metadata must be implemented for MLflow logging."
        )
