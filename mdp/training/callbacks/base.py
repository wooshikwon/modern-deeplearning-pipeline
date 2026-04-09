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
    콜백이 setup에서 등록하는 ``register_forward_hook`` 을 통해 이루어진다.
    """

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:  # noqa: ARG002
        """추론 시작 전 호출. 모델에 forward hook 등록, 내부 버퍼 준비."""

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:  # noqa: ARG002
        """매 배치 forward 후 호출. hook이 캡처한 활성화를 처리한다."""

    def teardown(self, **kwargs) -> None:  # noqa: ARG002
        """추론 완료 후 호출. 누적 결과 저장, hook 핸들 해제."""
