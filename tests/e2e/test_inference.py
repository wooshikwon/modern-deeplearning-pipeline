"""E2E tests for batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mdp.serving.inference import run_batch_inference
from mdp.training.callbacks.base import BaseInferenceCallback
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def test_batch_inference_classification(tmp_path: Path) -> None:
    """Run batch inference with jsonl output and verify the file has content."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    output_path = tmp_path / "preds"
    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=output_path,
        output_format="jsonl",
        task="classification",
        device="cpu",
    )

    assert result_path.exists()
    assert result_path.suffix == ".jsonl"

    lines = result_path.read_text().strip().splitlines()
    # 3 batches * 4 samples = 12 records
    assert len(lines) == 12
    assert eval_results == {}


def test_batch_inference_formats(tmp_path: Path) -> None:
    """Test csv and parquet output formats."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=2, num_classes=2)
    loader = ListDataLoader(batches)

    for fmt, suffix in [("csv", ".csv"), ("parquet", ".parquet")]:
        output_path = tmp_path / f"preds_{fmt}"
        result_path, _ = run_batch_inference(
            model=model,
            dataloader=loader,
            output_path=output_path,
            output_format=fmt,
            task="classification",
            device="cpu",
        )
        assert result_path.exists()
        assert result_path.suffix == suffix
        assert result_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Inference callbacks (BaseInferenceCallback lifecycle)
# ---------------------------------------------------------------------------


class _HiddenStateCaptureCallback(BaseInferenceCallback):
    """Test callback: registers a forward hook to capture hidden activations."""

    def __init__(self, layer_name: str) -> None:
        self.layer_name = layer_name
        self.captured: list[torch.Tensor] = []
        self.setup_called = False
        self.teardown_called = False
        self._hook_handle: Any = None
        self._latest_activation: torch.Tensor | None = None

    def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
        self.setup_called = True
        target = dict(model.named_modules())[self.layer_name]
        self._hook_handle = target.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self._latest_activation = output.detach().cpu()

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
        if self._latest_activation is not None:
            self.captured.append(self._latest_activation)
            self._latest_activation = None

    def teardown(self, **kwargs) -> None:
        self.teardown_called = True
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


def test_inference_callback_lifecycle(tmp_path: Path) -> None:
    """BaseInferenceCallback의 setup/on_batch/teardown lifecycle이 올바르게 dispatch된다."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    num_batches = 3
    batches = make_vision_batches(num_batches=num_batches, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    cb = _HiddenStateCaptureCallback(layer_name="classifier")

    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb],
    )

    # Lifecycle 호출 확인
    assert cb.setup_called
    assert cb.teardown_called

    # 매 배치마다 on_batch가 호출되어 활성화를 캡처했는지
    assert len(cb.captured) == num_batches

    # 캡처된 텐서의 shape: classifier는 Linear(8, hidden_dim=16) → (batch_size, 16)
    for activation in cb.captured:
        assert activation.shape == (4, 16)

    # hook이 teardown에서 해제되었는지
    assert cb._hook_handle is None

    # 추론 결과도 정상 저장
    assert result_path.exists()


def test_inference_callback_teardown_on_error(tmp_path: Path) -> None:
    """추론 루프에서 에러가 발생해도 teardown은 실행된다."""

    class _TrackingCallback(BaseInferenceCallback):
        def __init__(self) -> None:
            self.teardown_called = False

        def teardown(self, **kwargs) -> None:
            self.teardown_called = True

    class _ExplodingModel(nn.Module):
        """2번째 배치에서 에러를 발생시키는 모델."""
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3 * 8 * 8, 2)
            self._call_count = 0

        def forward(self, batch: dict) -> dict[str, torch.Tensor]:
            self._call_count += 1
            if self._call_count >= 2:
                raise RuntimeError("deliberate test explosion")
            x = batch["pixel_values"].flatten(1)
            return {"logits": self.linear(x)}

    model = _ExplodingModel()
    batches = make_vision_batches(num_batches=3, batch_size=2, num_classes=2)
    loader = ListDataLoader(batches)
    cb = _TrackingCallback()

    try:
        run_batch_inference(
            model=model,
            dataloader=loader,
            output_path=tmp_path / "preds",
            output_format="jsonl",
            device="cpu",
            callbacks=[cb],
        )
    except RuntimeError:
        pass

    assert cb.teardown_called, "teardown must run even when inference loop raises"


def test_inference_callback_critical_propagates(tmp_path: Path) -> None:
    """critical=True인 콜백의 on_batch 예외는 전파된다."""
    import pytest

    class _CriticalFailCallback(BaseInferenceCallback):
        critical = True

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            raise ValueError("critical inference failure")

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=2, num_classes=2)
    loader = ListDataLoader(batches)

    with pytest.raises(ValueError, match="critical inference failure"):
        run_batch_inference(
            model=model,
            dataloader=loader,
            output_path=tmp_path / "preds",
            output_format="jsonl",
            device="cpu",
            callbacks=[_CriticalFailCallback()],
        )


def test_inference_callback_noncritical_swallowed(tmp_path: Path) -> None:
    """critical=False(기본)인 콜백의 on_batch 예외는 삼켜진다."""

    class _FailingCallback(BaseInferenceCallback):
        def __init__(self) -> None:
            self.teardown_called = False

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            raise RuntimeError("non-critical failure")

        def teardown(self, **kwargs) -> None:
            self.teardown_called = True

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=2, num_classes=2)
    loader = ListDataLoader(batches)
    cb = _FailingCallback()

    # Should not raise
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        device="cpu",
        callbacks=[cb],
    )

    assert result_path.exists()
    assert cb.teardown_called


def test_inference_callback_no_callbacks(tmp_path: Path) -> None:
    """callbacks=None일 때 기존 동작이 그대로 유지된다."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        device="cpu",
        callbacks=None,
    )

    assert result_path.exists()
    lines = result_path.read_text().strip().splitlines()
    assert len(lines) == 8
