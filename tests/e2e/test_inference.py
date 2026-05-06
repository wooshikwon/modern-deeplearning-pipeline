"""E2E tests for batch inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from mdp.callbacks.inference import DefaultOutputCallback
from mdp.models.base import BaseModel
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
    output_cb = DefaultOutputCallback(
        output_path=output_path, output_format="jsonl", task="classification",
    )
    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=output_path,
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    assert result_path.suffix == ".jsonl"

    lines = result_path.read_text().strip().splitlines()
    # 3 batches * 4 samples = 12 records
    assert len(lines) == 12
    assert eval_results == {}


def test_batch_inference_hf_style_forward(tmp_path: Path) -> None:
    """Non-BaseModel modules are called with ``model(**batch)``."""

    class _HFStyleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.proj = nn.Linear(4, 2)

        def forward(self, input_ids=None, attention_mask=None):
            hidden = self.embed(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1)
            return {"logits": self.proj(hidden[:, -1])}

    model = _HFStyleModel()
    loader = ListDataLoader([
        {
            "input_ids": torch.tensor([[1, 2, 3], [3, 2, 1]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
    ])

    class _Inspector(BaseInferenceCallback):
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            self.shapes.append(tuple(outputs["logits"].shape))

    inspector = _Inspector()
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[inspector],
    )

    assert result_path is None
    assert inspector.shapes == [(2, 2)]


def test_batch_inference_formats(tmp_path: Path) -> None:
    """Test csv and parquet output formats."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=2, num_classes=2)
    loader = ListDataLoader(batches)

    for fmt, suffix in [("csv", ".csv"), ("parquet", ".parquet")]:
        output_path = tmp_path / f"preds_{fmt}"
        output_cb = DefaultOutputCallback(
            output_path=output_path, output_format=fmt, task="classification",
        )
        result_path, _ = run_batch_inference(
            model=model,
            dataloader=loader,
            output_path=output_path,
            output_format=fmt,
            task="classification",
            device="cpu",
            callbacks=[output_cb],
        )
        assert result_path is not None
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
    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="jsonl", task="classification",
    )

    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb, output_cb],
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
    assert result_path is not None
    assert result_path.exists()


def test_inference_callback_teardown_on_error(tmp_path: Path) -> None:
    """추론 루프에서 에러가 발생해도 teardown은 실행된다."""

    class _TrackingCallback(BaseInferenceCallback):
        def __init__(self) -> None:
            self.teardown_called = False

        def teardown(self, **kwargs) -> None:
            self.teardown_called = True

    class _ExplodingModel(BaseModel):
        """2번째 배치에서 에러를 발생시키는 모델."""
        _block_classes = None

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

        def training_step(self, batch: dict) -> torch.Tensor:
            return self.forward(batch)["logits"].mean()

        def validation_step(self, batch: dict) -> dict[str, float]:
            return {"val_loss": 0.0}

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
    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="jsonl", task="classification",
    )

    # Should not raise
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        device="cpu",
        callbacks=[cb, output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    assert cb.teardown_called


def test_inference_callback_no_callbacks(tmp_path: Path) -> None:
    """callbacks=None이면 DefaultOutputCallback이 없으므로 result_path가 None이다."""
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

    # DefaultOutputCallback이 없으면 출력 파일이 생성되지 않는다
    assert result_path is None


# ---------------------------------------------------------------------------
# Callback-only mode (no DefaultOutputCallback → no postprocess → memory savings)
# ---------------------------------------------------------------------------


def test_callback_only_mode_no_output_file(tmp_path: Path) -> None:
    """사용자 콜백만 등록하면 출력 파일 없이 추론이 완료된다."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    cb = _HiddenStateCaptureCallback(layer_name="classifier")

    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        device="cpu",
        callbacks=[cb],
    )

    # 콜백은 정상 동작
    assert cb.setup_called
    assert cb.teardown_called
    assert len(cb.captured) == 3

    # DefaultOutputCallback 없으므로 출력 파일 없음
    assert result_path is None
    assert not (tmp_path / "preds.jsonl").exists()


def test_callback_only_mode_no_softmax_on_logits(tmp_path: Path) -> None:
    """콜백 전용 모드에서 logits에 softmax가 적용되지 않는다.

    DefaultOutputCallback이 없으면 _postprocess(softmax+argmax+numpy)가 실행되지 않으므로,
    LM의 거대한 logits(vocab_size=151K)에 softmax를 적용하는 25GB 메모리 낭비가 없다.
    이 테스트는 콜백이 받는 outputs가 raw logits(softmax 미적용)인지 검증한다.
    """

    class _OutputInspector(BaseInferenceCallback):
        """on_batch에서 outputs의 logits 값 범위를 기록하는 콜백."""

        def __init__(self) -> None:
            self.logits_ranges: list[tuple[float, float]] = []

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            if "logits" in outputs:
                logits = outputs["logits"]
                self.logits_ranges.append((logits.min().item(), logits.max().item()))

    # num_classes=10으로 설정 — 클래스가 많아야 random init에서 logits가
    # [0,1] 범위를 확실히 벗어나 raw logits임을 검증할 수 있다.
    model = TinyVisionModel(num_classes=10, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=4, num_classes=10)
    loader = ListDataLoader(batches)

    inspector = _OutputInspector()

    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        device="cpu",
        callbacks=[inspector],
    )

    # logits는 raw 값 — softmax가 적용되지 않았으므로 [0,1] 범위가 아니다
    assert len(inspector.logits_ranges) == 2
    for lo, hi in inspector.logits_ranges:
        # raw logits는 음수를 포함할 수 있고 1을 초과할 수 있다
        # softmax가 적용되었다면 모든 값이 [0, 1]이어야 한다
        assert lo < 0 or hi > 1, (
            f"logits range [{lo}, {hi}] looks like softmax was applied"
        )


def test_callback_only_with_metadata(tmp_path: Path) -> None:
    """콜백 전용 모드에서 metadata가 올바르게 전달된다."""

    class _MetadataCollector(BaseInferenceCallback):
        def __init__(self) -> None:
            self.collected: list = []

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            meta = kwargs.get("metadata")
            if meta is not None:
                self.collected.extend(meta)

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    collector = _MetadataCollector()
    fake_metadata = [{"label": f"item_{i}"} for i in range(8)]

    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        device="cpu",
        callbacks=[collector],
        metadata=fake_metadata,
    )

    assert len(collector.collected) == 8
    assert collector.collected[0] == {"label": "item_0"}
    assert collector.collected[7] == {"label": "item_7"}
