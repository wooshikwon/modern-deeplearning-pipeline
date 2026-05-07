"""forward 어댑터 (_make_forward_fn) 테스트.

BaseModel과 HF-style 모델 양쪽 경로를 검증한다:
- BaseModel: model(batch) 호출, dict 출력 그대로 반환
- HF 모델: model(**batch) 호출, ModelOutput → dict 정규화
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mdp.callbacks.inference import DefaultOutputCallback
from mdp.serving.inference import _make_forward_fn, run_batch_inference
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


# ---------------------------------------------------------------------------
# Simulated HF ModelOutput (transformers를 import하지 않고 프로토콜을 재현)
# ---------------------------------------------------------------------------


class _FakeModelOutput:
    """HuggingFace ModelOutput의 최소 프로토콜을 재현한다.

    실제 ModelOutput은 OrderedDict 서브클래스이며 attribute access와
    dict-like access를 모두 지원한다. 여기서는 테스트에 필요한
    attribute access + keys()/items() 만 구현한다.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()


# ---------------------------------------------------------------------------
# Simulated HF-style models (non-BaseModel nn.Module)
# ---------------------------------------------------------------------------


class _HFClassificationModel(nn.Module):
    """HF AutoModelForSequenceClassification을 시뮬레이션한다.

    forward(input_ids, attention_mask) → ModelOutput(logits=...)
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> _FakeModelOutput:
        x = self.embedding(input_ids).mean(dim=1)  # (B, H)
        logits = self.classifier(x)  # (B, C)
        return _FakeModelOutput(logits=logits, hidden_states=None, attentions=None)


class _HFLossClassificationModel(_HFClassificationModel):
    """HF classifier output with both native loss and logits."""

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> _FakeModelOutput:
        logits = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return _FakeModelOutput(logits=logits, loss=loss, hidden_states=None, attentions=None)


class _HFLossOnlyModel(_HFClassificationModel):
    """HF-style output that exposes only a native loss."""

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> _FakeModelOutput:
        logits = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return _FakeModelOutput(logits=None, loss=loss, hidden_states=None, attentions=None)


class _HFFeatureExtractionModel(nn.Module):
    """HF AutoModel (feature extraction)을 시뮬레이션한다.

    logits 없이 last_hidden_state만 반환하는 경우.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> _FakeModelOutput:
        x = self.embedding(input_ids)  # (B, L, H)
        hidden = self.linear(x)  # (B, L, H)
        return _FakeModelOutput(logits=None, last_hidden_state=hidden, attentions=None)


class _HFDetectionModel(nn.Module):
    """HF AutoModelForObjectDetection을 시뮬레이션한다.

    logits + pred_boxes를 반환한다.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_classes: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> _FakeModelOutput:
        x = self.embedding(input_ids).mean(dim=1)  # (B, H)
        logits = self.cls_head(x)  # (B, C)
        boxes = self.box_head(x)  # (B, 4)
        return _FakeModelOutput(logits=logits, pred_boxes=boxes)


class _PlainDictModel(nn.Module):
    """dict를 반환하는 일반 nn.Module (BaseModel이 아닌).

    model(**batch)로 호출되어야 하며, 출력이 이미 dict이면 정규화 없이 통과한다.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        x = self.embedding(input_ids).mean(dim=1)
        logits = self.classifier(x)
        return {"logits": logits}


class _PlainBatchDictModel(nn.Module):
    """Plain nn.Module using the MDP-style forward(batch) convention."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = self.embedding(batch["input_ids"]).mean(dim=1)
        logits = self.classifier(x)
        return {"logits": logits}


class _PlainTensorModel(nn.Module):
    """단일 Tensor를 반환하는 nn.Module.

    HF forward 경로에서 Tensor 출력이 {"logits": tensor}로 감싸지는지 검증.
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> Tensor:
        x = self.embedding(input_ids).mean(dim=1)
        return self.classifier(x)


class _VisionTensorModel(nn.Module):
    """raw timm/torchvision-style model: forward(x) -> logits tensor."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.pool(x).flatten(1))


class _PeftLikeWrapper(nn.Module):
    """Minimal PEFT-like wrapper that keeps adapter call-through semantics."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self._base_model = base_model
        self.calls: list[str] = []

    def get_base_model(self) -> nn.Module:
        return self._base_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("args" if args else "kwargs")
        if args:
            return self._base_model(*args)
        return self._base_model(**kwargs)


# ---------------------------------------------------------------------------
# Unit tests: _make_forward_fn dispatch
# ---------------------------------------------------------------------------


def test_make_forward_fn_base_model() -> None:
    """BaseModel이면 model(batch) 호출 경로를 반환한다."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    forward_fn = _make_forward_fn(model)

    batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 2)


def test_make_forward_fn_peft_wrapped_base_model_uses_batch_call() -> None:
    """PEFT-like wrappers keep adapter call-through while using BaseModel dispatch."""
    wrapper = _PeftLikeWrapper(TinyVisionModel(num_classes=2, hidden_dim=16))
    forward_fn = _make_forward_fn(wrapper)

    batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
    outputs = forward_fn(batch)

    assert wrapper.calls == ["args"]
    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 2)


def test_make_forward_fn_hf_classification() -> None:
    """HF 분류 모델은 model(**batch) + ModelOutput(logits=...) → dict 정규화."""
    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)
    # hidden_states가 None이므로 포함되지 않아야 한다
    assert "hidden_states" not in outputs


def test_make_forward_fn_peft_wrapped_hf_model_keeps_kwarg_call() -> None:
    """PEFT-like wrappers around HF-style models still use model(**batch)."""
    wrapper = _PeftLikeWrapper(_HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3))
    forward_fn = _make_forward_fn(wrapper)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert wrapper.calls == ["kwargs"]
    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)


def test_make_forward_fn_hf_classification_preserves_loss() -> None:
    """HF output with logits and native loss keeps both fields."""
    model = _HFLossClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (2, 3)
    assert outputs["loss"].ndim == 0


def test_make_forward_fn_hf_loss_only_output() -> None:
    """HF output with native loss only normalizes to a loss dict."""
    model = _HFLossOnlyModel(vocab_size=64, hidden_dim=16, num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert set(outputs) == {"loss"}
    assert outputs["loss"].ndim == 0


def test_make_forward_fn_hf_feature_extraction() -> None:
    """logits가 없는 feature extraction 모델은 last_hidden_state로 정규화된다."""
    model = _HFFeatureExtractionModel(vocab_size=64, hidden_dim=16)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "last_hidden_state" in outputs
    assert "logits" not in outputs
    assert outputs["last_hidden_state"].shape == (2, 8, 16)


def test_make_forward_fn_hf_detection_boxes() -> None:
    """detection 모델은 logits + boxes가 함께 정규화된다."""
    model = _HFDetectionModel(vocab_size=64, hidden_dim=16, num_classes=5)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert "boxes" in outputs
    assert outputs["boxes"].shape == (2, 4)


def test_make_forward_fn_plain_dict_output() -> None:
    """non-BaseModel이 dict를 직접 반환하면 정규화 없이 그대로 통과한다."""
    model = _PlainDictModel(vocab_size=64, hidden_dim=16, num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)


def test_make_forward_fn_plain_batch_dict_output() -> None:
    """non-BaseModel using forward(batch) keeps the MDP batch convention."""
    model = _PlainBatchDictModel(vocab_size=64, hidden_dim=16, num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)


def test_make_forward_fn_plain_tensor_output() -> None:
    """non-BaseModel이 단일 Tensor를 반환하면 {"logits": tensor}로 감싸진다."""
    model = _PlainTensorModel(vocab_size=64, hidden_dim=16, num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "input_ids": torch.randint(0, 64, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)


def test_make_forward_fn_raw_vision_single_tensor_model() -> None:
    """raw timm/torchvision-style forward(x) consumes batch['pixel_values']."""
    model = _VisionTensorModel(num_classes=3)
    forward_fn = _make_forward_fn(model)

    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([0, 1]),
    }
    outputs = forward_fn(batch)

    assert isinstance(outputs, dict)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 3)


# ---------------------------------------------------------------------------
# Integration: run_batch_inference with HF-style model
# ---------------------------------------------------------------------------


def test_batch_inference_hf_classification_model(tmp_path: Path) -> None:
    """HF-style 분류 모델로 run_batch_inference가 end-to-end 동작한다."""
    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)

    batches = [
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    loader = ListDataLoader(batches)

    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="jsonl", task="classification",
    )
    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    lines = result_path.read_text().strip().splitlines()
    assert len(lines) == 12  # 3 batches * 4 samples
    assert eval_results == {}


def test_batch_inference_hf_feature_extraction_model(tmp_path: Path) -> None:
    """feature extraction 모델의 last_hidden_state 출력이 DefaultOutputCallback fallback으로 처리된다."""
    model = _HFFeatureExtractionModel(vocab_size=64, hidden_dim=16)

    batches = [
        {
            "input_ids": torch.randint(0, 64, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        for _ in range(2)
    ]
    loader = ListDataLoader(batches)

    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="jsonl", task="feature_extraction",
    )
    result_path, _ = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="feature_extraction",
        device="cpu",
        callbacks=[output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    lines = result_path.read_text().strip().splitlines()
    assert len(lines) == 4  # 2 batches * 2 samples


def test_batch_inference_hf_with_callbacks(tmp_path: Path) -> None:
    """HF-style 모델에서도 inference callback lifecycle이 정상 동작한다."""
    from mdp.callbacks.base import BaseInferenceCallback

    class _TrackingCallback(BaseInferenceCallback):
        def __init__(self) -> None:
            self.setup_called = False
            self.batch_count = 0
            self.teardown_called = False

        def setup(self, model: nn.Module, tokenizer: Any = None, **kwargs) -> None:
            self.setup_called = True

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            self.batch_count += 1
            # HF forward 어댑터가 정규화한 출력이 dict인지 확인
            assert isinstance(outputs, dict)
            assert "logits" in outputs

        def teardown(self, **kwargs) -> None:
            self.teardown_called = True

    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)

    batches = [
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        }
        for _ in range(3)
    ]
    loader = ListDataLoader(batches)
    cb = _TrackingCallback()

    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb],
    )

    assert cb.setup_called
    assert cb.batch_count == 3
    assert cb.teardown_called


def test_batch_inference_base_model_unchanged(tmp_path: Path) -> None:
    """기존 BaseModel 경로가 forward 어댑터 도입 후에도 동일하게 동작한다."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="jsonl", task="classification",
    )
    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    lines = result_path.read_text().strip().splitlines()
    assert len(lines) == 12
    assert eval_results == {}
