"""Metric 평가 통합 테스트 — run_batch_inference에 metrics를 전달했을 때."""

from __future__ import annotations

from pathlib import Path

import torch

from mdp.callbacks.inference import DefaultOutputCallback
from mdp.serving.inference import run_batch_inference
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


class _SimpleAccuracy:
    """torchmetrics 없이 동작하는 간단한 accuracy metric (테스트용).

    update(outputs, batch) / compute() 프로토콜을 따른다.
    """

    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, outputs: dict, batch: dict) -> None:
        logits = outputs.get("logits")
        labels = batch.get("labels")
        if logits is None or labels is None:
            return
        preds = logits.argmax(dim=-1)
        self.correct += (preds == labels).sum().item()
        self.total += labels.numel()

    def compute(self) -> float:
        return self.correct / max(self.total, 1)


def test_inference_with_metrics(tmp_path: Path) -> None:
    """metrics 전달 시 evaluation_results가 반환된다."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    metrics = [_SimpleAccuracy()]
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
        metrics=metrics,
        callbacks=[output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    assert "_SimpleAccuracy" in eval_results
    acc = eval_results["_SimpleAccuracy"]
    assert 0.0 <= acc <= 1.0


def test_inference_without_metrics(tmp_path: Path) -> None:
    """metrics=None이면 eval_results가 빈 dict."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=2, batch_size=2, num_classes=2)
    loader = ListDataLoader(batches)

    output_cb = DefaultOutputCallback(
        output_path=tmp_path / "preds", output_format="parquet", task="classification",
    )
    result_path, eval_results = run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="parquet",
        task="classification",
        device="cpu",
        callbacks=[output_cb],
    )

    assert result_path is not None
    assert result_path.exists()
    assert eval_results == {}


def test_metric_missing_key_silent(tmp_path: Path) -> None:
    """metric이 필요한 키가 없으면 해당 metric은 0 결과를 반환 (에러 없음)."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    # labels 없는 배치
    batches = [{"pixel_values": torch.randn(2, 3, 8, 8)} for _ in range(2)]
    loader = ListDataLoader(batches)

    metrics = [_SimpleAccuracy()]
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
        metrics=metrics,
        callbacks=[output_cb],
    )

    # labels가 없으므로 accuracy는 0/0 = 0.0
    assert eval_results["_SimpleAccuracy"] == 0.0
