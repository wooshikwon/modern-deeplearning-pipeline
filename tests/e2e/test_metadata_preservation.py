"""메타데이터 보존 + 콜백 전달 테스트 (U3).

pretrained 분기에서 토큰화 전에 메타데이터를 추출하고,
run_batch_inference가 콜백에 metadata 슬라이스를 전달하는 과정을 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from mdp.callbacks.base import BaseInferenceCallback
from mdp.serving.inference import run_batch_inference
from tests.e2e.datasets import ListDataLoader


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _HFClassificationModel(nn.Module):
    """HF AutoModelForSequenceClassification을 시뮬레이션한다."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, **kwargs) -> dict[str, Tensor]:
        x = self.embedding(input_ids).mean(dim=1)
        logits = self.classifier(x)
        return {"logits": logits}


class MetadataCapturingCallback(BaseInferenceCallback):
    """on_batch에서 metadata kwarg를 캡처하여 검증에 사용한다."""

    def __init__(self) -> None:
        self.captured_metadata: list[list[dict] | None] = []
        self.batch_count = 0

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
        self.batch_count += 1
        self.captured_metadata.append(kwargs.get("metadata"))


# ---------------------------------------------------------------------------
# Tests: run_batch_inference metadata 전달
# ---------------------------------------------------------------------------


def test_metadata_passed_to_callback(tmp_path: Path) -> None:
    """metadata가 있으면 on_batch에 배치 슬라이스가 전달된다."""
    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)

    batch_size = 4
    num_batches = 3
    batches = [
        {
            "input_ids": torch.randint(0, 64, (batch_size, 8)),
            "attention_mask": torch.ones(batch_size, 8, dtype=torch.long),
        }
        for _ in range(num_batches)
    ]
    loader = ListDataLoader(batches)

    # 12개 샘플에 대한 메타데이터
    total_samples = batch_size * num_batches
    metadata = [{"label": f"class_{i % 3}", "topic": f"topic_{i % 5}"} for i in range(total_samples)]

    cb = MetadataCapturingCallback()
    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb],
        metadata=metadata,
    )

    assert cb.batch_count == num_batches
    # 각 배치에 4개씩 메타데이터 슬라이스가 전달되었는지 확인
    for i, meta_slice in enumerate(cb.captured_metadata):
        assert meta_slice is not None, f"batch {i}: metadata가 None"
        assert len(meta_slice) == batch_size, f"batch {i}: 슬라이스 크기 불일치"
        # 내용 검증: batch 0은 sample 0-3, batch 1은 sample 4-7, ...
        start = i * batch_size
        for j, rec in enumerate(meta_slice):
            expected_label = f"class_{(start + j) % 3}"
            assert rec["label"] == expected_label, f"batch {i}, sample {j}: label 불일치"


def test_metadata_none_when_not_provided(tmp_path: Path) -> None:
    """metadata를 전달하지 않으면 on_batch의 metadata는 None이다."""
    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)

    batches = [
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        }
        for _ in range(2)
    ]
    loader = ListDataLoader(batches)

    cb = MetadataCapturingCallback()
    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb],
    )

    assert cb.batch_count == 2
    for meta_slice in cb.captured_metadata:
        assert meta_slice is None


def test_metadata_with_uneven_last_batch(tmp_path: Path) -> None:
    """마지막 배치가 작을 때 metadata 슬라이싱이 올바르게 동작한다."""
    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)

    # 3개 배치: 4, 4, 2 = 10 samples
    batches = [
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        },
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        },
        {
            "input_ids": torch.randint(0, 64, (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        },
    ]
    loader = ListDataLoader(batches)

    total_samples = 10
    metadata = [{"idx": i} for i in range(total_samples)]

    cb = MetadataCapturingCallback()
    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb],
        metadata=metadata,
    )

    assert cb.batch_count == 3

    # batch 0: samples 0-3
    assert len(cb.captured_metadata[0]) == 4
    assert cb.captured_metadata[0][0]["idx"] == 0
    assert cb.captured_metadata[0][3]["idx"] == 3

    # batch 1: samples 4-7
    assert len(cb.captured_metadata[1]) == 4
    assert cb.captured_metadata[1][0]["idx"] == 4
    assert cb.captured_metadata[1][3]["idx"] == 7

    # batch 2: samples 8-9 (마지막 배치, 크기 2)
    assert len(cb.captured_metadata[2]) == 2
    assert cb.captured_metadata[2][0]["idx"] == 8
    assert cb.captured_metadata[2][1]["idx"] == 9


def test_metadata_backward_compat_existing_callback(tmp_path: Path) -> None:
    """metadata를 사용하지 않는 기존 콜백이 metadata 전달 시에도 정상 동작한다.

    on_batch의 **kwargs가 metadata를 흡수하므로 기존 콜백은 영향 없다.
    """

    class _LegacyCallback(BaseInferenceCallback):
        """metadata를 받지 않는 기존 스타일 콜백."""

        def __init__(self) -> None:
            self.batch_count = 0

        def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
            self.batch_count += 1
            # metadata를 의도적으로 무시 — **kwargs에 흡수됨

    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)
    batches = [
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        }
        for _ in range(2)
    ]
    loader = ListDataLoader(batches)
    metadata = [{"label": "a"} for _ in range(8)]

    cb = _LegacyCallback()
    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb],
        metadata=metadata,
    )

    assert cb.batch_count == 2


def test_metadata_with_multiple_callbacks(tmp_path: Path) -> None:
    """여러 콜백이 동일한 metadata 슬라이스를 받는다."""
    model = _HFClassificationModel(vocab_size=64, hidden_dim=16, num_classes=3)

    batches = [
        {
            "input_ids": torch.randint(0, 64, (4, 8)),
            "attention_mask": torch.ones(4, 8, dtype=torch.long),
        }
        for _ in range(2)
    ]
    loader = ListDataLoader(batches)
    metadata = [{"idx": i} for i in range(8)]

    cb1 = MetadataCapturingCallback()
    cb2 = MetadataCapturingCallback()

    run_batch_inference(
        model=model,
        dataloader=loader,
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
        device="cpu",
        callbacks=[cb1, cb2],
        metadata=metadata,
    )

    # 두 콜백이 동일한 슬라이스를 받아야 한다
    assert cb1.captured_metadata == cb2.captured_metadata
    assert len(cb1.captured_metadata) == 2
    assert cb1.captured_metadata[0][0]["idx"] == 0
    assert cb1.captured_metadata[1][0]["idx"] == 4
