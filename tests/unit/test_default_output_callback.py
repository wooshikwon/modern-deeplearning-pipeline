"""Unit tests for DefaultOutputCallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from mdp.callbacks.inference import DefaultOutputCallback, _postprocess, _to_numpy


# ---------------------------------------------------------------------------
# on_batch: logits를 올바르게 후처리하는지
# ---------------------------------------------------------------------------


def test_default_output_callback_on_batch_classification(tmp_path: Path) -> None:
    """on_batch가 classification logits를 prediction + probabilities로 변환하여 누적한다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
    )

    batch_size = 4
    num_classes = 3
    outputs = {"logits": torch.randn(batch_size, num_classes)}
    batch = {"pixel_values": torch.randn(batch_size, 3, 8, 8)}

    cb.on_batch(batch_idx=0, batch=batch, outputs=outputs)

    assert len(cb._records) == batch_size
    for record in cb._records:
        assert "prediction" in record
        assert "probabilities" in record
        assert isinstance(record["prediction"], int)
        assert isinstance(record["probabilities"], list)
        assert len(record["probabilities"]) == num_classes


def test_default_output_callback_on_batch_multiple_batches(tmp_path: Path) -> None:
    """여러 배치 호출 시 records가 올바르게 누적된다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="parquet",
        task="classification",
    )

    num_batches = 3
    batch_size = 4
    for i in range(num_batches):
        outputs = {"logits": torch.randn(batch_size, 2)}
        batch = {"pixel_values": torch.randn(batch_size, 3, 8, 8)}
        cb.on_batch(batch_idx=i, batch=batch, outputs=outputs)

    assert len(cb._records) == num_batches * batch_size


def test_default_output_callback_on_batch_detection(tmp_path: Path) -> None:
    """detection 출력(boxes 키)이 올바르게 후처리된다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="detection",
    )

    batch_size = 2
    outputs = {"boxes": torch.randn(batch_size, 4)}
    batch = {"pixel_values": torch.randn(batch_size, 3, 8, 8)}

    cb.on_batch(batch_idx=0, batch=batch, outputs=outputs)

    assert len(cb._records) == batch_size
    for record in cb._records:
        assert "boxes" in record


# ---------------------------------------------------------------------------
# bf16 텐서가 numpy로 정상 변환되는지
# ---------------------------------------------------------------------------


def test_default_output_callback_bf16_tensor(tmp_path: Path) -> None:
    """bf16 logits 텐서가 TypeError 없이 on_batch에서 올바르게 처리된다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
    )

    batch_size = 4
    num_classes = 3
    # bf16 텐서 생성
    logits = torch.randn(batch_size, num_classes).to(torch.bfloat16)
    outputs = {"logits": logits}
    batch = {"pixel_values": torch.randn(batch_size, 3, 8, 8)}

    # TypeError 없이 정상 처리되어야 한다
    cb.on_batch(batch_idx=0, batch=batch, outputs=outputs)

    assert len(cb._records) == batch_size
    for record in cb._records:
        assert "prediction" in record
        assert "probabilities" in record


def test_to_numpy_bf16_fallback() -> None:
    """_to_numpy가 bf16 텐서를 float32로 캐스팅하여 변환한다."""
    t = torch.randn(4, 3).to(torch.bfloat16)
    result = _to_numpy(t)

    assert result.dtype.name == "float32"
    assert result.shape == (4, 3)


def test_to_numpy_float32_direct() -> None:
    """_to_numpy가 float32 텐서를 직접 변환한다."""
    t = torch.randn(4, 3)
    result = _to_numpy(t)

    assert result.dtype.name == "float32"
    assert result.shape == (4, 3)


# ---------------------------------------------------------------------------
# teardown 후 파일이 올바른 포맷으로 저장되는지
# ---------------------------------------------------------------------------


def test_default_output_callback_teardown_jsonl(tmp_path: Path) -> None:
    """teardown 후 JSONL 파일이 올바르게 저장된다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
    )

    # 2 배치 x 4 샘플
    for i in range(2):
        outputs = {"logits": torch.randn(4, 3)}
        cb.on_batch(batch_idx=i, batch={}, outputs=outputs)

    cb.teardown()

    assert cb.result_path is not None
    assert cb.result_path.exists()
    assert cb.result_path.suffix == ".jsonl"

    lines = cb.result_path.read_text().strip().splitlines()
    assert len(lines) == 8

    # 각 줄이 유효한 JSON인지
    for line in lines:
        record = json.loads(line)
        assert "prediction" in record
        assert "probabilities" in record


def test_default_output_callback_teardown_parquet(tmp_path: Path) -> None:
    """teardown 후 parquet 파일이 올바르게 저장된다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="parquet",
        task="classification",
    )

    for i in range(2):
        outputs = {"logits": torch.randn(4, 2)}
        cb.on_batch(batch_idx=i, batch={}, outputs=outputs)

    cb.teardown()

    assert cb.result_path is not None
    assert cb.result_path.exists()
    assert cb.result_path.suffix == ".parquet"
    assert cb.result_path.stat().st_size > 0


def test_default_output_callback_teardown_csv(tmp_path: Path) -> None:
    """teardown 후 CSV 파일이 올바르게 저장된다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="csv",
        task="classification",
    )

    for i in range(2):
        outputs = {"logits": torch.randn(4, 2)}
        cb.on_batch(batch_idx=i, batch={}, outputs=outputs)

    cb.teardown()

    assert cb.result_path is not None
    assert cb.result_path.exists()
    assert cb.result_path.suffix == ".csv"

    lines = cb.result_path.read_text().strip().splitlines()
    # header + 8 data rows
    assert len(lines) == 9


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_default_output_callback_result_path_none_before_teardown(tmp_path: Path) -> None:
    """teardown 전에는 result_path가 None이다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
    )

    assert cb.result_path is None


def test_default_output_callback_teardown_no_records(tmp_path: Path) -> None:
    """records가 비어 있으면 teardown이 파일을 생성하지 않는다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
    )

    cb.teardown()

    assert cb.result_path is None


def test_default_output_callback_invalid_format() -> None:
    """지원하지 않는 output_format이면 ValueError를 발생시킨다."""
    import pytest

    with pytest.raises(ValueError, match="Unsupported format"):
        DefaultOutputCallback(
            output_path="/tmp/preds",
            output_format="xlsx",
            task="classification",
        )


def test_default_output_callback_is_base_inference_callback() -> None:
    """DefaultOutputCallback이 BaseInferenceCallback의 서브클래스이다."""
    from mdp.callbacks.base import BaseInferenceCallback

    assert issubclass(DefaultOutputCallback, BaseInferenceCallback)


def test_default_output_callback_fallback_key(tmp_path: Path) -> None:
    """logits/boxes/generated_ids가 없으면 첫 번째 키를 fallback으로 사용한다."""
    cb = DefaultOutputCallback(
        output_path=tmp_path / "preds",
        output_format="jsonl",
        task="classification",
    )

    outputs = {"last_hidden_state": torch.randn(4, 16, 32)}
    cb.on_batch(batch_idx=0, batch={}, outputs=outputs)

    assert len(cb._records) == 4
    for record in cb._records:
        assert "last_hidden_state" in record
