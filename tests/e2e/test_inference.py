"""E2E tests for batch inference."""

from __future__ import annotations

from pathlib import Path

from mdp.serving.inference import run_batch_inference
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
