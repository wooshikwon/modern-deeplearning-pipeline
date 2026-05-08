"""E2E tests for baseline computation and drift detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from mdp.monitoring.baseline import compare_baselines, compute_baseline
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


# ---------------------------------------------------------------------------
# compute_baseline
# ---------------------------------------------------------------------------


def test_compute_baseline_structure() -> None:
    """compute_baseline returns a dict with meta, input_stats, output_stats."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2)
    loader = ListDataLoader(batches)

    baseline = compute_baseline(train_dataloader=loader, model=model, max_batches=3)

    assert "meta" in baseline
    assert "input_stats" in baseline
    assert "output_stats" in baseline

    # Meta section
    assert "timestamp" in baseline["meta"]
    assert "num_samples" in baseline["meta"]
    assert baseline["meta"]["num_samples"] > 0

    # Vision input stats should be present since we use pixel_values
    assert "vision" in baseline["input_stats"]
    assert "channel_mean" in baseline["input_stats"]["vision"]
    assert "channel_std" in baseline["input_stats"]["vision"]

    # Label distribution
    assert "label_distribution" in baseline["input_stats"]

    # Output stats
    assert "entropy_mean" in baseline["output_stats"]
    assert "confidence_mean" in baseline["output_stats"]


def test_compute_baseline_hf_style_model_outputs_stats() -> None:
    """compute_baseline uses the shared forward normalizer for HF-style models."""

    class _HFStyleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.proj = nn.Linear(4, 3)

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            hidden = self.embed(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1)
            return {"logits": self.proj(hidden[:, -1])}

    model = _HFStyleModel()
    loader = ListDataLoader([
        {
            "input_ids": torch.tensor([[1, 2, 3], [3, 2, 1]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
        }
    ])

    baseline = compute_baseline(train_dataloader=loader, model=model, max_batches=1)

    assert baseline["output_stats"]["entropy_mean"] > 0
    assert baseline["output_stats"]["confidence_mean"] > 0


# ---------------------------------------------------------------------------
# compare_baselines
# ---------------------------------------------------------------------------


def test_compare_baselines_no_drift() -> None:
    """Two baselines from the same model/data should show no drift."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2, seed=42)
    loader = ListDataLoader(batches)

    baseline = compute_baseline(train_dataloader=loader, model=model, max_batches=3)

    # Recompute with the same data
    loader2 = ListDataLoader(batches)
    current = compute_baseline(train_dataloader=loader2, model=model, max_batches=3)

    report = compare_baselines(baseline=baseline, current=current)

    assert "drift_detected" in report
    assert "drift_score" in report
    assert "alerts" in report
    assert report["drift_detected"] is False


def test_compare_baselines_detects_drift() -> None:
    """Baselines with very different output stats should trigger drift detection."""
    # Construct two artificial baselines with large entropy difference
    baseline = {
        "meta": {"timestamp": "2024-01-01T00:00:00", "num_samples": 100},
        "input_stats": {
            "label_distribution": {"0": 0.5, "1": 0.5},
        },
        "output_stats": {
            "entropy_mean": 0.5,
            "entropy_std": 0.1,
            "confidence_mean": 0.9,
        },
    }

    # Current with drastically different entropy and label distribution
    current = {
        "meta": {"timestamp": "2024-06-01T00:00:00", "num_samples": 100},
        "input_stats": {
            "label_distribution": {"0": 0.95, "1": 0.05},
        },
        "output_stats": {
            "entropy_mean": 2.0,
            "entropy_std": 0.5,
            "confidence_mean": 0.3,
        },
    }

    report = compare_baselines(baseline=baseline, current=current)

    assert report["drift_detected"] is True
    assert report["drift_score"] > 0
    assert len(report["alerts"]) > 0
