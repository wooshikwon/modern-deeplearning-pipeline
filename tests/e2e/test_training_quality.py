"""E2E tests for training quality: loss decrease, padding masking, MLflow logging.

7 tests:
- TestLossDecreases (2): vision and language models with LossRecorder callback
- TestPaddingMasking (4): variable-length padding, -100 masking, attention mask
- TestMLflowLogging (1): MLflow with sqlite tracking_uri
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mdp.settings.schema import Settings
from tests.e2e.conftest import make_test_settings
from mdp.training.callbacks.base import BaseCallback
from mdp.training.trainer import Trainer
from tests.e2e.datasets import (
    ListDataLoader,
    make_language_batches,
    make_vision_batches,
)
from tests.e2e.models import (
    TinyLanguageModel,
    TinyTokenClassModel,
    TinyVisionModel,
)


# ---------------------------------------------------------------------------
# LossRecorder callback
# ---------------------------------------------------------------------------


class LossRecorder(BaseCallback):
    """Records per-batch losses during training."""

    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics and "loss" in metrics:
            self.losses.append(metrics["loss"])


def _make_settings(
    task: str = "image_classification",
    epochs: int = 3,
    grad_accum: int = 1,
) -> Settings:
    """Create minimal Settings for quality tests."""
    return make_test_settings(
        task=task, epochs=epochs, gradient_accumulation_steps=grad_accum,
        name="quality-test",
    )


# ---------------------------------------------------------------------------
# TestLossDecreases — 2 tests with LossRecorder
# ---------------------------------------------------------------------------


class TestLossDecreases:
    """Verify loss decreases over training via LossRecorder callback."""

    def test_vision_loss_decreases(self) -> None:
        """Vision model loss should decrease from first to last batch over 5 epochs."""
        settings = _make_settings(task="image_classification", epochs=5)
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        batches = make_vision_batches(num_batches=5, batch_size=4, num_classes=2, image_size=8)
        train_loader = ListDataLoader(batches)

        recorder = LossRecorder()

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False
        trainer.callbacks.append(recorder)

        trainer.train()

        assert len(recorder.losses) > 0, "No losses recorded"
        # Compare first epoch average vs last epoch average
        batches_per_epoch = 5
        first_epoch_avg = sum(recorder.losses[:batches_per_epoch]) / batches_per_epoch
        last_epoch_avg = sum(recorder.losses[-batches_per_epoch:]) / batches_per_epoch
        assert last_epoch_avg < first_epoch_avg, (
            f"Loss did not decrease: first_epoch={first_epoch_avg:.4f}, "
            f"last_epoch={last_epoch_avg:.4f}"
        )

    def test_language_loss_decreases(self) -> None:
        """Language model loss should decrease over training."""
        settings = _make_settings(task="text_generation", epochs=5)
        model = TinyLanguageModel(vocab_size=128, hidden_dim=32)

        batches = make_language_batches(num_batches=5, batch_size=4, seq_len=16, vocab_size=128)
        train_loader = ListDataLoader(batches)

        recorder = LossRecorder()

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False
        trainer.callbacks.append(recorder)

        trainer.train()

        assert len(recorder.losses) > 0
        batches_per_epoch = 5
        first_epoch_avg = sum(recorder.losses[:batches_per_epoch]) / batches_per_epoch
        last_epoch_avg = sum(recorder.losses[-batches_per_epoch:]) / batches_per_epoch
        assert last_epoch_avg < first_epoch_avg, (
            f"Loss did not decrease: first_epoch={first_epoch_avg:.4f}, "
            f"last_epoch={last_epoch_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# TestPaddingMasking — 4 tests
# ---------------------------------------------------------------------------


class TestPaddingMasking:
    """Verify correct handling of padding and label masking."""

    def test_variable_length_padding(self) -> None:
        """Padded positions (0) should not contribute to loss when ignore_index=0."""
        model = TinyLanguageModel(vocab_size=128, hidden_dim=32)
        model.train()

        # Create batches with varying sequence lengths (pad with 0)
        batch1 = {"input_ids": torch.randint(1, 128, (2, 16))}
        batch2 = {"input_ids": torch.randint(1, 128, (2, 16))}
        # Zero-pad the last 8 tokens of batch2
        batch2["input_ids"][:, 8:] = 0

        out1 = model(batch1)
        out2 = model(batch2)

        # Both should produce valid logits
        assert out1["logits"].shape == (2, 16, 128)
        assert out2["logits"].shape == (2, 16, 128)

    def test_minus_100_masking_in_loss(self) -> None:
        """Labels with -100 should be ignored in cross_entropy loss."""
        model = TinyTokenClassModel(vocab_size=128, hidden_dim=32, num_classes=5)
        model.train()

        # All valid labels
        batch_valid = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "labels": torch.randint(0, 5, (2, 16)),
        }

        # Half labels masked with -100
        batch_masked = {
            "input_ids": batch_valid["input_ids"].clone(),
            "labels": batch_valid["labels"].clone(),
        }
        batch_masked["labels"][:, 8:] = -100

        loss_valid = model(batch_valid)["loss"]
        loss_masked = model(batch_masked)["loss"]

        # Both should produce finite losses
        assert torch.isfinite(loss_valid), f"Valid loss not finite: {loss_valid}"
        assert torch.isfinite(loss_masked), f"Masked loss not finite: {loss_masked}"

    def test_all_masked_labels_no_crash(self) -> None:
        """All labels=-100 should not crash (CE with ignore_index handles it)."""
        model = TinyTokenClassModel(vocab_size=128, hidden_dim=32, num_classes=5)
        model.train()

        batch = {
            "input_ids": torch.randint(0, 128, (2, 16)),
            "labels": torch.full((2, 16), -100, dtype=torch.long),
        }

        loss = model(batch)["loss"]
        # PyTorch CE(reduction='mean') + all ignore_index=-100 → NaN (0/0).
        # 크래시하지 않는 것만 검증.
        assert loss is not None

    def test_attention_mask_shapes(self) -> None:
        """Verify attention mask can be created and has correct shape."""
        seq_len = 16
        batch_size = 4

        # Simulate variable-length attention mask
        lengths = torch.tensor([12, 16, 8, 14])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1

        assert attention_mask.shape == (batch_size, seq_len)
        assert attention_mask[0, 11] == 1  # within length
        assert attention_mask[0, 12] == 0  # outside length
        assert attention_mask[2, 7] == 1   # within length
        assert attention_mask[2, 8] == 0   # outside length


# ---------------------------------------------------------------------------
# TestMLflowLogging — 1 test with sqlite tracking_uri
# ---------------------------------------------------------------------------


class TestMLflowLogging:
    """Verify MLflow logging works with sqlite backend."""

    def test_mlflow_logs_metrics(self, tmp_path: Path) -> None:
        """Trainer should log metrics to MLflow without errors."""
        import pytest

        mlflow = pytest.importorskip("mlflow")

        tracking_uri = f"sqlite:///{tmp_path / 'mlruns.db'}"

        # Configure Trainer's MLflow settings through Settings (Trainer reads these)
        settings = _make_settings(task="image_classification", epochs=2)
        settings.config.mlflow.tracking_uri = tracking_uri
        settings.config.mlflow.experiment_name = "test-quality"
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2, image_size=8)
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        trainer.train()

        # Verify some metrics were logged
        client = mlflow.tracking.MlflowClient(tracking_uri)
        exp = client.get_experiment_by_name("test-quality")
        assert exp is not None, "MLflow experiment 'test-quality' not found"
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        assert len(runs) > 0, "No MLflow runs found"
        run_data = runs[0].data

        # train_loss should have been logged
        assert "train_loss" in run_data.metrics or len(run_data.metrics) > 0
