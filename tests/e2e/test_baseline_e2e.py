"""E2E tests for Trainer monitoring baseline integration.

Tests verify that baseline computation is triggered when monitoring
is enabled and skipped when disabled.
"""

from __future__ import annotations

import torch

from mdp.settings.schema import Settings
from tests.e2e.conftest import make_test_settings
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


def _make_settings(
    epochs: int = 2,
    precision: str = "fp32",
    monitoring_enabled: bool = True,
    checkpoint_dir: str = "./checkpoints",
) -> Settings:
    return make_test_settings(
        epochs=epochs, precision=precision, monitoring_enabled=monitoring_enabled,
        checkpoint_dir=checkpoint_dir, name="baseline-test",
    )


class TestBaselineIntegration:
    """Trainer monitoring baseline E2E tests."""

    def test_baseline_saved_when_enabled(self, tmp_path) -> None:
        """With monitoring enabled, train should return monitoring info with baseline_saved."""
        settings = _make_settings(
            epochs=2,
            monitoring_enabled=True,
            checkpoint_dir=str(tmp_path),
        )
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

        result = trainer.train()

        assert "monitoring" in result, (
            f"Expected 'monitoring' key in result, got keys: {list(result.keys())}"
        )
        assert result["monitoring"]["baseline_saved"] is True

        # Verify baseline file exists on disk
        from pathlib import Path

        baseline_path = Path(result["monitoring"]["baseline_path"])
        assert baseline_path.exists(), (
            f"Baseline file not found at {baseline_path}"
        )

    def test_baseline_skipped_when_disabled(self) -> None:
        """With monitoring disabled, train result should not contain 'monitoring' key."""
        settings = _make_settings(
            epochs=2,
            monitoring_enabled=False,
        )
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

        result = trainer.train()

        assert "monitoring" not in result, (
            f"Expected no 'monitoring' key when disabled, got: {result.get('monitoring')}"
        )
