"""E2E tests for EarlyStopping callback + Trainer integration.

Tests verify that EarlyStopping halts training before reaching the
configured number of epochs when the monitored metric stops improving.
"""

from __future__ import annotations

import torch

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    ModelSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from mdp.training.callbacks.early_stopping import EarlyStopping
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


def _make_settings(
    epochs: int = 100,
    precision: str = "fp32",
) -> Settings:
    recipe = Recipe(
        name="early-stopping-test",
        task="image_classification",
        model=ModelSpec(class_path="tests.e2e.models.TinyVisionModel"),
        data=DataSpec(source="/tmp/fake"),
        training=TrainingSpec(epochs=epochs, precision=precision),
        optimizer={"_component_": "torch.optim.AdamW", "lr": 1e-3},
        metadata=MetadataSpec(author="test", description="early stopping e2e"),
    )
    config = Config()
    config.job.resume = "disabled"
    return Settings(recipe=recipe, config=config)


class TestEarlyStoppingIntegration:
    """EarlyStopping + Trainer end-to-end tests."""

    def test_early_stopping_stops_training(self) -> None:
        """With patience=2, training should stop well before 100 epochs."""
        settings = _make_settings(epochs=100)
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        # Use deterministic data so val_loss plateaus quickly
        train_batches = make_vision_batches(
            num_batches=3, batch_size=4, num_classes=2, image_size=8, seed=42
        )
        val_batches = make_vision_batches(
            num_batches=2, batch_size=4, num_classes=2, image_size=8, seed=99
        )
        train_loader = ListDataLoader(train_batches)
        val_loader = ListDataLoader(val_batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        # Inject EarlyStopping callback with low patience
        early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min", min_delta=0.1)
        trainer.callbacks.append(early_stop)

        result = trainer.train()

        assert result["total_epochs"] < 100, (
            f"Expected training to stop early, but ran all {result['total_epochs']} epochs"
        )
        assert result["stopped_reason"] == "early_stopped", (
            f"Expected stopped_reason='early_stopped', got '{result['stopped_reason']}'"
        )
