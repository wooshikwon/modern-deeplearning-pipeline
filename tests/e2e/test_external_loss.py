"""E2E tests for external loss_fn path in Trainer.

Tests verify that when recipe.loss is configured, the Trainer uses
model.forward() + loss_fn instead of model.training_step().
"""

from __future__ import annotations

import torch

from mdp.settings.schema import Settings
from tests.e2e.conftest import make_test_settings
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


def _make_settings(
    epochs: int = 3,
    precision: str = "fp32",
    loss: dict | None = None,
) -> Settings:
    settings = make_test_settings(epochs=epochs, precision=precision, name="external-loss-test")
    if loss is not None:
        settings.recipe.loss = loss
    return settings


class TestExternalLoss:
    """Trainer with external loss_fn (recipe.loss configured)."""

    def test_external_loss_training_completes(self) -> None:
        """TinyVisionModel + CrossEntropyLoss external loss trains 3 epochs successfully."""
        settings = _make_settings(
            epochs=3,
            loss={"_component_": "torch.nn.CrossEntropyLoss"},
        )
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        batches = make_vision_batches(num_batches=5, batch_size=4, num_classes=2, image_size=8)
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        result = trainer.train()

        assert result["total_epochs"] == 3
        assert result["total_steps"] == 5 * 3  # 5 batches * 3 epochs
        assert result["stopped_reason"] == "completed"

    def test_external_loss_uses_forward_not_training_step(self) -> None:
        """With loss_fn set, Trainer should call forward() not training_step()."""
        settings = _make_settings(
            epochs=1,
            loss={"_component_": "torch.nn.CrossEntropyLoss"},
        )
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        # Instrument forward and training_step with counters
        forward_count = 0
        training_step_count = 0
        original_forward = model.forward
        original_training_step = model.training_step

        def counting_forward(batch):
            nonlocal forward_count
            forward_count += 1
            return original_forward(batch)

        def counting_training_step(batch):
            nonlocal training_step_count
            training_step_count += 1
            return original_training_step(batch)

        model.forward = counting_forward
        model.training_step = counting_training_step

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

        assert forward_count > 0, "forward() should have been called"
        assert training_step_count == 0, (
            f"training_step() should not be called when loss_fn is set, "
            f"but was called {training_step_count} times"
        )
