"""E2E tests for external loss_fn path in Trainer.

Tests verify that when recipe.loss is configured, the Trainer uses
model.forward() + loss_fn.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mdp.settings.schema import Settings
from tests.e2e.conftest import make_test_settings
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


class _RawVisionModel(nn.Module):
    """raw timm/torchvision-style model with forward(x)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(x).flatten(1))


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

    def test_external_loss_trains_raw_vision_tensor_model(self) -> None:
        """raw timm/torchvision-style forward(x) works with recipe.loss."""
        settings = _make_settings(
            epochs=1,
            loss={"_component_": "torch.nn.CrossEntropyLoss"},
        )
        model = _RawVisionModel(num_classes=2)

        train_loader = ListDataLoader(
            make_vision_batches(num_batches=3, batch_size=4, num_classes=2, image_size=8)
        )

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        result = trainer.train()

        assert result["stopped_reason"] == "completed"
        assert result["total_steps"] == 3

    def test_external_loss_uses_forward(self) -> None:
        """With loss_fn set, Trainer should call forward()."""
        settings = _make_settings(
            epochs=1,
            loss={"_component_": "torch.nn.CrossEntropyLoss"},
        )
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        # Instrument forward with a counter
        forward_count = 0
        original_forward = model.forward

        def counting_forward(batch):
            nonlocal forward_count
            forward_count += 1
            return original_forward(batch)

        model.forward = counting_forward

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
