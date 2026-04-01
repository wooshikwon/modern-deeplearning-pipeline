"""E2E tests for Trainer validation fallback with models lacking validation_step().

Tests verify that the Trainer's _validate_fallback path works correctly
for models that either use an external loss function or return loss
directly in forward() output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    ModelSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches


# ── Models without validation_step ──


class ModelWithoutValidationStep(nn.Module):
    """Model with forward returning logits but no validation_step."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(8, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["pixel_values"]
        features = self.backbone(x)
        logits = self.head(features)
        return {"logits": logits}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        labels = batch["labels"]
        return F.cross_entropy(outputs["logits"], labels)


class ModelReturningLossInForward(nn.Module):
    """Model with forward returning both loss and logits."""

    def __init__(self, num_classes: int = 2, hidden_dim: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(8, num_classes)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        x = batch["pixel_values"]
        features = self.backbone(x)
        logits = self.head(features)
        labels = batch.get("labels")
        result: dict[str, Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:
        outputs = self.forward(batch)
        return outputs["loss"]


# ── Helpers ──


def _make_settings(
    epochs: int = 2,
    precision: str = "fp32",
    loss: dict | None = None,
) -> Settings:
    recipe = Recipe(
        name="fallback-test",
        task="image_classification",
        model=ModelSpec(class_path="tests.e2e.test_validation_fallback.ModelWithoutValidationStep"),
        data=DataSpec(source="/tmp/fake"),
        training=TrainingSpec(epochs=epochs, precision=precision),
        optimizer={"_component_": "torch.optim.AdamW", "lr": 1e-3},
        loss=loss,
        metadata=MetadataSpec(author="test", description="validation fallback e2e"),
    )
    config = Config()
    config.job.resume = "disabled"
    return Settings(recipe=recipe, config=config)


class TestValidationFallback:
    """Trainer validation fallback for models without validation_step."""

    def test_fallback_with_external_loss_fn(self) -> None:
        """Model without validation_step + external loss_fn should produce val_loss."""
        settings = _make_settings(
            epochs=2,
            loss={"_component_": "torch.nn.CrossEntropyLoss"},
        )
        model = ModelWithoutValidationStep(num_classes=2)

        batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2, image_size=8)
        train_loader = ListDataLoader(batches)
        val_loader = ListDataLoader(
            make_vision_batches(num_batches=2, batch_size=4, num_classes=2, image_size=8, seed=99)
        )

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        result = trainer.train()

        # Fallback should have computed val metrics using loss_fn(logits, labels)
        assert "loss" in result["metrics"], (
            f"Expected 'loss' in metrics, got: {result['metrics']}"
        )

    def test_fallback_with_loss_in_output(self) -> None:
        """Model returning loss in forward dict should produce val_loss via outputs['loss']."""
        settings = _make_settings(epochs=2)
        model = ModelReturningLossInForward(num_classes=2)

        batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2, image_size=8)
        train_loader = ListDataLoader(batches)
        val_loader = ListDataLoader(
            make_vision_batches(num_batches=2, batch_size=4, num_classes=2, image_size=8, seed=99)
        )

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        result = trainer.train()

        # Fallback should have used outputs["loss"] directly
        assert "loss" in result["metrics"], (
            f"Expected 'loss' in metrics, got: {result['metrics']}"
        )
