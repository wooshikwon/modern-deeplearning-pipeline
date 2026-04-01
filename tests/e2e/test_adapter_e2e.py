"""E2E tests for LoRA adapter + Trainer integration.

Tests verify that LoRA-adapted models train correctly through the Trainer,
that only adapter parameters are trainable, and that base parameters
remain frozen after training.
"""

from __future__ import annotations

import pytest
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
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel

peft = pytest.importorskip("peft")


def _make_settings(
    epochs: int = 3,
    precision: str = "fp32",
) -> Settings:
    recipe = Recipe(
        name="lora-test",
        task="image_classification",
        model=ModelSpec(class_path="tests.e2e.models.TinyVisionModel"),
        data=DataSpec(source="/tmp/fake"),
        training=TrainingSpec(epochs=epochs, precision=precision),
        optimizer={"_component_": "torch.optim.AdamW", "lr": 1e-3},
        metadata=MetadataSpec(author="test", description="lora adapter e2e"),
    )
    config = Config()
    config.job.resume = "disabled"
    return Settings(recipe=recipe, config=config)


def _make_lora_model() -> torch.nn.Module:
    from mdp.models.adapters.lora import apply_lora

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    model = apply_lora(
        model,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["classifier", "head"],
    )
    return model


class TestLoRATrainerIntegration:
    """LoRA adapter + Trainer end-to-end tests."""

    def test_lora_training_completes(self) -> None:
        """Apply LoRA to TinyVisionModel, train 3 epochs via Trainer, verify completion."""
        settings = _make_settings(epochs=3)
        model = _make_lora_model()

        batches = make_vision_batches(num_batches=5, batch_size=4, num_classes=2, image_size=8)
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

        assert result["total_epochs"] == 3
        assert "val_loss" in result["metrics"] or "loss" in result["metrics"]

    def test_lora_only_adapter_params_trainable(self) -> None:
        """After apply_lora, trainable params should be strictly less than total."""
        model = _make_lora_model()

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        assert trainable < total, (
            f"Trainable ({trainable}) should be less than total ({total})"
        )
        assert trainable > 0, "No trainable parameters after LoRA"

    def test_lora_base_params_frozen_after_training(self) -> None:
        """Train 1 epoch with LoRA; base (frozen) params must not change."""
        model = _make_lora_model()

        # Snapshot frozen params before training
        frozen_before: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_before[name] = param.data.clone()

        assert len(frozen_before) > 0, "No frozen parameters found"

        settings = _make_settings(epochs=1)
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

        # Verify frozen params are unchanged
        for name, param in model.named_parameters():
            if name in frozen_before:
                torch.testing.assert_close(
                    param.data,
                    frozen_before[name],
                    msg=f"Frozen param '{name}' changed during training",
                )
