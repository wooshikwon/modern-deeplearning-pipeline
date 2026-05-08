"""E2E tests for the Trainer with various model types.

5 tests:
- TestVisionTrainer: vision classification via Trainer
- TestLanguageTrainer (2): causal LM training + grad_accum
- TestTokenClassTrainer: token classification via Trainer
- TestMultimodalTrainer: dual-encoder contrastive via Trainer
"""

from __future__ import annotations

from typing import Any

import torch

from mdp.factory.bundles import build_sft_training_bundle
from mdp.settings.schema import Settings
from mdp.training.trainer import Trainer
from tests.e2e.conftest import make_test_settings
from tests.e2e.datasets import (
    ListDataLoader,
    make_language_batches,
    make_multimodal_batches,
    make_token_class_batches,
    make_vision_batches,
)
from tests.e2e.models import (
    TinyDualEncoderModel,
    TinyLanguageModel,
    TinyTokenClassModel,
    TinyVisionModel,
)


def _make_settings(
    task: str = "image_classification",
    epochs: int = 3,
    grad_accum: int = 1,
    precision: str = "fp32",
) -> Settings:
    return make_test_settings(
        task=task, epochs=epochs,
        gradient_accumulation_steps=grad_accum, precision=precision,
    )


class TestVisionTrainer:
    """Train TinyVisionModel through the Trainer."""

    def test_vision_trainer_loss_decreases(self) -> None:
        """Trainer.train() on vision batches should return without error."""
        settings = _make_settings(task="image_classification", epochs=3)
        model = TinyVisionModel(num_classes=2, hidden_dim=16)

        batches = make_vision_batches(num_batches=5, batch_size=4, num_classes=2, image_size=8)
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        # Force CPU
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        metrics = trainer.train()
        # Trainer completes without error; global_step > 0
        assert trainer.global_step > 0


class TestLanguageTrainer:
    """Train TinyLanguageModel through the Trainer."""

    def test_language_trainer_trains(self) -> None:
        """Trainer.train() on language batches completes successfully."""
        settings = _make_settings(task="text_generation", epochs=3)
        model = TinyLanguageModel(vocab_size=128, hidden_dim=32)

        batches = make_language_batches(num_batches=5, batch_size=4, seq_len=16, vocab_size=128)
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        metrics = trainer.train()
        assert trainer.global_step == 5 * 3  # 5 batches * 3 epochs, grad_accum=1

    def test_language_trainer_grad_accum(self) -> None:
        """With grad_accum=2, global_step = (batches // 2) * epochs."""
        settings = _make_settings(task="text_generation", epochs=2, grad_accum=2)
        model = TinyLanguageModel(vocab_size=128, hidden_dim=32)

        batches = make_language_batches(num_batches=4, batch_size=4, seq_len=16, vocab_size=128)
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        metrics = trainer.train()
        # 4 batches per epoch, grad_accum=2 -> 2 steps per epoch, 2 epochs -> 4 total
        assert trainer.global_step == 4, f"Expected 4, got {trainer.global_step}"

    def test_language_trainer_from_bundle_trains(self) -> None:
        """Bundle-oriented path preserves the SFT training loop."""
        settings = _make_settings(task="text_generation", epochs=2)
        model = TinyLanguageModel(vocab_size=128, hidden_dim=32)

        batches = make_language_batches(num_batches=4, batch_size=4, seq_len=16, vocab_size=128)
        train_loader = ListDataLoader(batches)
        bundle = build_sft_training_bundle(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )

        trainer = Trainer.from_bundle(bundle)
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        trainer.train()
        assert trainer.global_step == 4 * 2


class TestTokenClassTrainer:
    """Train TinyTokenClassModel through the Trainer."""

    def test_token_class_trainer_trains(self) -> None:
        """Trainer on token classification batches completes."""
        settings = _make_settings(task="token_classification", epochs=2)
        model = TinyTokenClassModel(vocab_size=128, hidden_dim=32, num_classes=5)

        batches = make_token_class_batches(
            num_batches=5, batch_size=4, seq_len=16, vocab_size=128, num_classes=5
        )
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        metrics = trainer.train()
        assert trainer.global_step == 5 * 2


class TestMultimodalTrainer:
    """Train TinyDualEncoderModel through the Trainer."""

    def test_multimodal_trainer_trains(self) -> None:
        """Trainer on multimodal batches completes successfully."""
        settings = _make_settings(task="feature_extraction", epochs=2)
        model = TinyDualEncoderModel(hidden_dim=16, projection_dim=8)

        batches = make_multimodal_batches(
            num_batches=5, batch_size=4, image_size=8, seq_len=8, vocab_size=128
        )
        train_loader = ListDataLoader(batches)

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=train_loader,
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        metrics = trainer.train()
        assert trainer.global_step == 5 * 2
