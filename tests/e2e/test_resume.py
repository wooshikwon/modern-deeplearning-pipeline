"""E2E tests for checkpoint save and resume training.

Tests verify that ModelCheckpoint saves trainer state correctly and
that Trainer can resume from a checkpoint and continue training.
"""

from __future__ import annotations

import json

import torch

from mdp.settings.schema import Settings
from tests.e2e.conftest import make_test_settings
from mdp.training.callbacks.checkpoint import ModelCheckpoint
from mdp.training.trainer import Trainer
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


def _make_settings(
    epochs: int = 3,
    precision: str = "fp32",
    resume: str = "disabled",
    checkpoint_dir: str = "./checkpoints",
) -> Settings:
    settings = make_test_settings(
        epochs=epochs, precision=precision, checkpoint_dir=checkpoint_dir,
        name="resume-test",
    )
    settings.config.job.resume = resume
    return settings


class TestResume:
    """Checkpoint save and resume training E2E tests."""

    def test_checkpoint_saves_epoch_plus_one(self, tmp_path) -> None:
        """ModelCheckpoint.on_validation_end(epoch=2) saves epoch=3 in trainer_state.json."""
        model = TinyVisionModel(num_classes=2, hidden_dim=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        ckpt_cb = ModelCheckpoint(
            dirpath=tmp_path / "checkpoints",
            monitor="val_loss",
            mode="min",
        )

        # Simulate on_validation_end at epoch=2 (0-indexed)
        # ModelCheckpoint saves epoch+1 internally
        ckpt_cb.on_validation_end(
            epoch=2,
            metrics={"val_loss": 0.5},
            model=model,
            optimizer=optimizer,
            scheduler=None,
            global_step=10,
            strategy=None,
        )

        # Find the saved checkpoint
        ckpt_dirs = list((tmp_path / "checkpoints").glob("checkpoint-*"))
        assert len(ckpt_dirs) == 1

        state_path = ckpt_dirs[0] / "trainer_state.json"
        assert state_path.exists()

        state = json.loads(state_path.read_text())
        assert state["epoch"] == 3, (
            f"Expected epoch=3 (epoch+1), got {state['epoch']}"
        )

    def test_resume_continues_from_correct_epoch(self, tmp_path) -> None:
        """Train 3 epochs with checkpoint, resume and train to epoch 5."""
        ckpt_dir = tmp_path / "checkpoints"

        # Phase 1: Train 3 epochs with checkpointing
        settings1 = _make_settings(epochs=3, checkpoint_dir=str(ckpt_dir))
        model1 = TinyVisionModel(num_classes=2, hidden_dim=16)

        batches = make_vision_batches(num_batches=3, batch_size=4, num_classes=2, image_size=8)
        train_loader1 = ListDataLoader(batches)
        val_loader1 = ListDataLoader(
            make_vision_batches(num_batches=2, batch_size=4, num_classes=2, image_size=8, seed=99)
        )

        trainer1 = Trainer(
            settings=settings1,
            model=model1,
            train_loader=train_loader1,
            val_loader=val_loader1,
        )
        trainer1.device = torch.device("cpu")
        trainer1.amp_enabled = False

        # Add checkpoint callback
        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            mode="min",
        )
        trainer1.callbacks.append(ckpt_cb)

        result1 = trainer1.train()
        assert result1["total_epochs"] == 3

        # Verify checkpoint was saved and "latest" symlink exists
        latest_link = ckpt_dir / "latest"
        assert latest_link.exists() or latest_link.is_symlink(), (
            "Expected 'latest' symlink in checkpoint directory"
        )

        # Phase 2: Resume and train for 2 more epochs (total target: 5)
        settings2 = _make_settings(
            epochs=5,
            resume="auto",
            checkpoint_dir=str(ckpt_dir),
        )
        model2 = TinyVisionModel(num_classes=2, hidden_dim=16)

        train_loader2 = ListDataLoader(batches)
        val_loader2 = ListDataLoader(
            make_vision_batches(num_batches=2, batch_size=4, num_classes=2, image_size=8, seed=99)
        )

        trainer2 = Trainer(
            settings=settings2,
            model=model2,
            train_loader=train_loader2,
            val_loader=val_loader2,
        )
        trainer2.device = torch.device("cpu")
        trainer2.amp_enabled = False

        result2 = trainer2.train()

        # Should have trained only 2 more epochs (from epoch 3 to epoch 4, total_epochs=2)
        assert result2["total_epochs"] == 2, (
            f"Expected 2 additional epochs, got {result2['total_epochs']}"
        )
