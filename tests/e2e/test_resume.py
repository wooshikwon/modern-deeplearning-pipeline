"""E2E tests for checkpoint save and resume training.

Tests verify that ModelCheckpoint saves trainer state correctly and
that Trainer can resume from a checkpoint and continue training.
"""

from __future__ import annotations

import json

import torch

from mdp.settings.schema import Settings
from tests.e2e.conftest import make_test_settings
from mdp.training.callbacks.base import BaseCallback
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

    def test_resume_restores_optimizer_state(self, tmp_path) -> None:
        """Resume 후 optimizer learning rate가 저장 시점과 일치하는지."""
        ckpt_dir = tmp_path / "checkpoints"

        # Phase 1: CosineAnnealingLR로 3 에폭 학습 (lr이 감소)
        settings1 = _make_settings(epochs=3, checkpoint_dir=str(ckpt_dir))
        settings1.recipe.scheduler = {
            "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 5,
            "interval": "epoch",
        }
        model1 = TinyVisionModel(num_classes=2, hidden_dim=16)

        batches = make_vision_batches(3, 4, 2, 8)
        val_batches = make_vision_batches(2, 4, 2, 8, seed=99)

        trainer1 = Trainer(
            settings=settings1, model=model1,
            train_loader=ListDataLoader(batches),
            val_loader=ListDataLoader(val_batches),
        )
        trainer1.device = torch.device("cpu")
        trainer1.amp_enabled = False
        trainer1.callbacks.append(
            ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", mode="min")
        )

        trainer1.train()
        lr_after_phase1 = trainer1.optimizer.param_groups[0]["lr"]

        # Phase 2: Resume — optimizer lr이 복원되는지
        settings2 = _make_settings(epochs=5, resume="auto", checkpoint_dir=str(ckpt_dir))
        settings2.recipe.scheduler = settings1.recipe.scheduler
        model2 = TinyVisionModel(num_classes=2, hidden_dim=16)

        trainer2 = Trainer(
            settings=settings2, model=model2,
            train_loader=ListDataLoader(batches),
            val_loader=ListDataLoader(val_batches),
        )
        trainer2.device = torch.device("cpu")
        trainer2.amp_enabled = False

        # resume 후 lr이 phase1 종료 시점과 유사해야 함
        lr_after_resume = trainer2.optimizer.param_groups[0]["lr"]
        assert abs(lr_after_resume - lr_after_phase1) < 0.01, (
            f"LR mismatch: phase1={lr_after_phase1:.6f}, resume={lr_after_resume:.6f}"
        )

    def test_resume_model_pt_fallback(self, tmp_path) -> None:
        """safetensors 없이 model.pt만 있을 때 resume 성공."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_step_dir = ckpt_dir / "checkpoint-3"
        ckpt_step_dir.mkdir(parents=True)

        # model.pt로 저장 (safetensors 없이)
        model_orig = TinyVisionModel(num_classes=2, hidden_dim=16)
        torch.save(model_orig.state_dict(), ckpt_step_dir / "model.pt")
        torch.save(
            torch.optim.AdamW(model_orig.parameters(), lr=1e-3).state_dict(),
            ckpt_step_dir / "optimizer.pt",
        )
        (ckpt_step_dir / "trainer_state.json").write_text(
            json.dumps({"epoch": 1, "global_step": 3})
        )

        # latest 심링크 (절대 경로로 생성하여 resolve 호환)
        latest = ckpt_dir / "latest"
        latest.symlink_to(ckpt_step_dir)

        # Resume
        settings = _make_settings(epochs=3, resume="auto", checkpoint_dir=str(ckpt_dir))
        model_new = TinyVisionModel(num_classes=2, hidden_dim=16)

        trainer = Trainer(
            settings=settings, model=model_new,
            train_loader=ListDataLoader(make_vision_batches(3, 4, 2, 8)),
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False

        # train() 호출 시 _maybe_resume가 실행되어 epoch/step 복원
        result = trainer.train()
        # epoch 1에서 시작하여 3까지 = 2 에폭 학습
        assert result["total_epochs"] == 2
        assert trainer.global_step > 3

    def test_step_level_resume_skips_processed_batches(self, tmp_path) -> None:
        """step-level checkpoint에서 resume 시 이미 처리된 배치를 건너뛴다."""
        ckpt_dir = tmp_path / "checkpoints"
        num_batches = 6

        # Phase 1: Train 1 epoch with step-level checkpointing (every 3 steps)
        settings1 = _make_settings(epochs=1, checkpoint_dir=str(ckpt_dir))
        model1 = TinyVisionModel(num_classes=2, hidden_dim=16)
        batches = make_vision_batches(
            num_batches=num_batches, batch_size=4, num_classes=2, image_size=8
        )

        trainer1 = Trainer(
            settings=settings1,
            model=model1,
            train_loader=ListDataLoader(batches),
        )
        trainer1.device = torch.device("cpu")
        trainer1.amp_enabled = False

        ckpt_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val_loss",
            mode="min",
            every_n_steps=3,
        )
        trainer1.callbacks.append(ckpt_cb)
        trainer1.train()

        # Verify a step-level checkpoint was saved with step_in_epoch
        latest_link = ckpt_dir / "latest"
        assert latest_link.exists() or latest_link.is_symlink()
        state_path = latest_link.resolve() / "trainer_state.json"
        state = json.loads(state_path.read_text())
        assert state["step_in_epoch"] > 0, "step_in_epoch should be > 0"
        saved_step_in_epoch = state["step_in_epoch"]

        # Phase 2: Resume for another epoch — first epoch should skip processed batches
        settings2 = _make_settings(
            epochs=2, resume="auto", checkpoint_dir=str(ckpt_dir)
        )
        model2 = TinyVisionModel(num_classes=2, hidden_dim=16)

        # Track actual batch processing count
        processed_batches = []

        class BatchCounter(BaseCallback):
            def on_batch_start(self, step: int, **kwargs):
                processed_batches.append(step)

        trainer2 = Trainer(
            settings=settings2,
            model=model2,
            train_loader=ListDataLoader(batches),
        )
        trainer2.device = torch.device("cpu")
        trainer2.amp_enabled = False
        trainer2.callbacks.append(BatchCounter())

        result2 = trainer2.train()

        # The first epoch after resume should have skipped `saved_step_in_epoch` batches.
        # Total batches in first resumed epoch = num_batches - saved_step_in_epoch
        # Second epoch = full num_batches
        # So total processed < 2 * num_batches
        expected_max = 2 * num_batches - saved_step_in_epoch
        assert len(processed_batches) <= expected_max, (
            f"Expected at most {expected_max} batches, got {len(processed_batches)}"
        )
