"""Model checkpoint callback for saving and managing training snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class ModelCheckpoint(BaseCallback):
    """Save model checkpoints during training.

    Checkpoints are stored as directories containing:
    - ``model.safetensors`` — model ``state_dict`` (or ``adapter_model.safetensors`` for LoRA)
    - ``optimizer.pt`` — optimizer ``state_dict``
    - ``scheduler.pt`` — scheduler ``state_dict`` (if provided)
    - ``trainer_state.json`` — epoch, global_step, metrics

    HuggingFace/PEFT models use ``save_pretrained()`` instead.
    Falls back to ``model.pt`` if safetensors is not installed.

    A ``latest`` symlink always points to the most recently saved
    checkpoint.  A ``best`` symlink points to the checkpoint with the
    best monitored metric value.

    Parameters
    ----------
    dirpath:
        Root directory for checkpoints.
    monitor:
        Metric name to decide which checkpoint is *best*.
    mode:
        ``"min"`` or ``"max"``.
    save_top_k:
        Maximum number of checkpoints to keep on disk.
    every_n_steps:
        If set, save a checkpoint every *n* training steps regardless
        of validation results (step-level checkpointing).
    """

    def __init__(
        self,
        dirpath: str | Path = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        every_n_steps: int | None = None,
    ) -> None:
        if mode not in ("min", "max"):
            msg = f"mode must be 'min' or 'max', got '{mode}'"
            raise ValueError(msg)

        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.every_n_steps = every_n_steps
        self.critical: bool = True

        # (metric_value, checkpoint_path) — worst first for easy eviction
        self.best_models: list[tuple[float, str]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_better(self, current: float, reference: float) -> bool:
        if self.mode == "min":
            return current < reference
        return current > reference

    def _sort_key(self, item: tuple[float, str]) -> float:
        """Return sort key so that the *worst* checkpoint comes first."""
        value = item[0]
        # For "min" mode, the worst (highest) value should be first
        # For "max" mode, the worst (lowest) value should be first
        return -value if self.mode == "min" else value

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        epoch: int,
        global_step: int,
        metrics: dict[str, float] | None = None,
        strategy: Any | None = None,
        recipe_dict: dict[str, Any] | None = None,
        scaler: Any | None = None,
        step_in_epoch: int = 0,
    ) -> Path:
        """Persist a checkpoint to disk and return its path."""
        ckpt_dir = self.dirpath / f"checkpoint-{global_step}"

        # 분산 학습 시 rank-0에서만 저장 (동시 쓰기로 인한 파일 손상 방지)
        is_main = True
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main = dist.get_rank() == 0
        except Exception:
            pass

        if not is_main:
            return ckpt_dir

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Recipe를 체크포인트에 내장 (최초 1회)
        recipe_path = ckpt_dir / "recipe.yaml"
        if recipe_dict is not None and not recipe_path.exists():
            import yaml

            recipe_path.write_text(
                yaml.dump(recipe_dict, allow_unicode=True, default_flow_style=False),
            )

        # Model weights: prefer strategy, then save_pretrained (HF/PEFT), then safetensors, fallback to .pt
        if strategy is not None:
            strategy.save_checkpoint(model, str(ckpt_dir / "model.safetensors"))
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(ckpt_dir)
        else:
            try:
                from safetensors.torch import save_file

                target = getattr(model, "module", model)
                save_file(target.state_dict(), ckpt_dir / "model.safetensors")
            except ImportError:
                logger.warning("safetensors not installed, falling back to torch.save")
                torch.save(model.state_dict(), ckpt_dir / "model.pt")

        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # GradScaler 상태 (fp16 resume 안정성)
        if scaler is not None and hasattr(scaler, "is_enabled") and scaler.is_enabled():
            torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")

        trainer_state = {
            "epoch": epoch,
            "global_step": global_step,
            "step_in_epoch": step_in_epoch,
            "metrics": metrics or {},
        }
        (ckpt_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2),
        )

        self._update_symlink("latest", ckpt_dir)

        logger.info("Saved checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def _update_symlink(self, name: str, target: Path) -> None:
        link = self.dirpath / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.name)

    def _manage_top_k(self, metric_value: float, ckpt_path: Path) -> None:
        """Keep only the top-k checkpoints (by monitored metric)."""
        self.best_models.append((metric_value, str(ckpt_path)))
        # Sort so worst checkpoint is first
        self.best_models.sort(key=self._sort_key)

        while len(self.best_models) > self.save_top_k:
            _, path_str = self.best_models.pop(0)
            removed = Path(path_str)
            if removed.exists():
                import shutil

                shutil.rmtree(removed)
                logger.info("Removed checkpoint: %s", removed)

        # Update "best" symlink to the checkpoint with the best metric
        if self.best_models:
            # Best is the last after sorting (worst-first order)
            best_path = Path(self.best_models[-1][1])
            self._update_symlink("best", best_path)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_batch_end(
        self,
        step: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        global_step = kwargs.get("global_step", step)
        if self.every_n_steps is not None and global_step > 0 and global_step % self.every_n_steps == 0:
            # RLTrainer: multi-model checkpoint 위임
            trainer = kwargs.get("trainer")
            if trainer is not None and hasattr(trainer, "_save_checkpoint"):
                ckpt_dir = self.dirpath / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                trainer._save_checkpoint(ckpt_dir)
                self._update_symlink("latest", ckpt_dir)
                logger.info("Saved RL checkpoint: %s", ckpt_dir)
            else:
                model = kwargs.get("model")
                optimizer = kwargs.get("optimizer")
                if model is not None and optimizer is not None:
                    scheduler = kwargs.get("scheduler")
                    strategy = kwargs.get("strategy")
                    epoch = kwargs.get("epoch", 0)
                    recipe_dict = kwargs.get("recipe_dict")
                    scaler = kwargs.get("scaler")
                    self.save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step, metrics,
                        strategy=strategy, recipe_dict=recipe_dict, scaler=scaler,
                        step_in_epoch=step + 1,
                    )

    def on_validation_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if metrics is None or self.monitor not in metrics:
            logger.warning(
                "ModelCheckpoint: metric '%s' not found, skipping.",
                self.monitor,
            )
            return

        model = kwargs.get("model")
        optimizer = kwargs.get("optimizer")
        if model is None or optimizer is None:
            logger.warning(
                "ModelCheckpoint: model/optimizer not provided in kwargs, skipping.",
            )
            return

        scheduler = kwargs.get("scheduler")
        strategy = kwargs.get("strategy")
        global_step = kwargs.get("global_step", 0)
        recipe_dict = kwargs.get("recipe_dict")
        scaler = kwargs.get("scaler")

        ckpt_path = self.save_checkpoint(
            model, optimizer, scheduler, epoch + 1, global_step, metrics,
            strategy=strategy, recipe_dict=recipe_dict, scaler=scaler,
        )

        metric_value = metrics[self.monitor]
        self._manage_top_k(metric_value, ckpt_path)
