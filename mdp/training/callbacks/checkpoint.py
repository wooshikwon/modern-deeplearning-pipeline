"""Model checkpoint callback for saving and managing training snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch

from mdp.training._checkpoint import (
    CheckpointContext,
    CheckpointManager,
    ModelSlot,
)
from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

_MODEL_SLOT_ROLES = {"policy", "reference", "reward", "critic", "value", "model"}


def _role_for_slot(name: str) -> Literal["policy", "reference", "reward", "critic", "value", "model"]:
    return name if name in _MODEL_SLOT_ROLES else "model"  # type: ignore[return-value]


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
    strict:
        When ``True``, the callback raises ``ValueError`` on the first
        ``on_validation_end`` call where ``monitor`` is not present in
        the returned metrics.  When ``False`` (default), a warning is
        emitted and the save is skipped — preserving legacy behaviour.
        Note: this is independent from ``self.critical`` (which controls
        whether the trainer re-raises exceptions thrown by the callback).
    """

    def __init__(
        self,
        dirpath: str | Path | None = None,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        every_n_steps: int | None = None,
        strict: bool = False,
    ) -> None:
        if mode not in ("min", "max"):
            msg = f"mode must be 'min' or 'max', got '{mode}'"
            raise ValueError(msg)

        # dirpath=None means "derive from trainer's storage.checkpoint_dir".
        # Trainer injects the resolved path via set_dirpath() before training starts.
        # dirpath=str/Path means explicitly set — trainer will not override.
        self._dirpath_explicit = dirpath is not None
        self.dirpath = Path(dirpath) if dirpath is not None else Path("checkpoints")
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.every_n_steps = every_n_steps
        self.critical: bool = True
        # strict: monitor 미매칭 시 즉시 실패할지 여부. self.critical과 이름은 비슷하지만
        # 의미가 다르다 — critical은 콜백 내 예외를 trainer가 재전파할지 결정하는 스위치고,
        # strict는 ModelCheckpoint가 monitor 미매칭을 silent skip 대신 ValueError로
        # 전환할지 결정한다. 이름 충돌 방지를 위해 의도적으로 다른 필드로 분리.
        self.strict: bool = strict

        # (metric_value, checkpoint_path) — worst first for easy eviction
        self.best_models: list[tuple[float, str]] = []
        # 저장에 성공한 체크포인트 디렉토리 목록. save_checkpoint / _save_checkpoint_state가
        # 예외 없이 반환한 직후에만 append한다 (타이밍 규칙). Trainer/RLTrainer의
        # _log_mlflow_summary가 duck typing으로 집계하여 MLflow tag·WARNING에 활용.
        self.saved_checkpoints: list[Path] = []
        self._last_save_wrote = False

    def set_dirpath(self, dirpath: str | Path) -> None:
        """Trainer calls this to inject storage.checkpoint_dir when dirpath was not explicit."""
        if not self._dirpath_explicit:
            self.dirpath = Path(dirpath)

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
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        epoch: int,
        global_step: int,
        metrics: dict[str, float] | None = None,
        strategy: Any | None = None,
        recipe_dict: dict[str, Any] | None = None,
        scaler: Any | None = None,
        step_in_epoch: int = 0,
        trainer: Any | None = None,
        saved_at: Literal["step", "validation", "train_end", "manual"] = "manual",
        kind: Literal["sft", "rl"] = "sft",
    ) -> Path:
        """Persist a checkpoint to disk and return its path."""
        ckpt_dir = self.dirpath / f"checkpoint-{global_step}"
        self._last_save_wrote = False

        is_main = True
        requires_all_ranks_for_save = False
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                is_main = dist.get_rank() == 0
            if strategy is not None:
                capability = getattr(strategy, "checkpoint_capability", None)
                requires_all_ranks_for_save = bool(
                    capability is not None
                    and capability.requires_all_ranks_for_save
                )
        except Exception:
            pass

        if not is_main and not requires_all_ranks_for_save:
            return ckpt_dir

        slots = self._model_slots_for_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer=trainer,
        )
        scaler_to_save = (
            scaler
            if scaler is not None
            and (not hasattr(scaler, "is_enabled") or scaler.is_enabled())
            else None
        )
        CheckpointManager().save(
            CheckpointContext(
                kind=kind,
                ckpt_dir=ckpt_dir,
                global_step=global_step,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                saved_at=saved_at,
                metrics=metrics or {},
                recipe_dict=recipe_dict,
                is_main_process=is_main,
            ),
            slots,
            strategy=strategy,
            scaler=scaler_to_save,
        )

        if is_main:
            self._update_symlink("latest", ckpt_dir)
            logger.info("Saved checkpoint: %s", ckpt_dir)
            self._last_save_wrote = True
        return ckpt_dir

    def _model_slots_for_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        trainer: Any | None,
    ) -> list[ModelSlot]:
        if trainer is not None and hasattr(trainer, "_checkpoint_model_slots"):
            return trainer._checkpoint_model_slots()

        if trainer is not None and hasattr(trainer, "trainable"):
            slots: list[ModelSlot] = []
            optimizers = getattr(trainer, "optimizers", {})
            schedulers = getattr(trainer, "schedulers", {})
            for name, slot_model in getattr(trainer, "trainable", {}).items():
                slots.append(
                    ModelSlot(
                        name=name,
                        role=_role_for_slot(name),
                        model=slot_model,
                        trainable=True,
                        optimizer=optimizers.get(name),
                        scheduler=schedulers.get(name),
                    )
                )
            for name, slot_model in getattr(trainer, "frozen", {}).items():
                slots.append(
                    ModelSlot(
                        name=name,
                        role=_role_for_slot(name),
                        model=slot_model,
                        trainable=False,
                    )
                )
            return slots

        return [
            ModelSlot(
                name="",
                role="policy",
                model=model,
                trainable=True,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        ]

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
            if trainer is not None and hasattr(trainer, "trainable"):
                policy = getattr(trainer, "policy", None)
                if policy is None:
                    policy = getattr(trainer, "trainable", {}).get("policy")
                optimizers = getattr(trainer, "optimizers", {})
                schedulers = getattr(trainer, "schedulers", {})
                # save_checkpoint가 예외 없이 완료되어야만 saved_checkpoints에 기록.
                # 저장 실패 경로에서 리스트에 추가하면 zero-checkpoint 경고의 신뢰성이 깨진다.
                ckpt_dir = self.save_checkpoint(
                    policy,
                    optimizers.get("policy"),
                    schedulers.get("policy"),
                    kwargs.get("epoch", 0),
                    global_step,
                    metrics,
                    strategy=kwargs.get("strategy"),
                    recipe_dict=kwargs.get("recipe_dict"),
                    scaler=kwargs.get("scaler"),
                    step_in_epoch=kwargs.get("step_in_epoch", step),
                    trainer=trainer,
                    saved_at="step",
                    kind="rl",
                )
                if self._last_save_wrote:
                    self.saved_checkpoints.append(ckpt_dir)
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
                    step_in_epoch = kwargs.get("step_in_epoch", 0)
                    ckpt_path = self.save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step, metrics,
                        strategy=strategy, recipe_dict=recipe_dict, scaler=scaler,
                        step_in_epoch=step_in_epoch,
                        trainer=kwargs.get("trainer"),
                        saved_at="step",
                    )
                    if self._last_save_wrote:
                        self.saved_checkpoints.append(ckpt_path)

    def on_validation_end(
        self,
        epoch: int,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        if metrics is None or self.monitor not in metrics:
            available = sorted(metrics.keys()) if metrics else []
            if self.strict:
                # strict=True: 사용자가 명시적으로 "monitor 미매칭은 즉시 실패"를 요청.
                # 학습을 계속하면 silent failure로 산출물이 0개가 되는 상황을 방지.
                raise ValueError(
                    f"ModelCheckpoint: monitor metric '{self.monitor}' not found in "
                    f"validation results. Available: {available}. "
                    f"Set strict=False to allow silent skip."
                )
            logger.warning(
                "ModelCheckpoint: metric '%s' not found, skipping. Available: %s",
                self.monitor,
                available,
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

        # save_checkpoint가 예외 없이 반환해야만 saved_checkpoints에 append.
        # 실패 경로(저장 중 예외)에서는 아래 append 라인까지 도달하지 않으므로
        # zero-checkpoint 경고의 신뢰성이 유지된다.
        ckpt_path = self.save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, metrics,
            strategy=strategy, recipe_dict=recipe_dict, scaler=scaler,
            trainer=kwargs.get("trainer"),
            saved_at="validation",
        )
        if self._last_save_wrote:
            self.saved_checkpoints.append(ckpt_path)

            metric_value = metrics[self.monitor]
            self._manage_top_k(metric_value, ckpt_path)
