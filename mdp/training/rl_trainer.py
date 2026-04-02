"""RLTrainer — RL alignment 학습 루프.

SFT Trainer와 독립된 학습 루프. DPO, weighted-NTP, GRPO, PPO를 지원한다.
복수 모델(policy + frozen reference/critic)을 관리하며,
optimizer는 trainable 모델별로 독립 운용한다.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings
from mdp.training.losses.rl import compute_rl_loss

logger = logging.getLogger(__name__)


class RLTrainer:
    """RL alignment 학습 루프."""

    def __init__(
        self,
        settings: Settings,
        models: dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.resolver = ComponentResolver()

        recipe = settings.recipe
        training = recipe.training
        self.algorithm = recipe.algorithm
        self.algo_config = getattr(recipe, self.algorithm, None)

        # Device
        self.device = self._detect_device()

        # Training config
        self.max_steps = training.max_steps
        self.epochs = training.epochs
        self.grad_accum_steps = training.gradient_accumulation_steps
        self.grad_clip_norm = training.gradient_clip_max_norm

        # AMP
        precision = training.precision
        scaler_device = self.device.type
        if precision == "fp16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
            self.scaler = GradScaler(scaler_device, enabled=True)
        elif precision == "bf16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            self.scaler = GradScaler(scaler_device, enabled=False)
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32
            self.scaler = GradScaler(scaler_device, enabled=False)

        # 모델 분리: trainable vs frozen
        self.trainable: dict[str, nn.Module] = {}
        self.frozen: dict[str, nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}

        for name, spec in recipe.models.items():
            model = models[name]
            if spec.optimizer is not None:
                self.trainable[name] = model
                klass, kwargs = self.resolver.resolve_partial(spec.optimizer)
                self.optimizers[name] = klass(model.parameters(), **kwargs)
                if spec.scheduler is not None:
                    sched_config = dict(spec.scheduler)
                    warmup_ratio = sched_config.pop("warmup_ratio", 0.0)
                    s_klass, s_kwargs = self.resolver.resolve_partial(sched_config)
                    scheduler = s_klass(self.optimizers[name], **s_kwargs)
                    if warmup_ratio > 0:
                        total_steps = self._estimate_total_steps()
                        warmup_steps = int(total_steps * warmup_ratio)
                        warmup = torch.optim.lr_scheduler.LinearLR(
                            self.optimizers[name], start_factor=1e-8, end_factor=1.0,
                            total_iters=warmup_steps,
                        )
                        scheduler = torch.optim.lr_scheduler.SequentialLR(
                            self.optimizers[name],
                            schedulers=[warmup, scheduler],
                            milestones=[warmup_steps],
                        )
                    self.schedulers[name] = scheduler
            else:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.frozen[name] = model

        self.policy = self.trainable["policy"]

        # Strategy
        self.strategy = self._create_strategy(settings)
        self._is_main_process = int(os.environ.get("RANK", "0")) == 0

        # Callbacks
        self.callbacks = self._create_callbacks(recipe.callbacks)

        # State
        self.global_step = 0

    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _estimate_total_steps(self) -> int:
        if self.max_steps:
            return self.max_steps
        steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
        return steps_per_epoch * (self.epochs or 1)

    def _create_strategy(self, settings: Settings) -> Any:
        from mdp.training.trainer import STRATEGY_MAP

        dist_config = settings.config.compute.distributed
        if dist_config is None:
            return None
        strategy_name = dist_config.get("strategy", "auto") if isinstance(dist_config, dict) else "auto"
        if strategy_name in ("none", "auto"):
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
                return None
            strategy_name = "ddp"
        class_path = STRATEGY_MAP.get(strategy_name)
        if class_path is None:
            raise ValueError(f"알 수 없는 분산 전략: {strategy_name}")
        return self.resolver.resolve({"_component_": class_path})

    def _create_callbacks(self, configs: list[dict[str, Any]]) -> list:
        callbacks = []
        for cfg in configs:
            try:
                callbacks.append(self.resolver.resolve(cfg))
            except Exception as e:
                logger.warning(f"콜백 생성 실패: {e}")
        return callbacks

    def _fire(self, hook_name: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # ── 학습 루프 ──

    def train(self) -> dict[str, Any]:
        """RL 학습을 실행하고 결과를 반환한다."""
        # Strategy setup
        if self.strategy is not None:
            trainable_names = set(self.trainable.keys())
            all_models = {**self.trainable, **self.frozen}
            if hasattr(self.strategy, "setup_models"):
                wrapped = self.strategy.setup_models(all_models, self.device, trainable_names)
                for name in self.trainable:
                    self.trainable[name] = wrapped[name]
                for name in self.frozen:
                    self.frozen[name] = wrapped[name]
                self.policy = self.trainable["policy"]
            else:
                for name, model in self.trainable.items():
                    self.trainable[name] = self.strategy.setup(model, self.device)
                self.policy = self.trainable["policy"]
                for name, model in self.frozen.items():
                    self.frozen[name] = model.to(self.device)
        else:
            for name in self.trainable:
                self.trainable[name] = self.trainable[name].to(self.device)
            for name in self.frozen:
                self.frozen[name] = self.frozen[name].to(self.device)
            self.policy = self.trainable["policy"]

        total_steps = self._estimate_total_steps()
        self._fire("on_train_start", total_steps=total_steps)
        start_time = time.time()

        device_type = self.device.type if self.device.type != "mps" else "cpu"
        train_iter = iter(self.train_loader)
        total_loss = 0.0
        num_steps = 0
        max_steps = self.max_steps or (len(self.train_loader) * (self.epochs or 1))

        try:
            while self.global_step < max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                batch = self._move_to_device(batch)

                with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    # frozen forward
                    with torch.no_grad():
                        frozen_out = {}
                        for name, model in self.frozen.items():
                            frozen_out[name] = self._forward_preference(model, batch)

                    # trainable forward
                    trainable_out = {}
                    for name, model in self.trainable.items():
                        trainable_out[name] = self._forward_preference(model, batch)

                    # loss
                    losses = compute_rl_loss(
                        self.algorithm, trainable_out, frozen_out, batch, self.algo_config,
                    )

                # backward — 모델별 독립
                for name, loss in losses.items():
                    scaled = loss / self.grad_accum_steps
                    self.scaler.scale(scaled).backward()

                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    for name in losses:
                        self.scaler.unscale_(self.optimizers[name])
                        if self.grad_clip_norm is not None:
                            clip_grad_norm_(self.trainable[name].parameters(), self.grad_clip_norm)
                        self.scaler.step(self.optimizers[name])
                        if name in self.schedulers:
                            self.schedulers[name].step()
                    self.scaler.update()
                    for opt in self.optimizers.values():
                        opt.zero_grad()

                self.global_step += 1
                policy_loss = losses.get("policy", list(losses.values())[0])
                total_loss += policy_loss.item()
                num_steps += 1

                if (self.global_step) % self.grad_accum_steps == 0:
                    self._fire(
                        "on_batch_end", step=self.global_step,
                        global_step=self.global_step,
                        metrics={"loss": policy_loss.item()},
                        model=self.policy,
                        optimizer=self.optimizers["policy"],
                        scheduler=self.schedulers.get("policy"),
                    )

        finally:
            if self.strategy is not None:
                try:
                    self.strategy.cleanup()
                except Exception as e:
                    logger.warning(f"Strategy cleanup 실패: {e}")

        training_duration = time.time() - start_time
        avg_loss = total_loss / max(num_steps, 1)

        self._fire("on_train_end", metrics={"loss": avg_loss})

        return {
            "metrics": {"loss": avg_loss},
            "training_duration_seconds": training_duration,
            "total_steps": self.global_step,
            "algorithm": self.algorithm,
        }

    def _forward_preference(self, model: nn.Module, batch: dict) -> dict:
        """preference 배치에서 chosen/rejected를 각각 forward한다."""
        result = {}
        if "chosen_input_ids" in batch:
            chosen_out = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            result["chosen_logits"] = chosen_out.logits if hasattr(chosen_out, "logits") else chosen_out.get("logits", chosen_out)
        if "rejected_input_ids" in batch:
            rejected_out = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )
            result["rejected_logits"] = rejected_out.logits if hasattr(rejected_out, "logits") else rejected_out.get("logits", rejected_out)
        return result
