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

        # algorithm을 _component_ 패턴으로 resolve
        self.algorithm = self.resolver.resolve(recipe.algorithm)

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

        needs_gen = getattr(self.algorithm, "needs_generation", False)

        # generation kwargs (Recipe의 generation 섹션에서)
        gen_config = self.settings.recipe.generation
        self._generation_kwargs = {}
        if gen_config is not None:
            self._generation_kwargs = gen_config.model_dump(exclude_none=True) if hasattr(gen_config, "model_dump") else dict(gen_config)

        try:
            while self.global_step < max_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                batch = self._move_to_device(batch)

                if needs_gen:
                    step_loss = self._train_step_generation(batch, device_type)
                else:
                    step_loss = self._train_step_offline(batch, device_type)

                total_loss += step_loss
                num_steps += 1

                if (self.global_step) % self.grad_accum_steps == 0:
                    self._fire(
                        "on_batch_end", step=self.global_step,
                        global_step=self.global_step,
                        metrics={"loss": step_loss},
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
            "algorithm": type(self.algorithm).__name__,
        }

    # ── Step 실행 ──

    def _train_step_offline(self, batch: dict, device_type: str) -> float:
        """DPO / weighted-NTP — 데이터가 이미 완성된 경로."""
        with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            with torch.no_grad():
                frozen_out = {name: self._forward_model(m, batch) for name, m in self.frozen.items()}
            trainable_out = {name: self._forward_model(m, batch) for name, m in self.trainable.items()}
            losses = self.algorithm(trainable_out, frozen_out, batch)

        self._backward_and_step(losses)
        self.global_step += 1
        return losses.get("policy", list(losses.values())[0]).item()

    def _train_step_generation(self, batch: dict, device_type: str) -> float:
        """GRPO / PPO — policy가 텍스트를 생성하고, 그 결과로 학습."""
        from mdp.training.losses.rl import compute_log_probs

        prompt_ids = batch["input_ids"]
        prompt_mask = batch.get("attention_mask")

        # 1. Generation (no_grad)
        with torch.no_grad():
            generated_ids = self.policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                **self._generation_kwargs,
            )
            gen_mask = (generated_ids != self.policy.config.pad_token_id).long() if hasattr(self.policy, "config") and hasattr(self.policy.config, "pad_token_id") and self.policy.config.pad_token_id is not None else torch.ones_like(generated_ids)

        # 2. Old log_probs (update 전 policy 상태)
        with torch.no_grad():
            old_logits = self._extract_logits(self.policy(input_ids=generated_ids, attention_mask=gen_mask))
            old_log_probs = compute_log_probs(old_logits, generated_ids)

        # 3. Frozen forward + reward scoring
        gen_input = {"input_ids": generated_ids, "attention_mask": gen_mask}
        with torch.no_grad():
            frozen_out = {
                name: self._forward_model(m, gen_input)
                for name, m in self.frozen.items()
            }

        # reward 계산: frozen "reward" 모델이 있으면 그 출력을 reward로 사용
        rewards = self._compute_rewards(frozen_out, generated_ids, gen_mask)

        gen_batch = {
            "input_ids": generated_ids,
            "attention_mask": gen_mask,
            "labels": generated_ids,
            "prompt_length": prompt_ids.shape[1],
            "old_log_probs": old_log_probs,
            "rewards": rewards,
        }

        # 4. Mini-epoch update
        ppo_epochs = getattr(self.algorithm, "ppo_epochs", 1)
        last_loss = 0.0
        for _ in range(ppo_epochs):
            with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                trainable_out = {
                    name: self._forward_model(m, {"input_ids": generated_ids, "attention_mask": gen_mask})
                    for name, m in self.trainable.items()
                }
                losses = self.algorithm(trainable_out, frozen_out, gen_batch)

            self._backward_and_step(losses)
            last_loss = losses.get("policy", list(losses.values())[0]).item()

        self.global_step += 1
        return last_loss

    def _compute_rewards(
        self,
        frozen_out: dict[str, dict],
        generated_ids: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen reward model에서 reward를 추출한다.

        reward model이 있으면 마지막 토큰의 scalar reward를 반환.

        Returns:
            (batch,) per-sequence scalar reward.
            Loss 클래스가 알고리즘에 맞게 사용한다:
            - GRPO: 그대로 group normalization
            - PPO: per-token reward 배열로 변환 (마지막 토큰에 배치)
        """
        if "reward" in frozen_out:
            reward_logits = frozen_out["reward"].get("logits")
            if reward_logits is not None:
                if reward_logits.dim() == 3:
                    last_token_idx = gen_mask.sum(dim=-1).long() - 1
                    scalar_rewards = reward_logits[
                        torch.arange(reward_logits.shape[0], device=reward_logits.device),
                        last_token_idx, 0,
                    ]
                elif reward_logits.dim() == 2:
                    scalar_rewards = reward_logits[:, -1]
                else:
                    scalar_rewards = reward_logits.squeeze()
                return scalar_rewards

        # reward model 없으면: reference log_prob sum을 scalar reward로 (fallback)
        if "reference" in frozen_out and "logits" in frozen_out["reference"]:
            from mdp.training.losses.rl import compute_log_probs
            ref_lp = compute_log_probs(frozen_out["reference"]["logits"], generated_ids)
            mask = gen_mask[:, 1:gen_mask.shape[1]]
            if ref_lp.shape[1] < mask.shape[1]:
                mask = mask[:, :ref_lp.shape[1]]
            return (ref_lp * mask).sum(dim=-1)

        return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

    def _backward_and_step(self, losses: dict[str, torch.Tensor]) -> None:
        """모델별 독립 backward + optimizer step."""
        for name, loss in losses.items():
            scaled = loss / self.grad_accum_steps
            self.scaler.scale(scaled).backward()

        if (self.global_step + 1) % self.grad_accum_steps == 0:
            for name in losses:
                if name in self.optimizers:
                    self.scaler.unscale_(self.optimizers[name])
                    if self.grad_clip_norm is not None:
                        clip_grad_norm_(self.trainable[name].parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizers[name])
                    if name in self.schedulers:
                        self.schedulers[name].step()
            self.scaler.update()
            for opt in self.optimizers.values():
                opt.zero_grad()

    @staticmethod
    def _extract_logits(out):
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict):
            return out.get("logits", out.get("output"))
        return out

    def _forward_model(self, model: nn.Module, batch: dict) -> dict:
        """배치 형태에 따라 모델 forward를 수행한다.

        preference 배치 (DPO): chosen_input_ids, rejected_input_ids → chosen_logits, rejected_logits
        causal 배치 (weighted-NTP, PPO): input_ids → logits
        """
        result = {}

        def _extract_logits(out):
            if hasattr(out, "logits"):
                return out.logits
            if isinstance(out, dict):
                return out.get("logits", out.get("output"))
            return out

        # preference 형태
        if "chosen_input_ids" in batch:
            result["chosen_logits"] = _extract_logits(model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch.get("chosen_attention_mask"),
            ))
        if "rejected_input_ids" in batch:
            result["rejected_logits"] = _extract_logits(model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch.get("rejected_attention_mask"),
            ))

        # causal 형태
        if "input_ids" in batch:
            result["logits"] = _extract_logits(model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            ))

        return result
