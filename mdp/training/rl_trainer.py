"""RLTrainer — RL alignment 학습 루프.

SFT Trainer와 독립된 학습 루프. DPO, weighted-NTP, GRPO, PPO를 지원한다.
복수 모델(policy + frozen reference/critic)을 관리하며,
optimizer는 trainable 모델별로 독립 운용한다.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings
from mdp.training._common import create_expert_parallel, create_strategy, detect_device

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
        self.device = detect_device()

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
        self.strategy = create_strategy(settings, self.resolver)
        self.expert_parallel = create_expert_parallel(settings)
        self._is_main_process = int(os.environ.get("RANK", "0")) == 0

        # Callbacks
        self.callbacks = self._create_callbacks(recipe.callbacks)

        # Validation
        self.val_check_interval = getattr(training, "val_check_interval", 0)
        if isinstance(self.val_check_interval, float) and self.val_check_interval < 1.0:
            self.val_check_interval = 0  # RL은 step 기반만 지원

        # State
        self.global_step = 0
        self.last_metrics: dict[str, float] = {}
        self._generation_kwargs: dict[str, Any] = {}

    def _estimate_total_steps(self) -> int:
        if self.max_steps:
            return self.max_steps
        steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
        return steps_per_epoch * (self.epochs or 1)

    def _create_callbacks(self, configs: list[dict[str, Any]]) -> list:
        callbacks = []
        for cfg in configs:
            try:
                callbacks.append(self.resolver.resolve(cfg))
            except Exception as e:
                logger.warning(f"콜백 생성 실패: {e}")
        return callbacks

    def _fire(self, hook_name: str, **kwargs: Any) -> None:
        needs_ep_gather = (
            self.expert_parallel is not None
            and hook_name in ("on_validation_end", "on_batch_end")
        )
        if needs_ep_gather:
            self.expert_parallel.gather_experts(self.policy)

        kwargs["trainer"] = self
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    if getattr(cb, "critical", False):
                        raise
                    logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")

        if needs_ep_gather:
            self.expert_parallel.scatter_experts(self.policy, self.device)

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _should_stop(self) -> bool:
        return any(getattr(cb, "should_stop", False) for cb in self.callbacks)

    def _run_rl_validation(self) -> dict[str, float]:
        """validation: generation+reward (GRPO/PPO) 또는 preference accuracy (DPO)."""
        if self.val_loader is None:
            return {}

        needs_gen = getattr(self.algorithm, "needs_generation", False)
        if not needs_gen:
            return self._run_dpo_validation()

        if "reward" not in self.frozen:
            return {}

        self.policy.eval()
        total_reward = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)
                prompt_ids = batch["input_ids"]
                prompt_mask = batch.get("attention_mask")

                gen_model = getattr(self.policy, "module", self.policy)
                generated_ids = gen_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    **self._generation_kwargs,
                )

                reward_out = self._forward_model(
                    self.frozen["reward"],
                    {"input_ids": generated_ids},
                    role="reward",
                )
                rewards = self._compute_rewards(
                    {"reward": reward_out}, generated_ids,
                    (generated_ids != getattr(getattr(gen_model, "config", None), "pad_token_id", -1)).long(),
                )
                total_reward += rewards.sum().item()
                count += rewards.shape[0]

        self.policy.train()
        mean_reward = total_reward / max(count, 1)
        return {"val_mean_reward": mean_reward}

    def _run_dpo_validation(self) -> dict[str, float]:
        """DPO validation: preference accuracy (chosen log-prob > rejected log-prob)."""
        from mdp.training.losses.rl import compute_log_probs

        self.policy.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)

                # chosen/rejected는 collator가 batch에 넣어준다
                chosen_ids = batch.get("chosen_input_ids")
                rejected_ids = batch.get("rejected_input_ids")
                if chosen_ids is None or rejected_ids is None:
                    continue

                chosen_labels = batch.get("chosen_labels", chosen_ids)
                rejected_labels = batch.get("rejected_labels", rejected_ids)

                # 각각 forward
                chosen_out = self._forward_model(
                    self.policy,
                    {"input_ids": chosen_ids, "attention_mask": batch.get("chosen_attention_mask")},
                    role="policy",
                )
                rejected_out = self._forward_model(
                    self.policy,
                    {"input_ids": rejected_ids, "attention_mask": batch.get("rejected_attention_mask")},
                    role="policy",
                )

                chosen_logits = chosen_out.get("logits")
                rejected_logits = rejected_out.get("logits")
                if chosen_logits is None or rejected_logits is None:
                    continue

                # per-token log probs → sequence-level sum
                chosen_lp = compute_log_probs(chosen_logits, chosen_labels)
                rejected_lp = compute_log_probs(rejected_logits, rejected_labels)

                # mask out padding (-100)
                chosen_mask = chosen_labels[:, 1:] != -100
                rejected_mask = rejected_labels[:, 1:] != -100

                chosen_sum = (chosen_lp * chosen_mask).sum(dim=-1)
                rejected_sum = (rejected_lp * rejected_mask).sum(dim=-1)

                correct += (chosen_sum > rejected_sum).sum().item()
                total += chosen_sum.shape[0]

        self.policy.train()
        accuracy = correct / max(total, 1)
        return {"val_preference_accuracy": accuracy}

    # ── MLflow ──

    def _start_mlflow_run(self) -> Any:
        try:
            import mlflow

            mlflow_cfg = self.settings.config.mlflow
            if mlflow_cfg is None:
                return nullcontext()
            if hasattr(mlflow_cfg, "tracking_uri") and mlflow_cfg.tracking_uri:
                mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
            experiment_name = getattr(mlflow_cfg, "experiment_name", None) or getattr(mlflow_cfg, "experiment", None)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            return mlflow.start_run()
        except Exception as e:
            logger.warning(f"MLflow run 시작 실패: {e}")
            return nullcontext()

    def _mlflow_log_metric(self, key: str, value: float, step: int) -> None:
        if not self._is_main_process:
            return
        try:
            import mlflow
            mlflow.log_metric(key, value, step=step)
        except Exception:
            pass

    def _log_mlflow_params(self) -> None:
        try:
            import mlflow

            recipe = self.settings.recipe
            policy_spec = recipe.models["policy"]
            params = {
                "task": recipe.task,
                "algorithm": type(self.algorithm).__name__,
                "policy_class": policy_spec.class_path,
                "policy_pretrained": policy_spec.pretrained or "none",
                "dataset_source": recipe.data.source,
                "batch_size": recipe.data.dataloader.batch_size,
                "max_steps": self.max_steps or 0,
                "precision": recipe.training.precision,
                "policy_lr": self.optimizers["policy"].param_groups[0]["lr"],
            }
            if policy_spec.adapter is not None:
                params["adapter_method"] = policy_spec.adapter.method
                if policy_spec.adapter.r is not None:
                    params["adapter_r"] = policy_spec.adapter.r
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow params 로깅 실패: {e}")

    def _log_mlflow_summary(self, training_duration: float) -> None:
        try:
            import mlflow
            from mdp.utils.sanitize import sanitize_config

            mlflow.log_metrics({
                "training_duration_seconds": training_duration,
                "total_steps": self.global_step,
            })
            config_dict = sanitize_config(self.settings.model_dump())
            mlflow.log_dict(config_dict, "config/settings.json")

            # policy adapter를 MLflow artifact로 저장
            self._export_policy_artifact()
        except Exception as e:
            logger.warning(f"MLflow summary 로깅 실패: {e}")

    def _export_policy_artifact(self) -> None:
        """Policy 모델을 MLflow artifact로 저장한다. LoRA면 adapter만."""
        import json
        import shutil
        import tempfile

        import mlflow

        try:
            target = getattr(self.policy, "module", self.policy)
            has_adapter = hasattr(target, "peft_config")

            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)
                if has_adapter:
                    target.save_pretrained(output_dir)
                elif hasattr(target, "save_pretrained"):
                    target.save_pretrained(output_dir)
                else:
                    from safetensors.torch import save_file
                    save_file(target.state_dict(), output_dir / "model.safetensors")

                # tokenizer
                recipe = self.settings.recipe
                tokenizer_config = recipe.data.tokenizer
                if tokenizer_config:
                    pretrained = tokenizer_config.get("pretrained") if isinstance(tokenizer_config, dict) else getattr(tokenizer_config, "pretrained", None)
                    if pretrained:
                        try:
                            from transformers import AutoTokenizer
                            AutoTokenizer.from_pretrained(pretrained).save_pretrained(output_dir)
                        except Exception:
                            pass

                # recipe.yaml
                import yaml
                recipe_dict = recipe.model_dump(mode="json")
                (output_dir / "recipe.yaml").write_text(yaml.dump(recipe_dict, allow_unicode=True))

                mlflow.log_artifacts(tmp, "model")
                logger.info("Policy 모델을 MLflow artifact로 등록: model/")
        except Exception as e:
            logger.warning(f"Policy artifact 저장 실패: {e}")

    # ── Checkpoint (Resume) ──

    def _save_checkpoint(self, ckpt_dir: Path) -> None:
        """모든 모델의 상태를 저장한다."""
        import json

        for name, model in {**self.trainable, **self.frozen}.items():
            model_dir = ckpt_dir / name
            model_dir.mkdir(parents=True, exist_ok=True)

            unwrapped = getattr(model, "module", model)
            if self.strategy is not None and hasattr(self.strategy, "save_checkpoint"):
                self.strategy.save_checkpoint(unwrapped, model_dir / "model.safetensors")
            elif hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(model_dir)
            else:
                torch.save(unwrapped.state_dict(), model_dir / "model.pt")

            if name in self.optimizers:
                torch.save(self.optimizers[name].state_dict(), model_dir / "optimizer.pt")
            if name in self.schedulers:
                torch.save(self.schedulers[name].state_dict(), model_dir / "scheduler.pt")

        (ckpt_dir / "trainer_state.json").write_text(json.dumps({
            "global_step": self.global_step,
        }))
        if self.scaler.is_enabled():
            torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")

    def _maybe_resume(self) -> None:
        """체크포인트에서 모든 모델 상태를 복원한다."""
        import json

        job_config = getattr(self.settings.config, "job", None)
        if job_config is None:
            return
        resume_cfg = getattr(job_config, "resume", "disabled")
        if resume_cfg == "disabled":
            return

        # checkpoint 경로 해석
        storage = getattr(self.settings.config, "storage", None)
        ckpt_root = Path(storage.checkpoint_dir) if storage and hasattr(storage, "checkpoint_dir") and storage.checkpoint_dir else None
        if resume_cfg == "auto":
            if ckpt_root is None:
                return
            latest = ckpt_root / "latest"
            if not latest.exists():
                return
            ckpt_dir = latest.resolve()
        else:
            ckpt_dir = Path(resume_cfg)

        if not ckpt_dir.exists():
            logger.warning(f"Resume 체크포인트 없음: {ckpt_dir}")
            return

        for name, model in {**self.trainable, **self.frozen}.items():
            model_dir = ckpt_dir / name
            if not model_dir.exists():
                logger.warning(f"Resume: {name} 체크포인트 없음, 건너뜀")
                continue

            unwrapped = getattr(model, "module", model)
            adapter_path = model_dir / "adapter_model.safetensors"
            if adapter_path.exists() and hasattr(unwrapped, "load_adapter"):
                unwrapped.load_adapter(model_dir, adapter_name="default")
            elif (model_dir / "model.safetensors").exists():
                from safetensors.torch import load_file
                unwrapped.load_state_dict(load_file(model_dir / "model.safetensors"), strict=False)
            elif (model_dir / "model.pt").exists():
                unwrapped.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))

            if name in self.optimizers and (model_dir / "optimizer.pt").exists():
                self.optimizers[name].load_state_dict(
                    torch.load(model_dir / "optimizer.pt", map_location="cpu"))
            if name in self.schedulers and (model_dir / "scheduler.pt").exists():
                self.schedulers[name].load_state_dict(
                    torch.load(model_dir / "scheduler.pt", map_location="cpu"))

        state_path = ckpt_dir / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.global_step = state.get("global_step", 0)

        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists() and self.scaler.is_enabled():
            self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))

        logger.info(f"Resumed from {ckpt_dir} (step={self.global_step})")

    # ── 학습 루프 ──

    def train(self) -> dict[str, Any]:
        """RL 학습을 실행하고 결과를 반환한다."""
        # Expert Parallelism (전략 setup 전에 적용)
        if self.expert_parallel is not None:
            if self.strategy is not None:
                import torch.distributed as _dist
                if not _dist.is_initialized():
                    backend = getattr(self.strategy, "backend", "nccl")
                    _dist.init_process_group(backend=backend)
            self.policy = self.expert_parallel.setup(self.policy, self.device)
            self.trainable["policy"] = self.policy

        # Strategy setup
        if self.strategy is not None:
            trainable_names = set(self.trainable.keys())
            all_models = {**self.trainable, **self.frozen}
            wrapped = self.strategy.setup_models(all_models, self.device, trainable_names)
            for name in self.trainable:
                self.trainable[name] = wrapped[name]
            for name in self.frozen:
                self.frozen[name] = wrapped[name]
            self.policy = self.trainable["policy"]
        else:
            for name in self.trainable:
                self.trainable[name] = self.trainable[name].to(self.device)
            for name in self.frozen:
                self.frozen[name] = self.frozen[name].to(self.device)
            self.policy = self.trainable["policy"]

        # Gap 5: Resume from checkpoint
        self._maybe_resume()

        total_steps = self._estimate_total_steps()
        self._fire("on_train_start", total_steps=total_steps)
        start_time = time.time()

        mlflow_ctx = self._start_mlflow_run() if self._is_main_process else nullcontext()

        device_type = self.device.type if self.device.type != "mps" else "cpu"
        epoch_counter = 0
        train_iter = iter(self.train_loader)
        total_loss = 0.0
        num_steps = 0
        max_steps = self.max_steps or (len(self.train_loader) * (self.epochs or 1))

        needs_gen = getattr(self.algorithm, "needs_generation", False)

        gen_config = self.settings.recipe.generation
        self._generation_kwargs = {}
        if gen_config is not None:
            self._generation_kwargs = gen_config.model_dump(exclude_none=True) if hasattr(gen_config, "model_dump") else dict(gen_config)

        with mlflow_ctx:
            if self._is_main_process:
                self._log_mlflow_params()

            try:
                while self.global_step < max_steps:
                    if self._should_stop():
                        break

                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        epoch_counter += 1
                        # Gap 4: 분산 학습 시 에폭 경계에서 sampler 셔플 갱신
                        if hasattr(self.train_loader.sampler, "set_epoch"):
                            self.train_loader.sampler.set_epoch(epoch_counter)
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)

                    batch = self._move_to_device(batch)

                    if needs_gen:
                        step_loss = self._train_step_generation(batch, device_type)
                    else:
                        step_loss = self._train_step_offline(batch, device_type)

                    total_loss += step_loss
                    num_steps += 1

                    # step-level logging
                    self._mlflow_log_metric("loss", step_loss, self.global_step)

                    if (self.global_step + 1) % self.grad_accum_steps == 0:
                        self._fire(
                            "on_batch_end", step=self.global_step,
                            global_step=self.global_step,
                            metrics={"loss": step_loss},
                            model=self.policy,
                            optimizer=self.optimizers["policy"],
                            scheduler=self.schedulers.get("policy"),
                        )

                    # RL validation
                    if (
                        self.val_loader is not None
                        and self.val_check_interval > 0
                        and self.global_step > 0
                        and self.global_step % self.val_check_interval == 0
                    ):
                        val_metrics = self._run_rl_validation()
                        self._fire("on_validation_end", metrics=val_metrics)
                        self.last_metrics.update(val_metrics)

            finally:
                if self.strategy is not None:
                    try:
                        self.strategy.cleanup()
                    except Exception as e:
                        logger.warning(f"Strategy cleanup 실패: {e}")

                training_duration = time.time() - start_time
                if self._is_main_process:
                    self._log_mlflow_summary(training_duration)

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
                frozen_out = {name: self._forward_model(m, batch, role=name) for name, m in self.frozen.items()}
            trainable_out = {name: self._forward_model(m, batch, role=name) for name, m in self.trainable.items()}
            losses = self.algorithm(trainable_out, frozen_out, batch)

        self._backward_and_step(losses)
        self.global_step += 1
        return losses.get("policy", list(losses.values())[0]).item()

    def _train_step_generation(self, batch: dict, device_type: str) -> float:
        """GRPO / PPO — policy가 텍스트를 생성하고, 그 결과로 학습."""
        from mdp.training.losses.rl import compute_log_probs

        prompt_ids = batch["input_ids"]
        prompt_mask = batch.get("attention_mask")

        # Gap 1: group_size (K개 응답 생성)
        gen_kwargs = dict(self._generation_kwargs)
        K = gen_kwargs.pop("group_size", 1)

        # 1. Generation (no_grad)
        with torch.no_grad():
            if K > 1:
                expanded_ids = prompt_ids.repeat_interleave(K, dim=0)
                expanded_mask = prompt_mask.repeat_interleave(K, dim=0) if prompt_mask is not None else None
            else:
                expanded_ids = prompt_ids
                expanded_mask = prompt_mask

            # DDP wrapper를 벗겨 generate() 호출 (no_grad 안이므로 gradient 동기화 불필요)
            gen_model = getattr(self.policy, "module", self.policy)
            generated_ids = gen_model.generate(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                **gen_kwargs,
            )
            pad_id = getattr(getattr(gen_model, "config", None), "pad_token_id", None)
            gen_mask = (generated_ids != pad_id).long() if pad_id is not None else torch.ones_like(generated_ids)

        # 2. Old log_probs (update 전 policy 상태)
        with torch.no_grad():
            old_out = self._forward_model(self.policy, {"input_ids": generated_ids, "attention_mask": gen_mask}, role="policy")
            old_logits = old_out["logits"]
            old_log_probs = compute_log_probs(old_logits, generated_ids)

        # 3. Frozen forward + reward scoring
        gen_input = {"input_ids": generated_ids, "attention_mask": gen_mask}
        with torch.no_grad():
            frozen_out = {
                name: self._forward_model(m, gen_input, role=name)
                for name, m in self.frozen.items()
            }

        rewards = self._compute_rewards(frozen_out, generated_ids, gen_mask)

        gen_batch = {
            "input_ids": generated_ids,
            "attention_mask": gen_mask,
            "labels": generated_ids,
            "prompt_length": prompt_ids.shape[1],
            "old_log_probs": old_log_probs,
            "rewards": rewards,
            "group_size": K,
        }

        # 4. Mini-epoch update
        ppo_epochs = getattr(self.algorithm, "ppo_epochs", 1)
        last_loss = 0.0
        for _ in range(ppo_epochs):
            with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                trainable_out = {
                    name: self._forward_model(m, {"input_ids": generated_ids, "attention_mask": gen_mask}, role=name)
                    for name, m in self.trainable.items()
                }
                losses = self.algorithm(trainable_out, frozen_out, gen_batch)

            self._backward_and_step(losses, force_step=True)
            last_loss = losses.get("policy", list(losses.values())[0]).item()

        self.global_step += 1
        return last_loss

    def _compute_rewards(
        self,
        frozen_out: dict[str, dict],
        generated_ids: torch.Tensor,
        gen_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen reward model에서 scalar reward를 추출한다.

        Returns:
            (batch,) per-sequence scalar reward.

        추출 우선순위:
        1. reward model forward 결과의 "reward" 키 (명시적 scalar)
        2. reward model의 logits에서 마지막 유효 토큰의 첫 번째 값
        3. reward model 없으면 0 반환
        """
        if "reward" not in frozen_out:
            return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

        reward_out = frozen_out["reward"]

        # 1순위: 명시적 scalar reward (reward model이 "reward" 키를 반환)
        if "reward" in reward_out:
            r = reward_out["reward"]
            return r.view(-1) if r.dim() > 1 else r

        # 2순위: logits에서 마지막 유효 토큰 추출
        logits = reward_out.get("logits")
        if logits is None:
            return torch.zeros(generated_ids.shape[0], device=generated_ids.device)

        last_idx = gen_mask.sum(dim=-1).long() - 1
        batch_arange = torch.arange(logits.shape[0], device=logits.device)
        if logits.dim() == 3:
            return logits[batch_arange, last_idx, 0]
        elif logits.dim() == 2:
            return logits[batch_arange, last_idx]
        else:
            return logits.squeeze()

    def _backward_and_step(self, losses: dict[str, torch.Tensor], force_step: bool = False) -> None:
        """모델별 독립 backward + optimizer step."""
        # NaN/Inf loss 감지
        for name, loss in losses.items():
            if not torch.isfinite(loss):
                logger.warning("NaN/Inf loss detected in '%s', skipping step", name)
                for n in losses:
                    if n in self.optimizers:
                        self.optimizers[n].zero_grad(set_to_none=True)
                return

        accum = 1 if force_step else self.grad_accum_steps
        for name, loss in losses.items():
            scaled = loss / accum
            self.scaler.scale(scaled).backward()

        if force_step or (self.global_step + 1) % self.grad_accum_steps == 0:
            for name in losses:
                if name in self.optimizers:
                    self.scaler.unscale_(self.optimizers[name])
                    if self.grad_clip_norm is not None:
                        clip_grad_norm_(self.trainable[name].parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizers[name])
                    if name in self.schedulers:
                        self.schedulers[name].step()
            self.scaler.update()
            for name in losses:
                if name in self.optimizers:
                    self.optimizers[name].zero_grad()

    @staticmethod
    def _extract_logits(out):
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict):
            return out.get("logits", out.get("output"))
        return out

    def _forward_model(self, model: nn.Module, batch: dict, role: str = "policy") -> dict:
        """배치 형태에 따라 모델 forward를 수행한다.

        role에 따라 출력 키가 달라진다:
        - "policy", "reference" → {"logits": (batch, seq, vocab)} 또는 preference 형태
        - "value" → {"values": (batch, seq)} — scalar head 또는 LM head[:, :, 0]
        - "reward" → {"reward": (batch,)} 우선, fallback으로 {"logits": tensor}
        """
        result = {}

        # preference 형태 (DPO)
        if "chosen_input_ids" in batch:
            result["chosen_logits"] = self._extract_logits(model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch.get("chosen_attention_mask"),
            ))
        if "rejected_input_ids" in batch:
            result["rejected_logits"] = self._extract_logits(model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch.get("rejected_attention_mask"),
            ))

        # causal 형태
        if "input_ids" in batch:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            logits = self._extract_logits(out)

            if role == "value":
                # value model: logits → (batch, seq) scalar values
                if logits.dim() == 3 and logits.shape[-1] == 1:
                    result["values"] = logits.squeeze(-1)
                elif logits.dim() == 3:
                    result["values"] = logits[:, :, 0]
                else:
                    result["values"] = logits
            elif role == "reward":
                # reward model: 명시적 "reward" 키 우선, 없으면 logits 반환
                if isinstance(out, dict) and "reward" in out:
                    result["reward"] = out["reward"]
                result["logits"] = logits
            else:
                result["logits"] = logits

        return result
