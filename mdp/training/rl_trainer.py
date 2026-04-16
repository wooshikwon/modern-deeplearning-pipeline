"""RLTrainer — RL alignment 학습 루프.

SFT Trainer와 독립된 학습 루프. 내장 DPO/GRPO/PPO 외에도 `_component_` 패턴으로
외부 알고리즘(compute_loss + needs_generation/mini_epochs 규약)을 주입할 수 있다.
복수 모델(policy + frozen reference/critic/reward/value)을 관리하며,
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
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings
from mdp.training._common import (
    backward_and_step,
    create_callbacks,
    create_expert_parallel,
    create_strategy,
    detect_device,
    setup_amp,
)

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
        self.algorithm = self.resolver.resolve(recipe.rl.algorithm)

        # Device
        self.device = detect_device()

        # Training config
        self.max_steps = training.max_steps
        self.epochs = training.epochs
        self.grad_accum_steps = training.gradient_accumulation_steps
        self.grad_clip_norm = training.gradient_clip_max_norm
        self.compile_mode = training.compile

        # AMP
        self.amp_enabled, self.amp_dtype, self.scaler = setup_amp(training.precision, self.device)

        # 모델 분리: trainable vs frozen
        self.trainable: dict[str, nn.Module] = {}
        self.frozen: dict[str, nn.Module] = {}
        self.optimizers: dict[str, torch.optim.Optimizer] = {}
        self.schedulers: dict[str, Any] = {}
        self.scheduler_intervals: dict[str, str] = {}

        for name, spec in recipe.rl.models.items():
            model = models[name]
            if spec.get("optimizer") is not None:
                self.trainable[name] = model
                klass, kwargs = self.resolver.resolve_partial(spec["optimizer"])
                self.optimizers[name] = klass(model.parameters(), **kwargs)
                if spec.get("scheduler") is not None:
                    sched_config = dict(spec["scheduler"])
                    interval = sched_config.pop("interval", "step")
                    warmup_steps = sched_config.pop("warmup_steps", 0)
                    warmup_ratio = sched_config.pop("warmup_ratio", 0.0)
                    s_klass, s_kwargs = self.resolver.resolve_partial(sched_config)
                    scheduler = s_klass(self.optimizers[name], **s_kwargs)
                    if warmup_steps > 0 and warmup_ratio > 0:
                        raise ValueError(
                            "warmup_steps와 warmup_ratio를 동시에 지정할 수 없습니다. "
                            f"warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio}"
                        )
                    if warmup_ratio > 0:
                        total_steps = self._estimate_total_steps()
                        warmup_steps = int(total_steps * warmup_ratio)
                    if warmup_steps > 0:
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
                    self.scheduler_intervals[name] = interval
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
        self.callbacks = create_callbacks(recipe.callbacks, self.resolver)

        # Validation
        self.val_check_interval = getattr(training, "val_check_interval", 0)
        self.val_check_unit = getattr(training, "val_check_unit", "step")

        # State
        self.global_step = 0
        self.epoch_counter = 0
        self.last_metrics: dict[str, float] = {}
        self._recipe_dict = settings.recipe.model_dump()
        self._generation_kwargs: dict[str, Any] = {}

    def _estimate_total_steps(self) -> int:
        if self.max_steps:
            return self.max_steps
        steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
        return int(steps_per_epoch * (self.epochs or 1))

    def _fire(self, hook_name: str, **kwargs: Any) -> None:
        needs_ep_gather = (
            self.expert_parallel is not None
            and hook_name in ("on_validation_end", "on_batch_end")
        )
        if needs_ep_gather:
            self.expert_parallel.gather_experts(self.policy)

        kwargs["trainer"] = self
        kwargs.setdefault("model", self.policy)
        kwargs.setdefault("optimizer", self.optimizers.get("policy"))
        kwargs.setdefault("scheduler", self.schedulers.get("policy"))
        kwargs.setdefault("global_step", self.global_step)
        kwargs.setdefault("strategy", self.strategy)
        kwargs.setdefault("scaler", self.scaler)
        kwargs.setdefault("recipe_dict", self._recipe_dict)
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
        except Exception as e:
            logger.debug("MLflow metric 로깅 실패 (key=%s): %s", key, e)

    def _log_mlflow_params(self) -> None:
        try:
            import mlflow

            recipe = self.settings.recipe
            policy_spec = recipe.rl.models["policy"]
            params = {
                "task": recipe.task,
                "algorithm": type(self.algorithm).__name__,
                "policy_class": policy_spec.get("_component_", "unknown"),
                "policy_pretrained": policy_spec.get("pretrained", "none"),
                "dataset_source": recipe.data.dataset.get("source", "unknown"),
                "batch_size": recipe.data.dataloader.batch_size,
                "max_steps": self.max_steps or 0,
                "precision": recipe.training.precision,
                "policy_lr": self.optimizers["policy"].param_groups[0]["lr"],
            }
            policy_adapter = policy_spec.get("adapter")
            if policy_adapter is not None:
                params["adapter_component"] = policy_adapter.get("_component_", "unknown")
                if policy_adapter.get("r") is not None:
                    params["adapter_r"] = policy_adapter["r"]
            # Strategy — Config.compute.distributed
            dist = self.settings.config.compute.distributed
            if isinstance(dist, dict) and dist.get("strategy"):
                s = dist["strategy"]
                params["strategy"] = s.get("_component_", s) if isinstance(s, dict) else s
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow params 로깅 실패: {e}")

    def _log_mlflow_summary(self, training_duration: float, stopped_reason: str = "completed", policy_state_dict: "dict | None" = None) -> None:
        try:
            import mlflow
            from mdp.utils.sanitize import sanitize_config

            mlflow.log_metrics({
                "training_duration_seconds": training_duration,
                "total_steps": self.global_step,
            })
            mlflow.set_tag("stopped_reason", stopped_reason)
            config_dict = sanitize_config(self.settings.model_dump())
            mlflow.log_dict(config_dict, "config/settings.json")

            # policy adapter를 MLflow artifact로 저장
            self._export_policy_artifact(policy_state_dict=policy_state_dict)
        except Exception as e:
            logger.warning(f"MLflow summary 로깅 실패: {e}")

    def _gather_fsdp_policy_state_dict(self) -> "dict | None":
        """FSDP 모델의 full state dict를 all-rank 협력으로 수집한다.

        모든 랭크가 반드시 호출해야 한다 (NCCL all-gather가 내부에서 실행됨).
        rank0_only=True이므로 실제 weight는 rank 0에만 채워지고, 나머지는 빈 dict.
        FSDP가 아닌 경우 None을 반환한다.
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if not isinstance(self.policy, FSDP):
                return None
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        except Exception as e:
            logger.warning("FSDP state dict cooperative gather failed: %s", e)
            return None

        # NCCL collective — outside try/except so a raise here propagates to all ranks
        # instead of one rank silently returning None while others block in all-gather.
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, cfg):
            # All ranks participate in NCCL all-gather here.
            # rank0_only=True → result populated on rank 0 only; others get {}.
            state_dict = self.policy.state_dict()
        return state_dict if self._is_main_process else None

    def _export_policy_artifact(self, policy_state_dict: "dict | None" = None) -> None:
        """Policy 모델을 MLflow artifact로 저장한다. LoRA면 adapter만.

        policy_state_dict가 제공된 경우 FSDP cooperative gather로 수집한 full state dict를
        사용한다 (FSDP 모델에서 직접 save_pretrained를 호출하면 모든 랭크가 필요하므로).
        """
        import tempfile

        import mlflow

        try:
            target = getattr(self.policy, "module", self.policy)
            has_adapter = hasattr(target, "peft_config")

            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)

                if policy_state_dict is not None:
                    # FSDP path: use pre-gathered full state dict to avoid NCCL on rank 0 only.
                    if has_adapter:
                        # Extract adapter-only weights via PEFT helper (respects state_dict arg).
                        from peft import get_peft_model_state_dict
                        adapter_names = list(target.peft_config.keys())
                        adapter_name = adapter_names[0] if adapter_names else "default"
                        adapter_sd = get_peft_model_state_dict(target, state_dict=policy_state_dict, adapter_name=adapter_name)
                        from safetensors.torch import save_file
                        save_file(adapter_sd, str(output_dir / "adapter_model.safetensors"))
                        # Save adapter config for each adapter name.
                        peft_config = target.peft_config  # dict[adapter_name, PeftConfigMixin]
                        for adapter_name, cfg in peft_config.items():
                            cfg.save_pretrained(str(output_dir))
                    else:
                        from safetensors.torch import save_file
                        save_file(policy_state_dict, str(output_dir / "model.safetensors"))
                elif has_adapter:
                    target.save_pretrained(output_dir)
                elif hasattr(target, "save_pretrained"):
                    target.save_pretrained(output_dir)
                else:
                    from safetensors.torch import save_file
                    save_file(target.state_dict(), str(output_dir / "model.safetensors"))

                # tokenizer — collator _component_의 init_args에서 추출
                recipe = self.settings.recipe
                tokenizer_name = recipe.data.collator.get("tokenizer") if isinstance(recipe.data.collator, dict) else None
                if tokenizer_name:
                    try:
                        from transformers import AutoTokenizer
                        AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_dir)
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

    def _maybe_compute_baseline(self) -> dict[str, Any] | None:
        """Compute monitoring baseline after training using policy model."""
        try:
            from mdp.monitoring.baseline import compute_baseline
        except ImportError:
            return None

        monitoring_cfg = getattr(self.settings.recipe, "monitoring", None)
        if monitoring_cfg is None or not getattr(monitoring_cfg, "enabled", False):
            return None

        try:
            baseline = compute_baseline(
                train_dataloader=self.val_loader or self.train_loader,
                model=self.policy,
                config=self.settings,
            )

            if self._is_main_process:
                checkpoint_dir = Path(self.settings.config.storage.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                baseline_path = checkpoint_dir / "baseline.json"
                import json
                baseline_path.write_text(json.dumps(baseline, indent=2))
                logger.info("Monitoring baseline saved: %s", baseline_path)
                return {"baseline_saved": True, "baseline_path": str(baseline_path)}

            return None
        except Exception as e:
            logger.warning(f"Monitoring baseline 계산 실패: {e}")
            return None

    # ── Checkpoint (Resume) ──

    def _save_checkpoint(self, ckpt_dir: Path) -> None:
        """모든 모델의 상태를 저장한다."""
        import json

        for name, model in {**self.trainable, **self.frozen}.items():
            model_dir = ckpt_dir / name
            model_dir.mkdir(parents=True, exist_ok=True)

            if self.strategy is not None and hasattr(self.strategy, "save_checkpoint"):
                # FSDP + PeftModel(LoRA): adapter만 저장한다.
                # strategy.save_checkpoint는 rank0_only=True 방식으로 full state dict를 rank 0에 수집한다.
                # rank 0에서 수집된 전체 state dict 중 LoRA + modules_to_save 키만 필터링해
                # adapter_model.safetensors + adapter_config.json으로 저장한다.
                _saved_as_peft = False
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from peft import PeftModel as _PeftModel
                    _inner = getattr(model, "module", None)
                    if isinstance(model, FSDP) and isinstance(_inner, _PeftModel):
                        import json as _json
                        import tempfile as _tempfile
                        from safetensors import safe_open as _safe_open
                        from safetensors.torch import save_file as _save_file

                        with _tempfile.TemporaryDirectory() as _tmp:
                            _full_path = Path(_tmp) / "full.safetensors"
                            self.strategy.save_checkpoint(model, str(_full_path))

                            # rank0_only=True: rank-0에서만 파일이 생성된다
                            if _full_path.exists():
                                with _safe_open(str(_full_path), framework="pt", device="cpu") as _f:
                                    _full_sd = {k: _f.get_tensor(k) for k in _f.keys()}

                                # LoRA 가중치 + modules_to_save(value_head 등)만 필터
                                _adapter_tokens = {
                                    "lora_A", "lora_B",
                                    "lora_embedding_A", "lora_embedding_B",
                                    "modules_to_save",
                                }
                                _adapter_sd = {
                                    k: v for k, v in _full_sd.items()
                                    if any(tok in k for tok in _adapter_tokens)
                                }
                                _save_file(_adapter_sd, model_dir / "adapter_model.safetensors")

                                # adapter_config.json: peft_config에서 직렬화
                                _peft_cfg = next(iter(_inner.peft_config.values()))
                                (model_dir / "adapter_config.json").write_text(
                                    _json.dumps(_peft_cfg.to_dict(), indent=2)
                                )
                                logger.info(
                                    "Saved PEFT adapter only (FSDP+LoRA, %d keys): %s",
                                    len(_adapter_sd),
                                    model_dir,
                                )
                        _saved_as_peft = True
                except ImportError:
                    pass

                if not _saved_as_peft:
                    self.strategy.save_checkpoint(model, model_dir / "model.safetensors")
            else:
                unwrapped = getattr(model, "module", model)
                if hasattr(unwrapped, "save_pretrained"):
                    unwrapped.save_pretrained(model_dir)
                else:
                    torch.save(unwrapped.state_dict(), model_dir / "model.pt")

            if name in self.optimizers:
                torch.save(self.optimizers[name].state_dict(), model_dir / "optimizer.pt")
            if name in self.schedulers:
                torch.save(self.schedulers[name].state_dict(), model_dir / "scheduler.pt")

        (ckpt_dir / "trainer_state.json").write_text(json.dumps({
            "global_step": self.global_step,
            "epoch_counter": self.epoch_counter,
        }))
        if self.scaler.is_enabled():
            torch.save(self.scaler.state_dict(), ckpt_dir / "scaler.pt")
        import yaml
        (ckpt_dir / "recipe.yaml").write_text(yaml.dump(self._recipe_dict, allow_unicode=True))

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
                unwrapped.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))

            if name in self.optimizers and (model_dir / "optimizer.pt").exists():
                self.optimizers[name].load_state_dict(
                    torch.load(model_dir / "optimizer.pt", map_location="cpu", weights_only=True))
            if name in self.schedulers and (model_dir / "scheduler.pt").exists():
                self.schedulers[name].load_state_dict(
                    torch.load(model_dir / "scheduler.pt", map_location="cpu", weights_only=True))

        state_path = ckpt_dir / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.global_step = state.get("global_step", 0)
            self.epoch_counter = state.get("epoch_counter", 0)

        scaler_path = ckpt_dir / "scaler.pt"
        if scaler_path.exists() and self.scaler.is_enabled():
            self.scaler.load_state_dict(torch.load(scaler_path, map_location="cpu", weights_only=True))

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

        # Frozen 모델을 policy와 동일 dtype으로 캐스팅 (fp32/bf16/fp16 불일치 방지)
        if self.amp_enabled:
            for name in self.frozen:
                self.frozen[name] = self.frozen[name].to(dtype=self.amp_dtype)

        # Guard: device_map 모델은 학습 불가 (trainer.py와 대칭)
        for _name, _m in {**self.trainable, **self.frozen}.items():
            if hasattr(_m, "hf_device_map"):
                raise RuntimeError(
                    f"device_map으로 분산 배치된 모델({_name})은 학습에 사용할 수 없습니다. "
                    "device_map은 추론/서빙 전용이며, 학습에는 DDP/FSDP 전략을 사용하세요."
                )

        # Gradient Checkpointing (FSDP/DDP wrap 전에 적용해야 한다)
        gc_cfg = self.settings.recipe.training.gradient_checkpointing
        if gc_cfg:
            for name, model in self.trainable.items():
                # DDP/FSDP 이전이므로 .module 없음. PEFT 래퍼를 뚫고 실제 PreTrainedModel까지 내려간다.
                # PeftModel은 gradient_checkpointing_enable을 노출하지 않으므로
                # base_model.model (LoraModel → PreTrainedModel) 경로로 접근한다.
                base = getattr(model, "module", model)          # DDP/FSDP 대비 (현재 no-op)
                base = getattr(base, "base_model", base)        # PeftModel → LoraModel
                base = getattr(base, "model", base)             # LoraModel → PreTrainedModel
                if hasattr(base, "gradient_checkpointing_enable"):
                    # LoRA: 입력 텐서에 requires_grad가 없으면 GC recompute 구간에서 grad 소실.
                    if hasattr(base, "enable_input_require_grads"):
                        base.enable_input_require_grads()
                    # FSDP + GC: use_reentrant=True(기본값)는 FSDP가 all-gathered params를
                    # recompute forward 동안 조기에 해제하지 못해 전 레이어 파라미터가 동시
                    # 상주 → OOM. use_reentrant=False(비재진입)로 FSDP 호환성을 확보한다.
                    base.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                    logger.info("Gradient checkpointing enabled for %s (use_reentrant=False)", name)
                else:
                    logger.warning("gradient_checkpointing_enable not found on %s (%s)", name, type(base).__name__)

        # Strategy setup
        if self.strategy is not None:
            trainable_names = set(self.trainable.keys())
            all_models = {**self.trainable, **self.frozen}
            wrapped = self.strategy.setup_models(all_models, self.device, trainable_names, optimizers=self.optimizers)
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

        # Training/eval mode 명시 설정
        # HF 모델은 from_pretrained() 후 eval() 상태로 반환된다.
        # FSDP 래핑은 training mode를 변경하지 않으므로 여기서 명시적으로 설정한다.
        # GC guard: LlamaModel.forward의 `self.gradient_checkpointing and self.training`에서
        # self.training이 False이면 GC가 비활성화되어 모든 activation이 저장된다 → OOM.
        for model in self.trainable.values():
            model.train()
        for model in self.frozen.values():
            model.eval()

        # ── FSDP 샤딩 + 메모리 베이스라인 진단 ──
        # FSDP wrap 직후, compile/학습 이전 시점의 순수 모델 메모리를 측정한다.
        #
        # 메모리 구성 (FULL_SHARD policy + NO_SHARD frozen, 8B bf16, 4 GPU):
        #   policy shard  : 16 GiB / world_size  (FULL_SHARD, GPU당 param 조각)
        #   frozen replica: frozen 모델 전체 × dtype bytes  (NO_SHARD, GPU당 복제본)
        #   예: policy 4 GiB + reference 16 GiB = ~20 GiB baseline
        #
        # 확인 항목:
        #   1. allocated vs expected_total: policy 샤딩 + frozen 복제본 합계와 비교
        #   2. trainable_params: LoRA 어댑터만 requires_grad여야 함 (~0.5% of total)
        if torch.cuda.is_available():
            _rank = int(os.environ.get("RANK", "0"))
            _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            _world_size = int(os.environ.get("WORLD_SIZE", "1"))
            _mem_alloc = torch.cuda.memory_allocated(_local_rank) / 1024 ** 3
            _mem_resv = torch.cuda.memory_reserved(_local_rank) / 1024 ** 3
            _trainable_params = sum(
                p.numel() for p in self.policy.parameters() if p.requires_grad
            )
            _total_params = sum(p.numel() for p in self.policy.parameters())
            # frozen 모델은 NO_SHARD → 각 GPU가 전체 복제본을 보유
            _frozen_replica_gib = sum(
                sum(p.numel() * p.element_size() for p in m.parameters()) / 1024 ** 3
                for m in self.frozen.values()
            )
            logger.info(
                "FSDP shard baseline: rank=%d | allocated=%.2f GiB | reserved=%.2f GiB"
                " | policy trainable=%.1fM params | frozen replica=%.1f GiB (NO_SHARD)",
                _rank, _mem_alloc, _mem_resv,
                _trainable_params / 1e6, _frozen_replica_gib,
            )
            if _rank == 0:
                # use_orig_params=True에서 total_orig는 unsharded 원본 크기를 반환한다.
                _policy_shard_gib = 16.0 / _world_size  # 8B bf16 ÷ world_size
                _expected_total_gib = _policy_shard_gib + _frozen_replica_gib
                _ratio = _mem_alloc / _expected_total_gib if _expected_total_gib else 0
                logger.info(
                    "FSDP shard baseline: expected policy shard=%.1f GiB"
                    " + frozen replica=%.1f GiB = %.1f GiB total."
                    " actual=%.2f GiB (%.1f×). >>3× → policy shard 미적용 의심.",
                    _policy_shard_gib, _frozen_replica_gib,
                    _expected_total_gib, _mem_alloc, _ratio,
                )
                if _total_params / 1e9 > 1.0 and _trainable_params / _total_params > 0.1:
                    logger.warning(
                        "FSDP shard baseline: trainable ratio %.1f%% — LoRA freeze 미적용 의심."
                        " 기대값: LoRA만 trainable (전체의 ~0.5%% 수준)",
                        100 * _trainable_params / _total_params,
                    )

        # torch.compile — must be AFTER FSDP wrapping, trainable models only
        if self.compile_mode:
            mode = self.compile_mode if isinstance(self.compile_mode, str) else "default"
            for name in list(self.trainable.keys()):
                self.trainable[name] = torch.compile(self.trainable[name], mode=mode)
            self.policy = self.trainable["policy"]
            logger.info("torch.compile applied to trainable models (mode=%s)", mode)

        # Gap 5: Resume from checkpoint
        self._maybe_resume()
        sampler = getattr(self.train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch_counter)

        total_steps = self._estimate_total_steps()
        self._fire("on_train_start", total_steps=total_steps)
        start_time = time.time()

        mlflow_ctx = self._start_mlflow_run() if self._is_main_process else nullcontext()
        mlflow_run_id: str | None = None

        device_type = self.device.type if self.device.type != "mps" else "cpu"
        train_iter = iter(self.train_loader)
        total_loss = 0.0
        num_steps = 0
        _epoch_steps = int(len(self.train_loader) * (self.epochs or 1))
        max_steps = min(self.max_steps, _epoch_steps) if self.max_steps else _epoch_steps

        needs_gen = getattr(self.algorithm, "needs_generation", False)
        if needs_gen and self.grad_accum_steps > 1:
            logger.warning(
                "Generation path에서 grad_accum_steps=%d → 1로 강제합니다. "
                "force_step=True로 매 mini-epoch마다 optimizer step을 실행하므로 "
                "gradient accumulation이 무의미합니다.",
                self.grad_accum_steps,
            )
            self.grad_accum_steps = 1

        gen_config = self.settings.recipe.rl.generation if self.settings.recipe.rl else None
        self._generation_kwargs = {}
        if gen_config is not None:
            self._generation_kwargs = gen_config.model_dump(exclude_none=True) if hasattr(gen_config, "model_dump") else dict(gen_config)

        batch_idx = 0

        with mlflow_ctx as mlflow_run:
            if self._is_main_process:
                if mlflow_run is not None and hasattr(mlflow_run, "info"):
                    mlflow_run_id = mlflow_run.info.run_id
                self._log_mlflow_params()

            try:
                self._fire("on_epoch_start", epoch=self.epoch_counter)

                while self.global_step < max_steps:
                    if self._should_stop():
                        break

                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        # Epoch-final: 잔여 gradient flush (offline path only)
                        # generation path는 grad_accum_steps=1이므로 잔여 없음
                        if not needs_gen and num_steps > 0 and num_steps % self.grad_accum_steps != 0:
                            for name, opt in self.optimizers.items():
                                self.scaler.unscale_(opt)
                                if self.grad_clip_norm is not None and name in self.trainable:
                                    clip_grad_norm_(self.trainable[name].parameters(), self.grad_clip_norm)
                            has_inf = any(
                                torch.isinf(p.grad).any() or torch.isnan(p.grad).any()
                                for m in self.trainable.values()
                                for p in m.parameters()
                                if p.grad is not None
                            )
                            if has_inf:
                                logger.warning("NaN/Inf gradient in residual flush, skipping step")
                                for opt in self.optimizers.values():
                                    opt.zero_grad(set_to_none=True)
                                self.scaler.update()
                            else:
                                for name, opt in self.optimizers.items():
                                    self.scaler.step(opt)
                                    if self.scheduler_intervals.get(name) == "step":
                                        sched = self.schedulers.get(name)
                                        if sched is not None:
                                            sched.step()
                                self.scaler.update()
                                for opt in self.optimizers.values():
                                    opt.zero_grad(set_to_none=True)
                                self.global_step += 1
                                self._fire(
                                    "on_batch_end", step=self.global_step,
                                    epoch=self.epoch_counter,
                                    global_step=self.global_step,
                                    metrics={"loss": step_loss},
                                    model=self.policy,
                                    optimizer=self.optimizers["policy"],
                                    scheduler=self.schedulers.get("policy"),
                                )

                        self._fire("on_epoch_end", epoch=self.epoch_counter, metrics={"loss": total_loss / max(num_steps, 1)})
                        self.epoch_counter += 1

                        # Epoch-level scheduler stepping
                        for name, sched in self.schedulers.items():
                            if self.scheduler_intervals.get(name) == "epoch":
                                sched.step()

                        # Epoch-based validation
                        if (
                            self.val_loader is not None
                            and self.val_check_interval > 0
                            and self.val_check_unit == "epoch"
                            and self.epoch_counter > 0
                            and self.epoch_counter % int(self.val_check_interval) == 0
                        ):
                            self._fire("on_validation_start", epoch=self.epoch_counter)
                            val_metrics = self._run_rl_validation()
                            self._fire("on_validation_end", epoch=self.epoch_counter, metrics=val_metrics)
                            self.last_metrics.update(val_metrics)

                        self._fire("on_epoch_start", epoch=self.epoch_counter)

                        # 분산 학습 시 에폭 경계에서 sampler 셔플 갱신
                        sampler = getattr(self.train_loader, "sampler", None)
                        if sampler is not None and hasattr(sampler, "set_epoch"):
                            sampler.set_epoch(self.epoch_counter)
                        train_iter = iter(self.train_loader)
                        batch_idx = 0
                        batch = next(train_iter)

                    batch = self._move_to_device(batch)
                    self._fire("on_batch_start", step=self.global_step)

                    if needs_gen:
                        step_loss = self._train_step_generation(batch, device_type)
                    else:
                        step_loss = self._train_step_offline(batch, device_type, batch_idx)

                    batch_idx += 1
                    if step_loss is None:
                        continue
                    total_loss += step_loss
                    num_steps += 1

                    # step-level logging
                    self._mlflow_log_metric("loss", step_loss, self.global_step)

                    if batch_idx % self.grad_accum_steps == 0:
                        self._fire(
                            "on_batch_end", step=self.global_step,
                            epoch=self.epoch_counter,
                            global_step=self.global_step,
                            metrics={"loss": step_loss},
                            model=self.policy,
                            optimizer=self.optimizers["policy"],
                            scheduler=self.schedulers.get("policy"),
                        )

                    # RL step-based validation
                    if (
                        self.val_loader is not None
                        and self.val_check_interval > 0
                        and self.val_check_unit == "step"
                        and self.global_step > 0
                        and self.global_step % int(self.val_check_interval) == 0
                    ):
                        self._fire("on_validation_start", epoch=self.epoch_counter)
                        val_metrics = self._run_rl_validation()
                        self._fire("on_validation_end", epoch=self.epoch_counter, metrics=val_metrics)
                        self.last_metrics.update(val_metrics)

            finally:
                avg_loss = total_loss / max(num_steps, 1)
                self._fire("on_train_end", metrics={"loss": avg_loss})

                # stopped_reason 결정
                if self._should_stop():
                    stopped_reason = "early_stopping"
                elif self.global_step >= max_steps:
                    stopped_reason = "max_steps"
                else:
                    stopped_reason = "completed"

                training_duration = time.time() - start_time
                # FSDP artifact export needs all ranks for parameter all-gather.
                # Must happen BEFORE strategy.cleanup() which destroys the process group.
                # Skip if an exception is in flight (e.g. OOM during forward): FSDP
                # training state would be FORWARD, not IDLE, causing AssertionError
                # inside state_dict() pre-hook before any NCCL call can start.
                import sys as _sys
                _training_exc = _sys.exc_info()[0]
                if _training_exc is not None:
                    logger.warning(
                        "Training aborted (%s) — skipping FSDP state dict gather",
                        _training_exc.__name__,
                    )
                    fsdp_policy_sd = None
                else:
                    fsdp_policy_sd = self._gather_fsdp_policy_state_dict()

                if self.strategy is not None:
                    try:
                        self.strategy.cleanup()
                    except Exception as e:
                        logger.warning(f"Strategy cleanup 실패: {e}")

                if self._is_main_process:
                    self._log_mlflow_summary(training_duration, stopped_reason, policy_state_dict=fsdp_policy_sd)

        # Monitoring baseline (policy 모델 사용)
        monitoring = self._maybe_compute_baseline()

        metrics = {"loss": avg_loss}
        metrics.update(self.last_metrics)

        result = {
            "metrics": metrics,
            "training_duration_seconds": training_duration,
            "total_steps": self.global_step,
            "total_epochs": self.epoch_counter,
            "stopped_reason": stopped_reason,
            "algorithm": type(self.algorithm).__name__,
        }
        if monitoring is not None:
            result["monitoring"] = monitoring
        if mlflow_run_id is not None:
            result["run_id"] = mlflow_run_id
        return result

    # ── Step 실행 ──

    def _forward_preference(self, models: dict[str, nn.Module], batch: dict) -> dict[str, dict]:
        """Preference 배치를 chosen/rejected로 분리하여 causal forward 2회 호출."""
        out = {}
        for name, m in models.items():
            chosen_out = self._forward_model(
                m,
                {"input_ids": batch["chosen_input_ids"],
                 "attention_mask": batch.get("chosen_attention_mask")},
                role=name,
            )
            rejected_out = self._forward_model(
                m,
                {"input_ids": batch["rejected_input_ids"],
                 "attention_mask": batch.get("rejected_attention_mask")},
                role=name,
            )
            out[name] = {
                "chosen_logits": chosen_out["logits"],
                "rejected_logits": rejected_out["logits"],
            }
        return out

    def _train_step_offline(self, batch: dict, device_type: str, batch_idx: int) -> float:
        """DPO — 데이터가 이미 완성된 경로."""
        is_preference = "chosen_input_ids" in batch
        with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            if is_preference:
                with torch.no_grad():
                    frozen_out = self._forward_preference(self.frozen, batch)
                trainable_out = self._forward_preference(self.trainable, batch)
            else:
                with torch.no_grad():
                    frozen_out = {name: self._forward_model(m, batch, role=name) for name, m in self.frozen.items()}
                trainable_out = {name: self._forward_model(m, batch, role=name) for name, m in self.trainable.items()}
            losses = self.algorithm.compute_loss(trainable_out, frozen_out, batch)

        step_schedulers = {
            n: s for n, s in self.schedulers.items()
            if self.scheduler_intervals.get(n) == "step"
        }
        result = backward_and_step(
            losses=losses,
            optimizers=self.optimizers,
            schedulers=step_schedulers,
            scaler=self.scaler,
            trainable_models=self.trainable,
            grad_accum_steps=self.grad_accum_steps,
            at_accum_boundary=(batch_idx + 1) % self.grad_accum_steps == 0,
            grad_clip_norm=self.grad_clip_norm,
        )
        if result is None:
            return None
        if result is True:
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
        mini_epochs = getattr(self.algorithm, "mini_epochs", 1)
        last_loss = 0.0
        all_nan = True
        for _ in range(mini_epochs):
            with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                trainable_out = {
                    name: self._forward_model(m, {"input_ids": generated_ids, "attention_mask": gen_mask}, role=name)
                    for name, m in self.trainable.items()
                }
                losses = self.algorithm.compute_loss(trainable_out, frozen_out, gen_batch)

            step_schedulers = {
                n: s for n, s in self.schedulers.items()
                if self.scheduler_intervals.get(n) == "step"
            }
            result = backward_and_step(
                losses=losses,
                optimizers=self.optimizers,
                schedulers=step_schedulers,
                scaler=self.scaler,
                trainable_models=self.trainable,
                grad_accum_steps=self.grad_accum_steps,
                at_accum_boundary=False,
                grad_clip_norm=self.grad_clip_norm,
                force_step=True,
            )
            if result is None:
                break
            all_nan = False
            last_loss = losses.get("policy", list(losses.values())[0]).item()

        if all_nan:
            return None
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

    @staticmethod
    def _extract_logits(out):
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict):
            return out.get("logits", out.get("output"))
        return out

    def _forward_model(self, model: nn.Module, batch: dict, role: str = "policy") -> dict:
        """Causal forward를 수행한다.

        role에 따라 출력 키가 달라진다:
        - "policy", "reference" → {"logits": (batch, seq, vocab)}
        - "value" → {"values": (batch, seq)} — scalar head 또는 LM head[:, :, 0]
        - "reward" → {"reward": (batch,)} 우선, fallback으로 {"logits": tensor}

        Preference 배치(chosen/rejected)는 caller에서 분리하여 2회 호출한다.
        """
        result = {}

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
