"""RLTrainer — RL alignment 학습 루프.

SFT Trainer와 독립된 학습 루프. 내장 DPO/GRPO/PPO 외에도 `_component_` 패턴으로
외부 알고리즘(compute_loss + needs_generation/mini_epochs 규약)을 주입할 수 있다.
복수 모델(policy + frozen reference/critic/reward/value)을 관리하며,
optimizer는 trainable 모델별로 독립 운용한다.
"""

from __future__ import annotations

import logging
import os
import signal
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
    aggregate_checkpoint_stats,
    backward_and_step,
    create_expert_parallel,
    create_strategy,
    detect_device,
    setup_amp,
)
from mdp.training._base import BaseTrainer
from mdp.training._checkpoint import (
    export_model_artifact,
    gather_fsdp_state_dict,
    load_checkpoint,
)
from mdp.training._mlflow_logging import (
    log_epoch_metrics,
    log_static_params,
    log_step_metrics,
    log_summary,
)
from mdp.training._features import (
    extract_hidden_states_and_head,
    forward_model as _features_forward_model,
)
from mdp.training._schedulers import (
    create_scheduler_with_warmup,
    parse_warmup_config,
)
from mdp.training.callbacks.base import BaseCallback
from mdp.training.callbacks.early_stopping import EarlyStopping
from mdp.training.callbacks.ema import EMACallback

logger = logging.getLogger(__name__)

# [PROBE 2026-04-23] 일회성 진단 플래그 (grad_norm=0.00 원인 추적용).
_probe_extract_fired = False


class RLTrainer(BaseTrainer):
    """RL alignment 학습 루프."""

    def __init__(
        self,
        settings: Settings,
        models: dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: list[BaseCallback] | None = None,
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
                    warmup = parse_warmup_config(sched_config, self._estimate_total_steps())
                    s_klass, s_kwargs = self.resolver.resolve_partial(sched_config)
                    base_scheduler = s_klass(self.optimizers[name], **s_kwargs)
                    self.schedulers[name] = create_scheduler_with_warmup(
                        self.optimizers[name], base_scheduler, warmup
                    )
                    self.scheduler_intervals[name] = warmup.interval
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
        self.callbacks = list(callbacks) if callbacks else []
        if training.early_stopping is not None:
            self.callbacks.append(EarlyStopping(**training.early_stopping.model_dump()))
        if training.ema is not None:
            self.callbacks.append(EMACallback(**training.ema.model_dump()))

        # Validation
        self.val_check_interval = getattr(training, "val_check_interval", 0)
        self.val_check_unit = getattr(training, "val_check_unit", "step")

        # State
        self.global_step = 0
        self.epoch_counter = 0
        self.last_metrics: dict[str, float] = {}
        self._recipe_dict = settings.recipe.model_dump()
        self._generation_kwargs: dict[str, Any] = {}

        # Signal handling — Trainer와 동일한 패턴.
        # SIGTERM/SIGINT 수신 시 while 루프의 조건에서 break하여 finally 블록이
        # 정상 실행되도록 한다.
        self._stop_requested: bool = False
        self._stop_signal_name: str | None = None

    # ── BaseTrainer abstract method 구현 ──

    def _optimizer_for_progress_log(self) -> "torch.optim.Optimizer | None":
        """RLTrainer 는 policy optimizer 의 첫 param_group 에서 LR 을 읽는다."""
        return self.optimizers.get("policy")

    def _algorithm_label(self) -> str:
        """RLTrainer 는 algorithm 클래스명을 label 로 사용한다."""
        return type(self.algorithm).__name__

    def _collect_mlflow_params(self) -> None:
        """Run 시작 시 static param 로깅을 공용 헬퍼에 위임한다.

        `_mlflow_logging.log_static_params`가 원칙 2(optimizer 인스턴스 상태는 param으로
        내보내지 않음)·원칙 3(multi-group slash 네이밍)를 일괄 책임진다. 본 래퍼는
        호출 타이밍과 rank 가드(caller 쪽 `_is_main_process`) 외에는 어떤 결정도
        내리지 않는다 — Trainer와 동일한 대칭 계약.
        """
        log_static_params(self.settings.recipe, self.settings)

    def _checkpoint_state(self) -> dict:
        """현재 학습 상태를 dict로 직렬화한다.

        RLTrainer는 trainable + frozen 모델 전체를 ``"models"`` 키로 포함한다.
        per-model 서브디렉토리 (``{name}/model.pt``, ``{name}/optimizer.pt``,
        ``{name}/scheduler.pt``) 구조로 저장된다.

        FSDP 환경에서는 ``gather_fsdp_state_dict`` 를 경유한다 — 반드시 모든 rank에서
        호출해야 하는 NCCL collective 이며, rank 0 에만 실제 weights 가 채워진다.
        non-rank-0 FSDP 에서는 해당 모델 항목이 ``"models"`` 에 포함되지 않는다.

        반환 dict는 ``save_checkpoint``로 전달되어 I/O를 담당한다.
        """
        trainer_state: dict[str, Any] = {
            "global_step": self.global_step,
            "epoch_counter": self.epoch_counter,
        }
        optimizers = {
            name: opt.state_dict()
            for name, opt in self.optimizers.items()
        }
        schedulers = {
            name: sched.state_dict()
            for name, sched in self.schedulers.items()
        }

        # Model weights — trainable + frozen 전부. FSDP 환경: gather_fsdp_state_dict는
        # NCCL collective이므로 모든 rank에서 호출. rank-0에서만 non-None을 반환.
        # non-FSDP 또는 FSDP rank-0 이외에서 None을 반환하면 "models" 항목에 추가하지 않음.
        models: dict[str, Any] = {}
        for name, model in {**self.trainable, **self.frozen}.items():
            fsdp_sd = gather_fsdp_state_dict(model, self._is_main_process)
            if fsdp_sd is not None:
                # FSDP rank-0: full state dict already gathered
                models[name] = {"state_dict_pt": fsdp_sd}
            else:
                # non-FSDP (or FSDP non-rank-0 → skip to avoid writing empty weights)
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    if isinstance(model, FSDP):
                        # non-rank-0 FSDP: fsdp_sd=None means we're not rank-0; skip
                        continue
                except Exception:
                    pass
                unwrapped = getattr(model, "module", model)
                models[name] = {"state_dict_pt": unwrapped.state_dict()}

        return {
            "trainer_state": trainer_state,
            "optimizers": optimizers,
            "schedulers": schedulers,
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "recipe_dict": self._recipe_dict,
            "models": models,
        }

    def _load_checkpoint_state(self, state: dict) -> None:
        """``load_checkpoint``가 반환한 state dict로 학습 상태를 복원한다.

        복원 순서 (순서 민감):
        1. 모델 weights (adapter → safetensors → pt 우선순위, trainable + frozen 전부)
        2. optimizer state_dict (per-model, trainable only)
        3. scheduler state_dict (per-model, trainable only)
        4. GradScaler state_dict
        5. trainer scalar state (global_step, epoch_counter)

        :param state: ``load_checkpoint(ckpt_dir)``가 반환한 dict.
        """
        ckpt_dir: Path = state["ckpt_dir"]
        logger.info(f"Resumed from {ckpt_dir}")

        # 1. Model weights — trainable + frozen 전부 복원
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
                unwrapped.load_state_dict(
                    load_file(model_dir / "model.safetensors"), strict=False
                )
            elif (model_dir / "model.pt").exists():
                unwrapped.load_state_dict(
                    torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
                )

            # 2. Optimizer
            if name in self.optimizers and (model_dir / "optimizer.pt").exists():
                self.optimizers[name].load_state_dict(
                    torch.load(model_dir / "optimizer.pt", map_location="cpu", weights_only=True)
                )

            # 3. Scheduler
            if name in self.schedulers and (model_dir / "scheduler.pt").exists():
                self.schedulers[name].load_state_dict(
                    torch.load(model_dir / "scheduler.pt", map_location="cpu", weights_only=True)
                )

        # 4. GradScaler
        scaler_sd = state.get("scaler")
        if scaler_sd is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_sd)

        # 5. Trainer scalar state
        trainer_state = state.get("trainer_state")
        if trainer_state is not None:
            self.global_step = trainer_state.get("global_step", 0)
            self.epoch_counter = trainer_state.get("epoch_counter", 0)

        logger.info(f"Resumed from {ckpt_dir} (step={self.global_step})")

    def _fire(self, hook_name: str, **kwargs: Any) -> None:
        _do_ep_gather = (
            self.expert_parallel is not None
            and hook_name in ("on_validation_end", "on_batch_end")
        )
        if _do_ep_gather:
            assert self.expert_parallel is not None  # guaranteed by _do_ep_gather
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

        if _do_ep_gather:
            assert self.expert_parallel is not None  # guaranteed by _do_ep_gather
            self.expert_parallel.scatter_experts(self.policy, self.device)

    def _run_rl_validation(self) -> dict[str, float]:
        """validation: generation+reward (GRPO/PPO) 또는 preference accuracy (DPO)."""
        if self.val_loader is None:
            return {}
        assert self.val_loader is not None  # Pyright instance-attr narrowing helper

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

                reward_out = _features_forward_model(
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
        assert self.val_loader is not None  # callers always guard with val_loader is not None
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
                chosen_out = _features_forward_model(
                    self.policy,
                    {"input_ids": chosen_ids, "attention_mask": batch.get("chosen_attention_mask")},
                    role="policy",
                )
                rejected_out = _features_forward_model(
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

    def _log_mlflow_summary(
        self,
        training_duration: float,
        stopped_reason: str = "completed",
        policy_state_dict: "dict | None" = None,
    ) -> None:
        """Run 종료 시 summary 로깅을 공용 헬퍼에 위임한다.

        Trainer와의 대칭을 위해 `final_metrics=self.last_metrics`를 포함한다.
        `last_metrics`가 비어 있으면 `log_summary` 내부가 guard로 skip
        하므로 caller는 그대로 넘기면 된다. Checkpoint 집계 결과는 기존대로
        `self._checkpoints_saved`에 보존해 `train()` 반환 dict가 그대로 쓸 수 있도록
        한다.

        policy artifact export는 FSDP all-gather·tempdir 수명 관리가 있어
        `log_summary`의 범용 `artifact_dirs` 경로로 흡수하지 않고 별도로 호출한다.
        """
        # Checkpoint 집계 — `_common.aggregate_checkpoint_stats`가 Trainer와 동일한
        # duck typing 규칙을 단일 구현으로 제공한다. RL 구성은 Critic + Policy 각각에
        # ModelCheckpoint를 붙일 수 있어 여러 콜백이 공존해도 `saved_checkpoints`
        # 속성을 가진 콜백만 누적된다. `monitor_hint`는 아래 zero-warning에서 사용.
        checkpoint_stats = aggregate_checkpoint_stats(self.callbacks)
        total_checkpoints, _, monitor_hint = checkpoint_stats
        self._checkpoints_saved = total_checkpoints

        # sanitized_config — Trainer와 동일 출처·동일 파일 경로.
        try:
            from mdp.utils.sanitize import sanitize_config
            sanitized_config = sanitize_config(self.settings.model_dump())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"sanitize_config 실패: {e}")
            sanitized_config = None

        # Peak memory metric — rank 0의
        # `torch.cuda.max_memory_allocated()`를 GiB 단위로 summary에 기록한다.
        # CUDA 미사용 / 예외 발생 시 조용히 skip한다.
        extra_summary = self._peak_memory_summary_extra()

        log_summary(
            training_duration_seconds=training_duration,
            total_steps=self.global_step,
            stopped_reason=stopped_reason,
            final_metrics=self.last_metrics,
            checkpoint_stats=checkpoint_stats,
            sanitized_config=sanitized_config,
            artifact_dirs=(),
            extra=extra_summary,
        )

        # policy adapter를 MLflow artifact로 저장 — FSDP cooperative gather 결과를 사용.
        if self._is_main_process:
            try:
                export_model_artifact(
                    self.policy,
                    self.settings,
                    policy_state_dict=policy_state_dict,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Policy artifact 등록 실패: {e}")

        # zero-checkpoint warning — 산출물 0인 run을 사용자가 놓치지 않도록.
        if total_checkpoints == 0:
            logger.warning(
                "체크포인트가 하나도 저장되지 않았습니다. monitor=[%s] 설정을 확인하세요.",
                monitor_hint,
            )

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

    def _maybe_resume(self) -> None:
        """체크포인트에서 모든 모델 상태를 복원한다."""
        job_config = getattr(self.settings.config, "job", None)
        if job_config is None:
            return
        resume_cfg = getattr(job_config, "resume", "disabled")
        if resume_cfg == "disabled":
            return

        # checkpoint 경로 해석
        storage = getattr(self.settings.config, "storage", None)
        ckpt_root = (
            Path(storage.checkpoint_dir)
            if storage and hasattr(storage, "checkpoint_dir") and storage.checkpoint_dir
            else None
        )
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

        state = load_checkpoint(ckpt_dir)
        self._load_checkpoint_state(state)

    # ── 학습 루프 ──

    def train(self) -> dict[str, Any]:
        """RL 학습을 실행하고 결과를 반환한다."""
        # Signal handlers — SIGTERM/SIGINT 수신 시 graceful stop 요청.
        # Trainer와 대칭. while 루프 조건(`not self._stop_requested`)으로 현재 step
        # 경계에서 break하여 finally 블록(cleanup + on_train_end + summary)이
        # 정상적으로 실행되게 한다. MLflow zombie run 방지의 핵심.
        self._stop_requested = False
        self._stop_signal_name = None
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def _signal_handler(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            # 첫 시그널만 기록한다. SIGTERM 수신 후 사용자가 Ctrl+C를 누르거나
            # 외부가 이중 신호를 보내도 `_stop_signal_name`이 덮어쓰이지 않게 하여
            # stopped_reason tag가 실제 종료 원인을 정확히 반영하도록 보장한다.
            # Trainer._signal_handler와 대칭.
            if not self._stop_requested:
                logger.warning(
                    "Signal %s received, requesting graceful stop at next step boundary.",
                    sig_name,
                )
                self._stop_signal_name = sig_name
                self._stop_requested = True
            else:
                logger.warning(
                    "Signal %s received (already stopping due to %s).",
                    sig_name, self._stop_signal_name,
                )

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        # Duration semantics (C-3): epochs와 max_steps가 둘 다 지정되었을 때
        # "먼저 도달한 조건에서 종료"라는 암묵적 규칙을 학습 시작 시 1회 기록한다.
        # 하나만 지정된 경우 의도가 명확하므로 로그를 찍지 않는다(노이즈 방지).
        if self.epochs is not None and self.max_steps is not None:
            logger.info(
                "epochs=%.2f, max_steps=%d 모두 지정됨. 먼저 도달한 조건에서 종료됩니다.",
                self.epochs,
                self.max_steps,
            )

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

        # ── 메모리 베이스라인 진단 (strategy 조건부) ──
        # Wrap 직후, compile/학습 이전 시점의 순수 모델 메모리를 측정한다.
        #
        # Strategy별 기대값:
        #   DDPStrategy       : 모델 전체 복제본 — shard 미적용이 설계상 정상
        #   FSDPStrategy      : policy는 FULL_SHARD → GPU당 param 조각, frozen은 NO_SHARD → GPU당 복제본
        #                       예: 8B bf16 policy 4 GiB/GPU + reference 16 GiB = ~20 GiB baseline
        #
        # 로그 level 규칙:
        #   DDP               : debug (운영 기본 조용, 설계상 shard 미적용이 정상)
        #   FSDP + ratio > 3  : warning (shard 미적용 의심)
        #   FSDP + ratio <= 3 : info (정상 baseline)
        # 모든 log는 rank별 정보라 `extra={"all_ranks": True}` 로 Rank0Filter 에스케이프.
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
            _strategy_name = (
                type(self.strategy).__name__ if self.strategy is not None else "NoStrategy"
            )
            _is_fsdp = _strategy_name.startswith("FSDP")
            _is_ddp = _strategy_name == "DDPStrategy"

            if _is_ddp:
                # DDPStrategy → shard 미적용이 설계상 정상. 운영 기본 조용(debug).
                logger.debug(
                    "DDP strategy — no model sharding (expected). rank=%d |"
                    " allocated=%.2f GiB | reserved=%.2f GiB |"
                    " policy trainable=%.1fM params | frozen replica=%.1f GiB",
                    _rank, _mem_alloc, _mem_resv,
                    _trainable_params / 1e6, _frozen_replica_gib,
                    extra={"all_ranks": True},
                )
            else:
                # FSDP / 기타 — shard baseline 전체 진단.
                logger.info(
                    "FSDP shard baseline: rank=%d | allocated=%.2f GiB | reserved=%.2f GiB"
                    " | policy trainable=%.1fM params | frozen replica=%.1f GiB (NO_SHARD)",
                    _rank, _mem_alloc, _mem_resv,
                    _trainable_params / 1e6, _frozen_replica_gib,
                    extra={"all_ranks": True},
                )
                if _rank == 0:
                    # use_orig_params=True에서 total_orig는 unsharded 원본 크기를 반환한다.
                    _policy_shard_gib = 16.0 / _world_size  # 8B bf16 ÷ world_size
                    _expected_total_gib = _policy_shard_gib + _frozen_replica_gib
                    _ratio = _mem_alloc / _expected_total_gib if _expected_total_gib else 0
                    if _is_fsdp and _ratio > 3:
                        logger.warning(
                            "FSDP shard baseline: expected policy shard=%.1f GiB"
                            " + frozen replica=%.1f GiB = %.1f GiB total."
                            " actual=%.2f GiB (%.1f×). >>3× → policy shard 미적용 의심.",
                            _policy_shard_gib, _frozen_replica_gib,
                            _expected_total_gib, _mem_alloc, _ratio,
                            extra={"all_ranks": True},
                        )
                    else:
                        logger.info(
                            "FSDP shard baseline: expected policy shard=%.1f GiB"
                            " + frozen replica=%.1f GiB = %.1f GiB total."
                            " actual=%.2f GiB (%.1f×).",
                            _policy_shard_gib, _frozen_replica_gib,
                            _expected_total_gib, _mem_alloc, _ratio,
                            extra={"all_ranks": True},
                        )
                    # LoRA freeze 의심은 FSDP에서만 유효 — DDP는 위에서 조기 return됨
                    if _is_fsdp and _total_params / 1e9 > 1.0 and _trainable_params / _total_params > 0.1:
                        logger.warning(
                            "FSDP shard baseline: trainable ratio %.1f%% — LoRA freeze 미적용 의심."
                            " 기대값: LoRA만 trainable (전체의 ~0.5%% 수준)",
                            100 * _trainable_params / _total_params,
                            extra={"all_ranks": True},
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

        # Inject storage.checkpoint_dir into ModelCheckpoint callbacks that don't have an
        # explicit dirpath set. Config takes precedence over recipe, so
        # --override config.storage.checkpoint_dir=X overrides recipe's default.
        _cfg_storage = getattr(self.settings.config, "storage", None)
        _ckpt_dir = _cfg_storage and getattr(_cfg_storage, "checkpoint_dir", None)
        if not _ckpt_dir:
            _rec_storage = getattr(self.settings.recipe, "storage", None)
            _ckpt_dir = _rec_storage and getattr(_rec_storage, "checkpoint_dir", None)
        if _ckpt_dir:
            for _cb in self.callbacks:
                _set = getattr(_cb, "set_dirpath", None)
                if callable(_set):
                    _set(_ckpt_dir)

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
        step_loss: float | None = None
        step_logits = None  # on_batch_end 콜백용 마지막 스텝 logits
        batch: dict = {}

        # OOM 관측 플래그 — train loop 이 torch.cuda.OutOfMemoryError 를 던지면
        # 내부 except 블록에서 True 로 세팅한 뒤 re-raise 한다. 기존 finally 블록
        # 안 stopped_reason 계산이 이 플래그를 최우선으로 확인하여 배너·summary
        # 양쪽에 "oom" 라벨을 전파한다.
        self._oom_observed = False

        # memory_history 시작 — recipe 의 monitoring.memory_history=True 에서만
        # 켜진다. rank-0 만 활성화하며, 실패 시 warning 후 False 반환하여 학습은
        # 계속 진행된다. 아래 finally 블록이 snapshot dump 를 호출한다.
        _mem_history_active = self._maybe_start_memory_history()

        with mlflow_ctx as mlflow_run:
            if self._is_main_process:
                if mlflow_run is not None and hasattr(mlflow_run, "info"):
                    mlflow_run_id = mlflow_run.info.run_id
                self._log_mlflow_params()

            # Run start banner — rank-0 only · is_json_mode 이면 자동 skip.
            # max_steps 는 local 변수를 extra 로 힌트 — banner 포맷은 self.max_steps 를 쓰지만
            # epoch-only run 에서는 self.max_steps=None 이라 "-"으로 출력된다.
            self._log_run_banner("start", extra={"run_id": mlflow_run_id})

            try:
                self._fire("on_epoch_start", epoch=self.epoch_counter)

                while self.global_step < max_steps and not self._stop_requested:
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
                                    batch=batch,
                                    logits=step_logits,
                                )

                        _epoch_train_loss = total_loss / max(num_steps, 1)
                        self._fire(
                            "on_epoch_end",
                            epoch=self.epoch_counter,
                            metrics={"loss": _epoch_train_loss},
                        )
                        # Epoch-level MLflow logging — Trainer와 대칭. `step=epoch`으로
                        # LR snapshot + epoch_train_loss를 1회 기록한다. step-level과는
                        # 축이 분리되어 있으며 MLflow UI에서 독립 시계열로 표시된다.
                        if self._is_main_process:
                            log_epoch_metrics(
                                self.optimizers,
                                self.epoch_counter,
                                extra={"epoch_train_loss": _epoch_train_loss},
                            )
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
                            # Trainer와 대칭으로, validation metric을 학습 중 MLflow에 즉시
                            # 기록한다(`trainer.py` _validate의 log_epoch_metrics 대칭).
                            # 키는 `val_*` prefix를 붙여 epoch 축으로 흘린다. 없으면
                            # 사용자가 학습 중 validation 추이를 UI에서 볼 수 없다.
                            if self._is_main_process and val_metrics:
                                log_epoch_metrics(
                                    self.optimizers,
                                    self.epoch_counter,
                                    extra={
                                        (k if k.startswith("val_") else f"val_{k}"): v
                                        for k, v in val_metrics.items()
                                    },
                                )

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
                        step_loss, step_logits, step_grad_norms = self._train_step_generation(batch, device_type)
                    else:
                        step_loss, step_logits, step_grad_norms = self._train_step_offline(batch, device_type, batch_idx)

                    batch_idx += 1
                    if step_loss is None:
                        continue
                    total_loss += step_loss
                    num_steps += 1

                    if batch_idx % self.grad_accum_steps == 0:
                        self._fire(
                            "on_batch_end", step=self.global_step,
                            epoch=self.epoch_counter,
                            global_step=self.global_step,
                            metrics={"loss": step_loss},
                            model=self.policy,
                            optimizer=self.optimizers["policy"],
                            scheduler=self.schedulers.get("policy"),
                            batch=batch,
                            logits=step_logits,
                        )

                        # step-level logging — grad_accum 경계에서만 발화하여 Trainer와
                        # 대칭(`trainer.py` L556 내부의 `log_step_metrics` 호출과 동일한
                        # 시점). 공용 헬퍼가 (a) `self.optimizers`의 param_group별
                        # `learning_rate[/{group_name_or_idx}]` metric + (b) `extra`의
                        # loss를 같은 step 인덱스로 1회 `mlflow.log_metrics` 호출에
                        # 흡수한다. CriticValueModel처럼 policy optimizer가 2-group
                        # (LoRA + head)인 경우 `learning_rate/group_0`·
                        # `learning_rate/group_1`(또는 name이 있으면 `learning_rate/lora`·
                        # `learning_rate/head`)이 매 step 기록된다. 과거 이 호출이
                        # grad_accum 경계 밖에 있어 `grad_accum_steps > 1`에서 같은
                        # `self.global_step` 값으로 여러 entry가 누적되며 MLflow UI
                        # 곡선이 왜곡되던 결함을 해소한다(원칙 4 "같은 시점·같은 API").
                        # rank 가드는 `_is_main_process`에서 일괄 처리한다.
                        if self._is_main_process:
                            # grad_norm/{name}/{total|lora_A|lora_B}: backward_and_step가
                            # 측정한 pre-clip gradient norm (LoRA 없으면 키 생략).
                            _throughput = self.global_step / max(time.time() - start_time, 1e-9)
                            extra_metrics: dict[str, float] = {
                                "loss": step_loss,
                                "throughput": _throughput,
                            }
                            extra_metrics.update(
                                {f"grad_norm/{k}": v for k, v in step_grad_norms.items()}
                            )
                            log_step_metrics(
                                self.optimizers,
                                self.global_step,
                                extra=extra_metrics,
                            )

                            # Text step-progress.
                            # stdout은 policy/total만 — LoRA 세분값은 MLflow UI에서 확인.
                            _mon_cfg = self._recipe_dict.get("monitoring", {}) if isinstance(self._recipe_dict, dict) else {}
                            _every_n = int(_mon_cfg.get("log_every_n_steps", 10) or 10)
                            if self.global_step > 0 and (
                                self.global_step % _every_n == 0
                                or self.global_step >= max_steps
                            ):
                                self._log_step_progress(
                                    loss=step_loss,
                                    grad_norm=step_grad_norms.get("policy/total"),
                                    start_time=start_time,
                                    max_steps=max_steps,
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
                        # Trainer와 대칭으로, step-based validation metric을 학습 중
                        # MLflow에 즉시 기록(`val_*` prefix, step 축).
                        if self._is_main_process and val_metrics:
                            log_step_metrics(
                                self.optimizers,
                                self.global_step,
                                extra={
                                    (k if k.startswith("val_") else f"val_{k}"): v
                                    for k, v in val_metrics.items()
                                },
                            )

            except torch.cuda.OutOfMemoryError:
                # OOM 포착 — rank 별 memory 상태를 rank-0 로그에 집계한 뒤 원래 예외를
                # 재전파하여 torchrun 이 종료 상태를 정확히 인지하게 한다. 이 except
                # 는 아래 finally 보다 먼저 실행되며, finally 블록은 여전히
                # on_train_end / cleanup / end banner / summary 를 정상 처리한다.
                self._oom_observed = True
                try:
                    self._dump_oom_summary()
                except Exception as summary_err:  # noqa: BLE001 — summary 실패가 OOM 을 가려선 안 된다
                    logger.warning("OOM summary dump failed: %s", summary_err)
                raise
            finally:
                # Nested try/finally 구조: on_train_end / FSDP gather / cleanup /
                # _log_mlflow_summary 중 어디에서 예외가 재전파되어도 **signal handler
                # 복원은 반드시 실행**되어야 한다. 또한 `_fire("on_train_end")`가
                # critical=True 콜백의 예외를 재전파해도 `_log_mlflow_summary`가
                # 독립적으로 실행되어 MLflow run이 zombie 상태로 남지 않도록,
                # on_train_end 호출을 별도 try/except로 감싼다.
                # Trainer.train()과 동일 패턴.
                try:
                    avg_loss = total_loss / max(num_steps, 1)
                    try:
                        self._fire("on_train_end", metrics={"loss": avg_loss})
                    except Exception as e:
                        logger.warning("on_train_end 콜백 실패: %s", e)

                    # stopped_reason 결정. OOM 관측이 최우선, 그 다음 signal > early_stop >
                    # max_steps > completed. OOM 관측 시 "oom" 라벨을 end banner 및 MLflow
                    # summary 에 그대로 반영해 grep 으로 장애 원인을 파악할 수 있게 한다.
                    if self._oom_observed:
                        stopped_reason = "oom"
                    elif self._stop_requested:
                        stopped_reason = (
                            "signal_term"
                            if self._stop_signal_name == "SIGTERM"
                            else "signal_int"
                        )
                    elif self._should_stop():
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
                        fsdp_policy_sd = gather_fsdp_state_dict(self.policy, self._is_main_process)

                    if self.strategy is not None:
                        try:
                            self.strategy.cleanup()
                        except Exception as e:
                            logger.warning(f"Strategy cleanup 실패: {e}")

                    # End banner — summary 로깅 바로 전에 rank-0 한 줄 요약.
                    # grep 으로 run 의 종료 상태를 파악 가능하게 한다. checkpoints_saved
                    # 는 rank-0 쪽에서 log_mlflow_summary 가 집계하므로 여기에서는 최소
                    # 정보만 보낸다(배너는 별도 집계 하지 않고 사후 단계의 값을
                    # 사용한다 — 순서상 callbacks 누적은 on_train_end 완료 시점에
                    # 이미 최종 상태다).
                    _banner_ckpts, _, _ = aggregate_checkpoint_stats(self.callbacks)
                    self._log_run_banner(
                        "end",
                        extra={
                            "stopped_reason": stopped_reason,
                            "duration": training_duration,
                            "checkpoints_saved": _banner_ckpts,
                            "final_loss": avg_loss,
                            "total_steps": self.global_step,
                        },
                    )

                    if self._is_main_process:
                        self._log_mlflow_summary(training_duration, stopped_reason, policy_state_dict=fsdp_policy_sd)
                finally:
                    # memory_history snapshot — 정상 종료·OOM·signal 모두에서 실행되어야
                    # 프로파일링 파일이 확실히 남는다. 내부적으로 active=False 일 때는
                    # no-op. _log_mlflow_summary 의 임의 예외가 snapshot 을 못 남기게
                    # 하지 않도록 별도 try/except 로 방어한다.
                    try:
                        self._maybe_dump_memory_snapshot(_mem_history_active)
                    except Exception as snap_err:  # noqa: BLE001
                        logger.warning("memory snapshot final dump failed: %s", snap_err)

                    # Signal handler 복원 — cleanup/on_train_end/_log_mlflow_summary가
                    # 예외를 던지더라도 handler는 반드시 원복되어야 한다.
                    # Trainer.train()과 동일 패턴.
                    signal.signal(signal.SIGTERM, original_sigterm)
                    signal.signal(signal.SIGINT, original_sigint)

        # Monitoring baseline (policy 모델 사용)
        monitoring = self._maybe_compute_baseline()

        metrics = {"loss": avg_loss}
        metrics.update(self.last_metrics)

        # Checkpoint 집계 — Trainer.train()과 대칭.
        # rank 0에서는 _log_mlflow_summary가 self._checkpoints_saved에 값을 채워두었고,
        # 그 외 rank나 MLflow 미사용 환경에서는 `aggregate_checkpoint_stats`로 한 번 더
        # 집계해 결과 dict를 보완한다(동일 duck typing 규칙).
        checkpoints_saved = getattr(self, "_checkpoints_saved", None)
        if checkpoints_saved is None:
            checkpoints_saved, _, _ = aggregate_checkpoint_stats(self.callbacks)

        result = {
            "metrics": metrics,
            "training_duration_seconds": training_duration,
            "total_steps": self.global_step,
            "total_epochs": self.epoch_counter,
            "stopped_reason": stopped_reason,
            "algorithm": type(self.algorithm).__name__,
            "checkpoints_saved": checkpoints_saved,
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
            chosen_out = _features_forward_model(
                m,
                {"input_ids": batch["chosen_input_ids"],
                 "attention_mask": batch.get("chosen_attention_mask")},
                role=name,
            )
            rejected_out = _features_forward_model(
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

    def _train_step_offline(
        self, batch: dict, device_type: str, batch_idx: int
    ) -> tuple[float | None, "torch.Tensor | None", dict[str, float]]:
        """DPO / WeightedNTPLoss — 데이터가 이미 완성된 경로.

        Returns:
            (loss_scalar, policy_logits, grad_norms):
            - logits는 on_batch_end 콜백에 전달된다.
              preference 배치는 logits가 chosen/rejected로 분리되어 있으므로 None.
            - grad_norms는 backward_and_step가 반환한 pre-clip gradient norm dict
              (키: ``"{optimizer_name}/total"``, ``"{optimizer_name}/lora_A"``,
              ``"{optimizer_name}/lora_B"``). step이 실행되지 않은 micro-step은 빈 dict.
        """
        is_preference = "chosen_input_ids" in batch
        needs_hidden = getattr(self.algorithm, "needs_hidden_states", False)
        needs_logits = getattr(self.algorithm, "needs_logits", True)
        with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            if is_preference:
                # Preference 경로: DPO는 chosen/rejected logits 둘 다 필요 →
                # needs_logits=False 선언이 있어도 무시하고 기존 forward 유지 (원칙 4).
                with torch.no_grad():
                    frozen_out = self._forward_preference(self.frozen, batch)
                trainable_out = self._forward_preference(self.trainable, batch)
            else:
                with torch.no_grad():
                    frozen_out = {name: _features_forward_model(m, batch, role=name) for name, m in self.frozen.items()}
                # needs_logits=False + needs_hidden_states=True인 fused-loss 알고리즘
                # (예: WeightedNTPLoss)은 trainable forward를 스킵하여 logits tensor와
                # LlamaForCausalLM backbone activation 중복을 제거한다. 빈 dict로 초기화하여
                # downstream hidden/head 주입 경로(setdefault)와 호환성을 유지한다.
                if needs_logits:
                    trainable_out = {name: _features_forward_model(m, batch, role=name) for name, m in self.trainable.items()}
                else:
                    trainable_out = {name: {} for name in self.trainable}
                if needs_hidden and "policy" in self.trainable:
                    hidden, head_weight = extract_hidden_states_and_head(
                        self.trainable["policy"], batch
                    )
                    trainable_out.setdefault("policy", {})["hidden_states"] = hidden
                    trainable_out["policy"]["output_head_weight"] = head_weight
            losses = self.algorithm.compute_loss(trainable_out, frozen_out, batch)

        # policy 로짓 추출 — pointwise 배치에서만 available
        policy_out = trainable_out.get("policy", {}) if isinstance(trainable_out, dict) else {}
        step_logits = policy_out.get("logits") if not is_preference else None

        step_schedulers = {
            n: s for n, s in self.schedulers.items()
            if self.scheduler_intervals.get(n) == "step"
        }
        result, grad_norms = backward_and_step(
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
            return None, None, {}
        if result is True:
            self.global_step += 1
        return (
            losses.get("policy", list(losses.values())[0]).item(),
            step_logits,
            grad_norms,
        )

    def _train_step_generation(
        self, batch: dict, device_type: str
    ) -> tuple[float | None, None, dict[str, float]]:
        """GRPO / PPO — policy가 텍스트를 생성하고, 그 결과로 학습.

        Returns:
            (loss_scalar, None, grad_norms):
            - logits는 생성 루프 내 중간 값 노출 시 메모리 압력이 커서 반환하지 않는다.
            - grad_norms는 mini-epoch 마지막 ``backward_and_step``의 반환값
              (pre-clip gradient norm). all_nan으로 break된 경로는 빈 dict.
        """
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
            old_out = _features_forward_model(self.policy, {"input_ids": generated_ids, "attention_mask": gen_mask}, role="policy")
            old_logits = old_out["logits"]
            old_log_probs = compute_log_probs(old_logits, generated_ids)

        # 3. Frozen forward + reward scoring
        gen_input = {"input_ids": generated_ids, "attention_mask": gen_mask}
        with torch.no_grad():
            frozen_out = {
                name: _features_forward_model(m, gen_input, role=name)
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
        needs_hidden = getattr(self.algorithm, "needs_hidden_states", False)
        needs_logits = getattr(self.algorithm, "needs_logits", True)
        last_loss = 0.0
        last_grad_norms: dict[str, float] = {}
        all_nan = True
        for _ in range(mini_epochs):
            with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                gen_forward_batch = {"input_ids": generated_ids, "attention_mask": gen_mask}
                # needs_logits=False 알고리즘은 trainable forward를 스킵 (offline 경로와 동일 패턴).
                # rollout 단계의 old_logits forward(위 L1544)는 PPO/GRPO의 KL penalty 계산에 필수이므로
                # 플래그와 무관하게 유지된다.
                if needs_logits:
                    trainable_out = {
                        name: _features_forward_model(m, gen_forward_batch, role=name)
                        for name, m in self.trainable.items()
                    }
                else:
                    trainable_out = {name: {} for name in self.trainable}
                if needs_hidden and "policy" in self.trainable:
                    hidden, head_weight = extract_hidden_states_and_head(
                        self.trainable["policy"], gen_forward_batch
                    )
                    trainable_out.setdefault("policy", {})["hidden_states"] = hidden
                    trainable_out["policy"]["output_head_weight"] = head_weight
                losses = self.algorithm.compute_loss(trainable_out, frozen_out, gen_batch)

            step_schedulers = {
                n: s for n, s in self.schedulers.items()
                if self.scheduler_intervals.get(n) == "step"
            }
            result, grad_norms = backward_and_step(
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
            last_grad_norms = grad_norms

        if all_nan:
            return None, None, {}
        self.global_step += 1
        # generation 경로는 logits를 수집하지 않는다 (생성 루프 내 중간 logits를 노출하면 과도한 메모리 압력 발생)
        return last_loss, None, last_grad_norms

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


