"""Trainer — MDP 학습 루프.

에폭/스텝 기반 학습, AMP, gradient accumulation, gradient clipping,
콜백 호출, MLflow 로깅을 담당한다.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from mdp.assembly.bundles import (
    SFTTrainingBundle,
    build_sft_training_bundle,
    create_loss,
    create_sft_optimizer,
    create_sft_scheduler,
)
from mdp.models.forward import make_forward_fn
from mdp.settings.components import ComponentSpec
from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Settings
from mdp.training._common import (
    aggregate_checkpoint_stats,
    backward_and_step,
    detect_device,
    set_epoch_on_loader,
    setup_amp,
)
from mdp.training._base import BaseTrainer
from mdp.training._checkpoint import (
    export_sft_model_artifact,
    find_best_checkpoint,
    load_checkpoint,
    ModelSlot,
)
from mdp.training._mlflow_logging import (
    log_epoch_metrics,
    log_static_params,
    log_step_metrics,
    log_summary,
)
from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """MDP 학습 루프."""

    def __init__(
        self,
        settings: Settings,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        callbacks: list[BaseCallback] | None = None,
    ) -> None:
        resolver = ComponentResolver()
        bundle = build_sft_training_bundle(
            settings=settings,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            callbacks=callbacks,
            resolver=resolver,
        )
        self._init_from_bundle(bundle, resolver=resolver)

    @classmethod
    def from_bundle(cls, bundle: SFTTrainingBundle) -> "Trainer":
        """Construct a Trainer from a materialized SFTTrainingBundle."""
        trainer = cls.__new__(cls)
        trainer._init_from_bundle(bundle)
        return trainer

    def _init_from_bundle(
        self,
        bundle: SFTTrainingBundle,
        *,
        resolver: ComponentResolver | None = None,
    ) -> None:
        self.settings = bundle.settings
        self.model = bundle.model
        self.train_loader = bundle.train_loader
        self.val_loader = bundle.val_loader
        self.resolver = resolver or ComponentResolver()

        recipe = self.settings.recipe
        training = recipe.training

        # Device
        self.device = detect_device()

        # Training config
        self.epochs = training.epochs
        self.max_steps = training.max_steps
        self.grad_accum_steps = training.gradient_accumulation_steps
        self.grad_clip_norm = training.gradient_clip_max_norm
        self.gradient_checkpointing = training.gradient_checkpointing
        self.compile_mode = training.compile
        self.val_check_interval = training.val_check_interval
        self.val_check_unit = training.val_check_unit

        # AMP setup
        self.amp_enabled, self.amp_dtype, self.scaler = setup_amp(training.precision, self.device)

        # Components
        self.optimizer = bundle.optimizer or self._create_optimizer(recipe.optimizer)
        if bundle.scheduler is None and recipe.scheduler is not None:
            self.scheduler, self.scheduler_interval = self._create_scheduler(recipe.scheduler)
        else:
            self.scheduler = bundle.scheduler
            self.scheduler_interval = bundle.scheduler_interval
        if bundle.loss_fn is None and recipe.loss is not None:
            self.loss_fn = self._create_loss(recipe.loss)
        else:
            self.loss_fn = bundle.loss_fn
        self.callbacks = list(bundle.callbacks) if bundle.callbacks else []
        self.strategy = bundle.strategy
        self.expert_parallel = bundle.expert_parallel
        self._is_main_process = int(os.environ.get("RANK", "0")) == 0

        # Recipe snapshot (체크포인트에 내장용)
        self._recipe_dict = self.settings.recipe.model_dump()

        # State
        self.global_step = 0
        self.start_epoch = 0
        self._resume_step_in_epoch = 0
        self.last_metrics: dict[str, float] = {}

        # Signal handling — SIGTERM/SIGINT로 graceful stop 요청을 받는 플래그.
        # train() 진입 시 SIGTERM/SIGINT handler가 self._stop_requested = True를 세우고,
        # 현재 step 경계에서 break하여 finally 블록(cleanup + on_train_end + summary)이
        # 정상적으로 실행되도록 한다.
        self._stop_requested: bool = False
        self._stop_signal_name: str | None = None
        self._warned_ignored_forward_loss: bool = False

    # ── Component creation ──

    def _create_optimizer(self, config: ComponentSpec) -> torch.optim.Optimizer:
        return create_sft_optimizer(self.model, config, self.resolver)

    def _create_scheduler(
        self, config: ComponentSpec | None
    ) -> tuple[Any, str] | tuple[None, str]:
        if config is None:
            return None, "step"
        return create_sft_scheduler(
            self.optimizer,
            config,
            total_steps=self._estimate_total_steps(),
            resolver=self.resolver,
        )

    def _create_loss(self, config: ComponentSpec | None) -> nn.Module | None:
        return create_loss(config, self.resolver)

    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        """공용 로깅 헬퍼(``_mlflow_logging``)용 optimizer dict 시그니처.

        RLTrainer(``self.optimizers``는 이미 dict)와의 대칭을 위해 Trainer에서도
        단일 ``self.optimizer``를 ``{"policy": ...}`` dict로 포장해 동일 함수
        (``log_step_metrics``·``log_epoch_metrics``·``collect_optimizer_state``)에
        전달한다. 키 이름은 RLTrainer의 policy optimizer와 일치하도록 ``"policy"``
        로 통일.
        """
        return {"policy": self.optimizer}

    # ── BaseTrainer abstract method 구현 ──

    def _optimizer_for_progress_log(self) -> torch.optim.Optimizer | None:
        """SFT Trainer 는 단일 optimizer 를 그대로 반환한다."""
        return self.optimizer

    def _algorithm_label(self) -> str:
        """SFT Trainer 는 recipe.task 를 algorithm 슬롯에 사용한다."""
        recipe = getattr(self.settings, "recipe", None)
        return getattr(recipe, "task", "sft") if recipe else "sft"

    def _collect_mlflow_params(self) -> None:
        """Run 시작 시 실험 재현에 필요한 하이퍼파라미터를 기록한다.

        공용 헬퍼 ``log_static_params``에 위임한다. recipe의 선언 lr은
        ``learning_rate_init`` 키로 기록된다. warmup step 0 값을 recipe 선언값으로
        오인하지 않도록 optimizer 런타임 상태는 param으로 내보내지 않는다.
        """
        log_static_params(self.settings.recipe, self.settings)

    def _checkpoint_state(self) -> dict:
        """현재 학습 상태를 dict로 직렬화한다.

        반환 dict는 ``_load_checkpoint_state``에 그대로 전달될 수 있어야 한다.
        SFT Trainer는 단일 optimizer를 사용하며, key ``""``(빈 문자열)로 루트 ckpt_dir에
        직접 저장한다 (RLTrainer의 per-model 서브디렉토리와 구별).

        Note: Trainer는 ModelCheckpoint 콜백이 저장을 담당하므로 이 메서드는
        ``_maybe_resume``에서 ``load_checkpoint`` + ``_load_checkpoint_state`` 경로를
        통해 복원용으로 사용된다.
        """
        return {
            "trainer_state": {
                "epoch": getattr(self, "start_epoch", 0),
                "global_step": self.global_step,
                "step_in_epoch": getattr(self, "_resume_step_in_epoch", 0),
            },
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }

    def _checkpoint_model_slots(self) -> list[ModelSlot]:
        """SFT checkpoint manager input for the single trainable policy model."""
        return [
            ModelSlot(
                name="",
                role="policy",
                model=self.model,
                trainable=True,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        ]

    def _load_checkpoint_state(self, state: dict) -> None:
        """``load_checkpoint``가 반환한 state dict로 학습 상태를 복원한다.

        복원 순서 (순서 민감):
        1. 모델 weights (adapter → safetensors → pt 우선순위)
        2. optimizer state_dict
        3. scheduler state_dict
        4. GradScaler state_dict
        5. trainer scalar state (global_step, start_epoch, _resume_step_in_epoch)
        6. EP scatter (expert parallel 사용 시)

        :param state: ``load_checkpoint(ckpt_dir)``가 반환한 dict.
        """
        ckpt_path: Path = state["ckpt_dir"]
        logger.info(f"체크포인트에서 재개: {ckpt_path}")

        # 1. Model weights: adapter_model.safetensors → model.safetensors → model.pt
        adapter_path = ckpt_path / "adapter_model.safetensors"
        safetensors_path = ckpt_path / "model.safetensors"
        model_pt_path = ckpt_path / "model.pt"

        target = getattr(self.model, "module", self.model)

        if adapter_path.exists():
            # LoRA / PEFT adapter
            if hasattr(target, "load_adapter"):
                from mdp.serving.model_loader import _get_adapter_name
                adapter_name = _get_adapter_name(ckpt_path)
                target.load_adapter(str(ckpt_path), adapter_name=adapter_name)
                logger.info("LoRA adapter loaded from %s (adapter_name=%s)", ckpt_path, adapter_name)
            else:
                logger.warning(
                    "adapter_model.safetensors found but model has no load_adapter method"
                )
        elif safetensors_path.exists():
            try:
                from safetensors.torch import load_file
                target.load_state_dict(load_file(safetensors_path))
            except ImportError:
                logger.warning("safetensors not installed, cannot load model.safetensors")
        elif model_pt_path.exists():
            target.load_state_dict(
                torch.load(model_pt_path, map_location="cpu", weights_only=True)
            )

        # 2. Optimizer
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location="cpu", weights_only=True)
            )

        # 3. Scheduler
        sched_path = ckpt_path / "scheduler.pt"
        if sched_path.exists() and self.scheduler is not None:
            self.scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=True)
            )

        # 4. GradScaler
        scaler_sd = state.get("scaler")
        if scaler_sd is not None and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_sd)

        # 5. Trainer scalar state
        trainer_state = state.get("trainer_state")
        if trainer_state is not None:
            saved_epoch = trainer_state.get("epoch", 0)
            self.global_step = trainer_state.get("global_step", 0)
            self._resume_step_in_epoch = trainer_state.get("step_in_epoch", 0)
            # epoch 필드는 "저장 시점의 epoch". step_in_epoch이 0이면
            # 에폭 끝 checkpoint이므로 다음 에폭부터 재개.
            if self._resume_step_in_epoch == 0:
                self.start_epoch = saved_epoch + 1
            else:
                self.start_epoch = saved_epoch

        # 6. EP scatter (checkpoint에서 전체 expert를 로드한 후, 비담당 expert를 다시 분배)
        if self.expert_parallel is not None:
            self.expert_parallel.scatter_experts(self.model, self.device)

    # ── Callback dispatch ──

    def _fire(self, hook_name: str, **extra_kwargs: Any) -> None:
        # EP gather: checkpoint를 저장할 수 있는 hook 전에 expert를 모은다.
        # 이후 strategy.save_checkpoint이 완전한 state_dict를 저장할 수 있다.
        _do_ep_gather = (
            self.expert_parallel is not None
            and hook_name in ("on_validation_end", "on_batch_end")
        )
        if _do_ep_gather:
            assert self.expert_parallel is not None  # guaranteed by _do_ep_gather
            self.expert_parallel.gather_experts(self.model)

        kwargs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "global_step": self.global_step,
            "strategy": self.strategy,
            "recipe_dict": self._recipe_dict,
            "scaler": self.scaler,
            "trainer": self,
        }
        kwargs.update(extra_kwargs)
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    if getattr(cb, "critical", False):
                        raise
                    logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")

        # EP scatter: checkpoint 저장 후 비담당 expert를 다시 CPU + frozen
        if _do_ep_gather:
            assert self.expert_parallel is not None  # guaranteed by _do_ep_gather
            self.expert_parallel.scatter_experts(self.model, self.device)

    # ── Training loop ──

    def train(self) -> dict[str, Any]:
        """학습을 실행하고 최종 메트릭을 반환한다."""
        # Signal handlers — SIGTERM/SIGINT 수신 시 graceful stop 요청.
        # 핵심: C 레벨 exit() 경로(SIGTERM 기본 동작)를 피하고, Python handler에서
        # flag만 세워 현재 step 경계에서 break → finally로 빠지게 한다.
        # 이로써 cleanup / on_train_end / _log_mlflow_summary가 실행되어
        # MLflow run이 zombie 상태로 남는 것을 방지한다.
        # torchrun 분산 환경: 각 rank 프로세스의 main thread가 독립적으로 handler를
        # 설치하며, torchrun이 SIGTERM을 모든 rank에 전파하면 동일 step에서
        # break하므로 NCCL collective 데드락도 피한다.
        self._stop_requested = False
        self._stop_signal_name = None
        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def _signal_handler(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            # 첫 시그널만 기록한다. SIGTERM 수신 후 사용자가 Ctrl+C를 누르거나
            # 외부가 이중 신호를 보내도 `_stop_signal_name`이 덮어쓰이지 않게 하여
            # stopped_reason tag가 실제 종료 원인을 정확히 반영하도록 보장한다.
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

        # Expert Parallelism (전략 setup 전에 적용 — hook 설치 + expert 분배)
        if self.expert_parallel is not None:
            if self.strategy is not None and not torch.distributed.is_initialized():
                # EP는 process group이 필요하므로, 전략이 초기화하게 한다.
                # 대부분의 전략은 setup() 첫 줄에서 init_process_group()을 호출한다.
                # EP를 먼저 적용하려면 process group이 먼저 있어야 하므로,
                # 직접 초기화한다.
                import torch.distributed as _dist
                backend = getattr(self.strategy, "backend", "nccl")
                _dist.init_process_group(backend=backend)
            self.model = self.expert_parallel.setup(self.model, self.device)

        # Guard: device_map 모델은 학습 불가
        if hasattr(self.model, "hf_device_map"):
            raise RuntimeError(
                "device_map으로 분산 배치된 모델은 학습에 사용할 수 없습니다. "
                "device_map은 추론/서빙 전용이며, 학습에는 DDP/FSDP 전략을 사용하세요."
            )

        # Gradient Checkpointing (DDP/FSDP wrap 전에 적용해야 한다)
        # FSDP 이후에 활성화하면 use_reentrant=True(기본값)가 FSDP의 param 조기 해제를 막아 OOM.
        # LoRA: 입력 텐서에 requires_grad가 없으면 GC recompute 구간에서 grad 소실 → silent failure.
        if self.gradient_checkpointing:
            base = getattr(self.model, "module", self.model)   # DDP/FSDP 대비 (현재 no-op)
            base = getattr(base, "base_model", base)           # PeftModel → LoraModel
            base = getattr(base, "model", base)                # LoraModel → PreTrainedModel
            if hasattr(base, "gradient_checkpointing_enable"):
                if hasattr(base, "enable_input_require_grads"):
                    base.enable_input_require_grads()
                base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("Gradient checkpointing enabled (use_reentrant=False)")
            else:
                logger.warning(
                    "gradient_checkpointing_enable not found on %s — skipping",
                    type(base).__name__,
                )

        # Strategy setup (DDP/FSDP/DeepSpeed wrapping)
        if self.strategy is not None:
            self.model = self.strategy.setup(self.model, self.device, optimizer=self.optimizer)
        else:
            self.model = self.model.to(self.device)

        # torch.compile — must be AFTER distributed wrapping
        if self.compile_mode:
            mode = self.compile_mode if isinstance(self.compile_mode, str) else "default"
            self.model = torch.compile(self.model, mode=mode)

        # Resume
        self._maybe_resume()

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
        # `_train_one_epoch` 의 step-progress 로그가 참조한다.
        # train loop 전체에서 단조 증가하는 단일 기준 시각이다.
        self._progress_start_time = start_time
        self._progress_max_steps = total_steps

        stopped_reason = "completed"
        last_epoch = self.start_epoch
        baseline_info = None
        mlflow_ctx = self._start_mlflow_run() if self._is_main_process else nullcontext()
        mlflow_run_id: str | None = None

        if self.epochs is not None:
            max_epochs = int(self.epochs) + (1 if self.epochs != int(self.epochs) else 0)
            self._total_batch_budget = int(len(self.train_loader) * self.epochs)
        else:
            max_epochs = sys.maxsize
            self._total_batch_budget = None
        # Resume 시 이미 처리된 배치를 반영
        self._batches_consumed = (
            self.start_epoch * len(self.train_loader) + self._resume_step_in_epoch
        )

        # OOM 관측 플래그. 학습 loop 안에서 torch.cuda.OutOfMemoryError 가 raise 되면
        # except 블록이 True 로 세팅하고 원래 예외를 재전파한다. finally 블록의
        # stopped_reason 계산이 이를 최우선 확인하여 end banner 및 MLflow summary 에
        # "oom" 라벨을 전파한다.
        self._oom_observed = False

        # memory_history 시작 — recipe 의 monitoring.memory_history=True 에서만
        # 켜진다. rank-0 만 활성화하며, 아래 innermost finally 가 snapshot dump 를
        # 호출한다. RLTrainer.train() 과 동일 패턴.
        _mem_history_active = self._maybe_start_memory_history()

        with mlflow_ctx as mlflow_run:
            if self._is_main_process:
                if mlflow_run is not None and hasattr(mlflow_run, "info"):
                    mlflow_run_id = mlflow_run.info.run_id
                self._log_mlflow_params()

            # Run start banner — rank-0 only · is_json_mode 이면 자동 skip.
            self._log_run_banner("start", extra={"run_id": mlflow_run_id})

            try:
                for epoch in range(self.start_epoch, max_epochs):
                    if self._stop_requested:
                        stopped_reason = (
                            "signal_term"
                            if self._stop_signal_name == "SIGTERM"
                            else "signal_int"
                        )
                        break
                    if self._should_stop():
                        stopped_reason = "early_stopped"
                        break
                    if self.max_steps and self.global_step >= self.max_steps:
                        stopped_reason = "max_steps_reached"
                        break

                    # 분산 학습: 매 에폭 셔플 순서 갱신
                    set_epoch_on_loader(self.train_loader, epoch)

                    self._fire("on_epoch_start", epoch=epoch)
                    train_loss = self._train_one_epoch(epoch)
                    self._fire(
                        "on_epoch_end", epoch=epoch, metrics={"train_loss": train_loss}
                    )

                    # Epoch-level metrics — ``log_epoch_metrics``가 optimizer
                    # param_groups 전체를 순회해 ``learning_rate``(single-group) 또는
                    # ``learning_rate/group_*`` (multi-group)을 epoch 축에 기록한다.
                    # 추가로 ``epoch_train_loss``를 ``extra``로 같이 내보낸다.
                    log_epoch_metrics(
                        self._optimizer_dict(),
                        epoch,
                        extra={"epoch_train_loss": train_loss},
                    )

                    # Epoch-end validation (step/fractional은 _train_one_epoch 내부에서 처리)
                    if (
                        self.val_loader is not None
                        and self.val_check_unit == "epoch"
                        and self.val_check_interval >= 1.0
                        and (epoch + 1) % int(self.val_check_interval) == 0
                    ):
                        self._run_validation(epoch)

                    # Epoch-level scheduler
                    if self.scheduler is not None and self.scheduler_interval == "epoch":
                        self.scheduler.step()

                    last_epoch = epoch

                baseline_info = self._maybe_compute_baseline()

            except torch.cuda.OutOfMemoryError:
                # OOM 포착 — rank 별 memory 상태를 rank-0 로그에 집계한 뒤 원래 예외를
                # 재전파하여 torchrun 이 종료 상태를 정확히 인지하게 한다. finally 는
                # 여전히 cleanup / on_train_end / end banner / summary 를 정상 처리한다.
                self._oom_observed = True
                try:
                    self._dump_oom_summary()
                except Exception as summary_err:  # noqa: BLE001 — summary 실패가 OOM 을 가려선 안 된다
                    logger.warning("OOM summary dump failed: %s", summary_err)
                raise
            finally:
                # Nested try/finally 구조: cleanup → on_train_end → summary 중 어디에서
                # 예외가 재전파되어도 **signal handler 복원은 반드시 실행**되어야 한다.
                # 또한 `_fire("on_train_end")`가 critical=True 콜백(ModelCheckpoint 등)의
                # 예외를 재전파하더라도 `_log_mlflow_summary`가 독립적으로 실행되어
                # MLflow run이 zombie 상태(stopped_reason tag 누락)로 남지 않도록,
                # on_train_end 호출을 별도 try/except로 감싼다.
                if self._oom_observed:
                    # stopped_reason 은 loop 내부에서 미리 세팅될 수 있으므로 OOM 을
                    # 최우선으로 덮어쓴다. end banner / MLflow summary 양쪽에 "oom"
                    # 라벨이 들어가 grep 으로 장애 원인을 파악할 수 있게 한다.
                    stopped_reason = "oom"
                try:
                    # Strategy cleanup
                    if self.strategy is not None:
                        try:
                            self.strategy.cleanup()
                        except Exception as e:
                            logger.warning(f"Strategy cleanup 실패: {e}")

                    # on_train_end를 MLflow 컨텍스트 안에서 먼저 발화한다.
                    # EMACallback이 여기서 가중치를 복원하므로,
                    # 이후 _log_mlflow_summary의 모델 export가 EMA 가중치를 포함한다.
                    try:
                        self._fire("on_train_end", metrics=self.last_metrics)
                    except Exception as e:
                        logger.warning("on_train_end 콜백 실패: %s", e)

                    training_duration = time.time() - start_time

                    # End banner — rank-0 한 줄 요약을 summary 로깅 직전에 출력.
                    # grep 으로 run 종료 상태를 파악 가능하게 한다. checkpoints_saved 는
                    # on_train_end 이후 최종 상태가 되므로 이 시점에서 집계한다.
                    _banner_ckpts, _, _ = aggregate_checkpoint_stats(self.callbacks)
                    _banner_loss = self.last_metrics.get("train_loss") or self.last_metrics.get("loss")
                    self._log_run_banner(
                        "end",
                        extra={
                            "stopped_reason": stopped_reason,
                            "duration": training_duration,
                            "checkpoints_saved": _banner_ckpts,
                            "final_loss": _banner_loss,
                            "total_steps": self.global_step,
                        },
                    )

                    if self._is_main_process:
                        self._log_mlflow_summary(training_duration, stopped_reason)
                finally:
                    # memory_history snapshot — 정상 종료·OOM·signal 모두에서 실행되어야
                    # 프로파일링 파일이 확실히 남는다. 내부적으로 active=False 일 때는
                    # no-op. summary 실패가 snapshot 저장을 막지 않도록 방어한다.
                    try:
                        self._maybe_dump_memory_snapshot(_mem_history_active)
                    except Exception as snap_err:  # noqa: BLE001
                        logger.warning("memory snapshot final dump failed: %s", snap_err)

                    # Signal handler 복원 — cleanup/on_train_end/_log_mlflow_summary가
                    # 예외를 던지더라도 handler는 반드시 원복되어야 한다. 라이브러리
                    # 사용자(Python API로 Trainer 직접 호출) 환경에서 원래 핸들러를
                    # 그대로 돌려주어 handler 오염 누적을 방지한다.
                    signal.signal(signal.SIGTERM, original_sigterm)
                    signal.signal(signal.SIGINT, original_sigint)

        # Checkpoint 집계는 rank 0의 _log_mlflow_summary가 계산해 self._checkpoints_saved에
        # 저장한다. 그 외 rank나 MLflow 경로 미사용 환경에서는 이 속성이 없으므로
        # 여기서 `aggregate_checkpoint_stats`로 한 번 더 집계해 결과 dict를 보완한다
        # (동일 duck typing 규칙).
        checkpoints_saved = getattr(self, "_checkpoints_saved", None)
        if checkpoints_saved is None:
            checkpoints_saved, _, _ = aggregate_checkpoint_stats(self.callbacks)

        result: dict[str, Any] = {
            "metrics": self.last_metrics,
            "training_duration_seconds": training_duration,
            "total_epochs": last_epoch - self.start_epoch + 1,
            "total_steps": self.global_step,
            "stopped_reason": stopped_reason,
            "checkpoints_saved": checkpoints_saved,
        }
        if baseline_info is not None:
            result["monitoring"] = baseline_info
        if mlflow_run_id is not None:
            result["run_id"] = mlflow_run_id

        return result

    def _run_validation(self, epoch: int) -> None:
        """검증을 실행하고 콜백을 발화한다."""
        self._fire("on_validation_start", epoch=epoch)
        val_metrics = self._validate(epoch)
        self._fire("on_validation_end", epoch=epoch, metrics=val_metrics)
        self.last_metrics.update(val_metrics)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        device_type = self.device.type if self.device.type != "mps" else "cpu"

        # Mid-epoch validation 설정
        steps_in_epoch = len(self.train_loader)
        interval = self.val_check_interval
        unit = self.val_check_unit

        if unit == "step":
            val_every_n = max(1, int(interval))
        elif interval < 1.0:
            # epoch 단위 소수: 에폭의 비율마다 검증
            val_every_n = max(1, int(steps_in_epoch * interval))
        else:
            val_every_n = 0  # 정수 에폭 단위 → train() 메서드에서 처리

        # Skip already-processed batches when resuming from a step-level checkpoint
        start_step = 0
        if self._resume_step_in_epoch > 0:
            start_step = self._resume_step_in_epoch
            self._resume_step_in_epoch = 0  # Only apply once (first epoch after resume)
            logger.info("Skipping %d already-processed batches for epoch resume", start_step)

        actual_loss = 0.0  # default for residual-flush path when loop body skips
        for step, batch in enumerate(self.train_loader):
            if step < start_step:
                continue
            if self._stop_requested:
                break
            if self.max_steps and self.global_step >= self.max_steps:
                break
            if self._total_batch_budget is not None and self._batches_consumed >= self._total_batch_budget:
                break

            batch = self._move_to_device(batch)
            self._fire("on_batch_start", step=self.global_step)

            with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                loss = self._compute_loss(batch)

            step_schedulers = (
                {"model": self.scheduler}
                if self.scheduler is not None and self.scheduler_interval == "step"
                else {}
            )
            result, grad_norms = backward_and_step(
                losses={"model": loss},
                optimizers={"model": self.optimizer},
                schedulers=step_schedulers,
                scaler=self.scaler,
                trainable_models={"model": self.model},
                grad_accum_steps=self.grad_accum_steps,
                at_accum_boundary=(step + 1) % self.grad_accum_steps == 0,
                grad_clip_norm=self.grad_clip_norm,
            )
            if result is None:
                continue
            if result:
                self.global_step += 1

            actual_loss = loss.item()
            total_loss += actual_loss
            num_batches += 1
            self._batches_consumed += 1

            if (step + 1) % self.grad_accum_steps == 0:
                self._fire(
                    "on_batch_end", step=self.global_step, epoch=epoch,
                    global_step=self.global_step,
                    step_in_epoch=step + 1,
                    metrics={"loss": actual_loss},
                )

                # MLflow step logging (non-blocking).
                # ``log_step_metrics``는 optimizer param_groups의 scheduler-adjusted
                # LR을 slash 네이밍으로 흘리며, ``extra``의 train_loss를 같은 step 축에
                # 병합한다(단일 MLflow round-trip).
                # grad_norm/{name}/{total|lora_A|lora_B}는 backward_and_step에서
                # pre-clip 측정된 pre-optimizer gradient norm을 그대로 MLflow 축에 태운다.
                if self._is_main_process:
                    _start = getattr(self, "_progress_start_time", None)
                    _throughput = (
                        self.global_step / max(time.time() - _start, 1e-9)
                        if _start is not None else None
                    )
                    extra_metrics: dict[str, float] = {"train_loss": actual_loss}
                    extra_metrics.update(
                        {f"grad_norm/{k}": v for k, v in grad_norms.items()}
                    )
                    if _throughput is not None:
                        extra_metrics["throughput"] = _throughput
                    log_step_metrics(
                        self._optimizer_dict(),
                        self.global_step,
                        extra=extra_metrics,
                    )

                    # Text step-progress — file-redirect 환경에서도 step 진행이
                    # stdout 에 남도록 한다.
                    # `log_every_n_steps` 간격마다 + 마지막 step 에서 1회 출력.
                    _mon_cfg = self._recipe_dict.get("monitoring", {}) if isinstance(self._recipe_dict, dict) else {}
                    _every_n = int(_mon_cfg.get("log_every_n_steps", 10) or 10)
                    _max_steps = getattr(self, "_progress_max_steps", 0) or self._estimate_total_steps()
                    if self.global_step > 0 and (
                        self.global_step % _every_n == 0
                        or (_max_steps and self.global_step >= _max_steps)
                    ):
                        self._log_step_progress(
                            loss=actual_loss,
                            grad_norm=grad_norms.get("model/total"),
                            start_time=getattr(self, "_progress_start_time", time.time()),
                            max_steps=_max_steps or max(self.global_step, 1),
                        )

            # Mid-epoch validation (step 단위 또는 소수 에폭)
            if (
                val_every_n > 0
                and self.val_loader is not None
                and (step + 1) % val_every_n == 0
                and (step + 1) < steps_in_epoch  # 에폭 마지막 step은 아래에서 처리
            ):
                self._run_validation(epoch)

        # 에폭 마지막: 잔여 gradient flush + on_batch_end 발화
        # 루프 내에서 마지막 배치의 backward는 이미 수행됨 (at_accum_boundary=False).
        # gradient가 누적된 상태이므로 optimizer step만 실행한다.
        if num_batches > 0 and num_batches % self.grad_accum_steps != 0:
            step_schedulers = (
                {"model": self.scheduler}
                if self.scheduler is not None and self.scheduler_interval == "step"
                else {}
            )
            self.scaler.unscale_(self.optimizer)
            # NaN/Inf guard: unscale 후 gradient에 inf/NaN이 있으면 step을 건너뛴다
            has_inf = any(
                torch.isinf(p.grad).any() or torch.isnan(p.grad).any()
                for p in self.model.parameters()
                if p.grad is not None
            )
            if has_inf:
                logger.warning("NaN/Inf gradient in residual flush, skipping step")
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
            else:
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                sched = step_schedulers.get("model")
                if sched is not None:
                    sched.step()
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self._fire(
                    "on_batch_end", step=self.global_step, epoch=epoch,
                    global_step=self.global_step,
                    step_in_epoch=num_batches,
                    metrics={"loss": actual_loss},
                )

                # Residual flush 경계에서도 step-level LR·loss 로깅을 일관되게 보낸다
                # (루프 내부 grad_accum 경계와 동일한 경로).
                if self._is_main_process:
                    log_step_metrics(
                        self._optimizer_dict(),
                        self.global_step,
                        extra={"train_loss": actual_loss},
                    )

        # mid-epoch 모드: 에폭 끝에서도 1회 검증 (마지막 구간 커버)
        if val_every_n > 0 and self.val_loader is not None:
            self._run_validation(epoch)

        return total_loss / max(num_batches, 1)

    def _compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        outputs = self._forward_batch(batch)
        if self.loss_fn is not None:
            if self._extract_output_loss(outputs) is not None:
                self._warn_ignored_forward_loss_once()
            logits = self._extract_logits(outputs)
            labels = self._extract_labels(batch)
            if logits is None:
                raise ValueError("model.forward()가 'logits' 또는 'output' 키를 반환하지 않았습니다")
            if labels is None:
                raise ValueError("배치에 'labels' 키가 없습니다")
            return self.loss_fn(logits, labels)
        if (loss := self._extract_output_loss(outputs)) is not None:
            return loss
        raise ValueError(
            "No train loss found. Provide recipe.loss, or return `loss` from "
            "model forward output."
        )

    def _invoke_model_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Strategy가 있으면 분산 의미를 보존해 호출, 없으면 직접 호출한다."""
        if self.strategy is not None:
            return self.strategy.invoke_custom(self.model, method_name, *args, **kwargs)
        return getattr(self.model, method_name)(*args, **kwargs)

    def _forward_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run model forward through the shared MDP/HF adapter."""
        return make_forward_fn(self.model)(batch)

    def _extract_output_loss(self, outputs: dict[str, Any]) -> torch.Tensor | None:
        """Return a backward-capable forward loss candidate, if present."""
        loss = outputs.get("loss")
        return loss if isinstance(loss, torch.Tensor) else None

    def _extract_logits(self, outputs: dict[str, Any]) -> torch.Tensor | None:
        """Return logits for external supervised criteria."""
        logits = outputs.get("logits", outputs.get("output"))
        return logits if isinstance(logits, torch.Tensor) else None

    def _extract_labels(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Return labels for external supervised criteria."""
        labels = batch.get("labels", batch.get("label"))
        return labels if isinstance(labels, torch.Tensor) else None

    def _warn_ignored_forward_loss_once(self) -> None:
        if self._warned_ignored_forward_loss:
            return
        logger.warning("recipe.loss is configured; model forward output loss will be ignored.")
        self._warned_ignored_forward_loss = True

    def _unwrapped_model(self) -> nn.Module:
        """hasattr/getattr 같은 read-only 접근용. 분산 래퍼를 벗긴 model 반환."""
        if self.strategy is not None:
            return self.strategy.unwrap(self.model)
        return self.model

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        assert self.val_loader is not None  # callers always guard with val_loader is not None
        self.model.eval()
        all_metrics: dict[str, list[float]] = {}
        # DDP/FSDP 래핑 상태에선 ``hasattr(self.model, "validation_step")``가 False를
        # 반환해 silent fallback으로 빠질 수 있다. unwrap된 실제 model에서 검사한다.
        use_fallback = not hasattr(self._unwrapped_model(), "validation_step")

        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            if use_fallback:
                metrics = self._validate_fallback(batch)
            else:
                try:
                    metrics = self._invoke_model_method("validation_step", batch)
                except NotImplementedError:
                    use_fallback = True
                    metrics = self._validate_fallback(batch)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        # MLflow epoch logging — 공용 헬퍼 ``log_epoch_metrics``의 ``extra`` 인자로
        # ``val_*`` prefix를 흘려 보내 RLTrainer 쪽과 동일한 경로로 수렴한다
        # (`rl_trainer.py`의 validation 블록 참조). LR 축이 같은 epoch 인덱스로
        # 함께 기록되지만 값이 동일하므로 의미 왜곡 없음. rank 가드는 caller
        # ``_run_validation``이 ``_validate`` 호출 전 보증하지 않으므로 여기서 명시.
        if self._is_main_process and avg_metrics:
            log_epoch_metrics(
                self._optimizer_dict(),
                epoch,
                extra={
                    (k if k.startswith("val_") else f"val_{k}"): v
                    for k, v in avg_metrics.items()
                },
            )

        self.model.train()
        return avg_metrics

    def _validate_fallback(self, batch: dict[str, Any]) -> dict[str, float]:
        """Fallback validation when model lacks validation_step."""
        device_type = self.device.type if self.device.type != "mps" else "cpu"
        with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            outputs = self._forward_batch(batch)

        # Priority matches _compute_loss: loss_fn first, then outputs["loss"]/outputs.loss
        if self.loss_fn is not None:
            logits = self._extract_logits(outputs)
            labels = self._extract_labels(batch)
            if logits is not None and labels is not None:
                loss = self.loss_fn(logits, labels)
            else:
                logger.warning("_validate_fallback: logits 또는 labels를 찾을 수 없습니다")
                return {}
        elif (_out_loss := self._extract_output_loss(outputs)) is not None:
            loss = _out_loss
        else:
            logger.warning("_validate_fallback: loss를 계산할 수 없습니다")
            return {}

        return {"loss": loss.item() if isinstance(loss, torch.Tensor) else float(loss)}

    # ── Resume ──

    def _maybe_resume(self) -> None:
        resume = self.settings.config.job.resume
        if resume == "disabled":
            return

        checkpoint_dir = Path(self.settings.config.storage.checkpoint_dir)

        if resume == "auto":
            latest = checkpoint_dir / "latest"
            if not latest.exists():
                return
            ckpt_path = latest.resolve()
        else:
            ckpt_path = Path(resume)

        if not ckpt_path.exists():
            logger.warning(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
            return

        state = load_checkpoint(
            ckpt_path,
            self._checkpoint_model_slots(),
            strategy=self.strategy,
            scaler=self.scaler,
        )
        self._load_checkpoint_state(state)

    # ── Monitoring baseline ──

    def _maybe_compute_baseline(self) -> dict[str, Any] | None:
        """Compute monitoring baseline after training. All ranks execute forward for FSDP all-gather."""
        try:
            from mdp.monitoring.baseline import compute_baseline
        except ImportError:
            return None

        monitoring_cfg = getattr(self.settings.recipe, "monitoring", None)
        if monitoring_cfg is None or not getattr(monitoring_cfg, "enabled", False):
            return None

        try:
            # ALL ranks must execute forward pass (FSDP all-gather requirement)
            baseline = compute_baseline(
                train_dataloader=self.val_loader or self.train_loader,
                model=self.model,
                config=self.settings,
            )

            # Only rank-0 saves the baseline
            if self._is_main_process:
                checkpoint_dir = Path(self.settings.config.storage.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                baseline_path = checkpoint_dir / "baseline.json"
                baseline_path.write_text(json.dumps(baseline, indent=2))
                logger.info("Monitoring baseline saved: %s", baseline_path)
                return {"baseline_saved": True, "baseline_path": str(baseline_path)}

            return None
        except Exception as e:
            logger.warning(f"Monitoring baseline 계산 실패: {e}")
            return None

    # ── MLflow ──

    def _log_mlflow_summary(
        self, training_duration: float, stopped_reason: str,
    ) -> None:
        """Run 종료 시 최종 메트릭과 config snapshot을 기록한다.

        MLflow 쓰기 경로는 공용 헬퍼 ``log_summary``에 위임된다. 본 래퍼는
        다음 이유로 여전히 필요하다:

        1. ``self._checkpoints_saved`` 속성을 세팅해 ``train()`` 반환 dict가 조회할
           수 있게 한다 (test_checkpoint_monitor 계약).
        2. 0 체크포인트 시 경고 로그를 MLflow 쓰기 성공/실패와 무관하게 항상 발화.
        3. ``_export_and_log_model``은 tempdir·tokenizer·adapter 처리가 엮인 특수 경로라
           ``log_summary``의 일반 ``artifact_dirs`` 시그니처만으로는 수용되지 않아
           별도 호출을 유지한다.
        """
        # Checkpoint 집계 — `_common.aggregate_checkpoint_stats`가 duck typing 규칙을
        # 단일 구현으로 담는다(Trainer/RLTrainer 대칭).
        total_checkpoints, best_path, monitor_hint = aggregate_checkpoint_stats(
            self.callbacks,
        )
        self._checkpoints_saved = total_checkpoints

        # sanitize_config·find_best_checkpoint 둘 다 self.settings.config에 의존한다.
        # ``test_log_mlflow_summary_aggregates_multi_checkpoint_callbacks`` 같은 테스트는
        # self.settings=None으로 Trainer를 조립해 집계 경로만 고립 검증하므로, 인자
        # 준비 단계 전체를 try로 감싸서 AttributeError를 통째로 흡수한다.
        sanitized_config: dict[str, Any] | None = None
        best_ckpt: Path | None = None
        artifact_dirs: list[tuple[Path, str]] = []
        try:
            from mdp.utils.sanitize import sanitize_config

            sanitized_config = sanitize_config(self.settings.model_dump())

            ckpt_dir = Path(self.settings.config.storage.checkpoint_dir)
            best_ckpt = find_best_checkpoint(ckpt_dir)
            if best_ckpt:
                artifact_dirs.append((best_ckpt, "checkpoint"))
        except Exception as e:
            logger.warning(f"MLflow summary 인자 준비 실패 (학습 결과는 유효합니다): {e}")

        # Peak memory metric — RLTrainer와 대칭. rank 0의
        # `torch.cuda.max_memory_allocated()`를 GiB 단위로 summary에 기록한다.
        extra_summary = self._peak_memory_summary_extra()

        log_summary(
            training_duration_seconds=training_duration,
            total_steps=self.global_step,
            stopped_reason=stopped_reason,
            final_metrics=self.last_metrics,
            checkpoint_stats=(total_checkpoints, best_path, monitor_hint),
            sanitized_config=sanitized_config,
            artifact_dirs=artifact_dirs,
            extra=extra_summary,
        )

        # 서빙 가능 모델 생성 + artifact 등록 — 공용 헬퍼와 별도 경로.
        # (tempdir + tokenizer + recipe.yaml 복사 + adapter/전체 분기 로직이 있어
        # ``artifact_dirs``로 일반화하기 부적합하다.)
        if best_ckpt:
            try:
                export_sft_model_artifact(self.model, self.settings, best_ckpt)
            except Exception as e:
                logger.warning(f"모델 export 실패 (학습 결과는 유효합니다): {e}")

        # zero-checkpoint warning — MLflow 기록 성공/실패와 무관하게 항상 발화한다.
        # 학습은 돌았는데 산출물이 0개인 상황(monitor 오타 등)을 사용자가 놓치지
        # 않도록 로그 경로에서 한 번 더 알린다.
        if total_checkpoints == 0:
            logger.warning(
                "체크포인트가 하나도 저장되지 않았습니다. monitor=[%s] 설정을 확인하세요.",
                monitor_hint,
            )
