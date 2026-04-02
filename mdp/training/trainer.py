"""Trainer — MDP 학습 루프.

에폭/스텝 기반 학습, AMP, gradient accumulation, gradient clipping,
콜백 호출, MLflow 로깅을 담당한다.
"""

from __future__ import annotations

import json
import logging
import os
import sys
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
from mdp.training.callbacks.base import BaseCallback

logger = logging.getLogger(__name__)

STRATEGY_MAP: dict[str, str] = {
    "ddp": "mdp.training.strategies.ddp.DDPStrategy",
    "fsdp": "mdp.training.strategies.fsdp.FSDPStrategy",
    "deepspeed": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
    "deepspeed_zero3": "mdp.training.strategies.deepspeed.DeepSpeedStrategy",
}


class Trainer:
    """MDP 학습 루프."""

    def __init__(
        self,
        settings: Settings,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.resolver = ComponentResolver()

        recipe = settings.recipe
        training = recipe.training

        # Device
        self.device = self._detect_device()

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
        scaler_device = self.device.type
        precision = training.precision
        if precision == "fp16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
            if self.device.type == "mps":
                logger.warning("MPS에서 fp16은 GradScaler를 지원하지 않습니다. bf16을 권장합니다.")
                self.scaler = GradScaler(scaler_device, enabled=False)
            else:
                self.scaler = GradScaler(scaler_device, enabled=True)
        elif precision == "bf16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            self.scaler = GradScaler(scaler_device, enabled=False)
        else:  # fp32
            self.amp_enabled = False
            self.amp_dtype = torch.float32
            self.scaler = GradScaler(scaler_device, enabled=False)

        # Components
        self.optimizer = self._create_optimizer(recipe.optimizer)
        self.scheduler, self.scheduler_interval = self._create_scheduler(
            recipe.scheduler
        )
        self.loss_fn = self._create_loss(recipe.loss)
        self.callbacks = self._create_callbacks(recipe.callbacks)
        self.strategy = self._create_strategy(settings)
        self._is_main_process = int(os.environ.get("RANK", "0")) == 0

        # Recipe snapshot (체크포인트에 내장용)
        self._recipe_dict = settings.recipe.model_dump()

        # State
        self.global_step = 0
        self.start_epoch = 0
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _detect_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ── Component creation ──

    def _create_optimizer(self, config: dict[str, Any]) -> torch.optim.Optimizer:
        # Model-defined optimizer takes priority
        custom = self.model.configure_optimizers() if hasattr(self.model, "configure_optimizers") else None
        if custom and isinstance(custom, dict) and "optimizer" in custom:
            return custom["optimizer"]

        klass, kwargs = self.resolver.resolve_partial(config)
        return klass(self.model.parameters(), **kwargs)

    def _create_scheduler(
        self, config: dict[str, Any] | None
    ) -> tuple[Any, str] | tuple[None, str]:
        if config is None:
            return None, "step"

        config = dict(config)  # don't mutate original
        interval = config.pop("interval", "step")
        warmup_steps = config.pop("warmup_steps", 0)
        warmup_ratio = config.pop("warmup_ratio", 0.0)

        if warmup_steps > 0 and warmup_ratio > 0:
            raise ValueError(
                "warmup_steps와 warmup_ratio를 동시에 지정할 수 없습니다. "
                f"warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio}"
            )

        if warmup_steps == 0 and warmup_ratio > 0:
            total_steps = self._estimate_total_steps()
            warmup_steps = int(total_steps * warmup_ratio)

        klass, kwargs = self.resolver.resolve_partial(config)
        base_scheduler = klass(self.optimizer, **kwargs)

        if warmup_steps > 0:
            base_scheduler = self._wrap_with_warmup(
                base_scheduler, warmup_steps
            )

        return base_scheduler, interval

    def _wrap_with_warmup(self, scheduler: Any, warmup_steps: int) -> Any:
        """LinearLR warmup → base scheduler via SequentialLR."""
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_steps],
        )

    def _create_loss(self, config: dict[str, Any] | None) -> nn.Module | None:
        if config is None:
            return None
        return self.resolver.resolve(config)

    def _create_callbacks(
        self, configs: list[dict[str, Any]]
    ) -> list[BaseCallback]:
        callbacks = []
        for cfg in configs:
            try:
                cb = self.resolver.resolve(cfg)
                callbacks.append(cb)
            except Exception as e:
                logger.warning(f"콜백 생성 실패: {e}")
        return callbacks

    def _create_strategy(self, settings: Settings) -> Any:
        dist_config = settings.config.compute.distributed
        if dist_config is None:
            return None

        strategy_name = dist_config.get("strategy", "auto") if isinstance(dist_config, dict) else "auto"

        if strategy_name == "none":
            return None
        if strategy_name == "auto":
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
                return None
            strategy_name = "ddp"

        class_path = STRATEGY_MAP.get(strategy_name)
        if class_path is None:
            raise ValueError(f"알 수 없는 분산 전략: {strategy_name}")

        strategy_kwargs = {
            k: v for k, v in (dist_config if isinstance(dist_config, dict) else {}).items()
            if k != "strategy"
        }
        return self.resolver.resolve(
            {"_component_": class_path, **strategy_kwargs}
        )

    def _estimate_total_steps(self) -> int:
        if self.max_steps:
            return self.max_steps
        steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
        return steps_per_epoch * (self.epochs or 1)

    # ── Callback dispatch ──

    def _fire(self, hook_name: str, **extra_kwargs: Any) -> None:
        kwargs = {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "global_step": self.global_step,
            "strategy": self.strategy,
            "recipe_dict": self._recipe_dict,
            "scaler": self.scaler,
        }
        kwargs.update(extra_kwargs)
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")

    def _should_stop(self) -> bool:
        return any(
            getattr(cb, "should_stop", False) for cb in self.callbacks
        )

    # ── Training loop ──

    def train(self) -> dict[str, Any]:
        """학습을 실행하고 최종 메트릭을 반환한다."""
        # Strategy setup (DDP/FSDP/DeepSpeed wrapping)
        if self.strategy is not None:
            self.model = self.strategy.setup(self.model, self.device, optimizer=self.optimizer)
        else:
            self.model = self.model.to(self.device)

        # Gradient checkpointing
        if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # torch.compile — must be AFTER distributed wrapping
        if self.compile_mode:
            mode = self.compile_mode if isinstance(self.compile_mode, str) else "default"
            self.model = torch.compile(self.model, mode=mode)

        # Resume
        self._maybe_resume()

        total_steps = self._estimate_total_steps()
        self._fire("on_train_start", total_steps=total_steps)
        start_time = time.time()

        stopped_reason = "completed"
        last_epoch = self.start_epoch
        baseline_info = None
        mlflow_ctx = self._start_mlflow_run() if self._is_main_process else nullcontext()

        max_epochs = self.epochs or sys.maxsize
        with mlflow_ctx:
            if self._is_main_process:
                self._log_mlflow_params()

            try:
                for epoch in range(self.start_epoch, max_epochs):
                    if self._should_stop():
                        stopped_reason = "early_stopped"
                        break
                    if self.max_steps and self.global_step >= self.max_steps:
                        stopped_reason = "max_steps_reached"
                        break

                    # 분산 학습: 매 에폭 셔플 순서 갱신
                    sampler = getattr(self.train_loader, "sampler", None)
                    if sampler is not None and hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(epoch)

                    self._fire("on_epoch_start", epoch=epoch)
                    train_loss = self._train_one_epoch(epoch)
                    self._fire(
                        "on_epoch_end", epoch=epoch, metrics={"train_loss": train_loss}
                    )

                    # Epoch-level metrics
                    self._mlflow_log_metric("epoch_train_loss", train_loss, epoch)
                    self._mlflow_log_metric(
                        "learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        epoch,
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

            finally:
                # Strategy cleanup
                if self.strategy is not None:
                    try:
                        self.strategy.cleanup()
                    except Exception as e:
                        logger.warning(f"Strategy cleanup 실패: {e}")

                training_duration = time.time() - start_time
                if self._is_main_process:
                    self._log_mlflow_summary(training_duration, stopped_reason)

        self._fire("on_train_end", metrics=self.last_metrics)

        result: dict[str, Any] = {
            "metrics": self.last_metrics,
            "training_duration_seconds": training_duration,
            "total_epochs": last_epoch - self.start_epoch + 1,
            "total_steps": self.global_step,
            "stopped_reason": stopped_reason,
        }
        if baseline_info is not None:
            result["monitoring"] = baseline_info

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

        for step, batch in enumerate(self.train_loader):
            if self.max_steps and self.global_step >= self.max_steps:
                break

            batch = self._move_to_device(batch)
            self._fire("on_batch_start", step=step)

            with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                loss = self._compute_loss(batch)
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if (
                    self.scheduler is not None
                    and self.scheduler_interval == "step"
                ):
                    self.scheduler.step()

                self.global_step += 1

            actual_loss = loss.item() * self.grad_accum_steps
            total_loss += actual_loss
            num_batches += 1

            if (step + 1) % self.grad_accum_steps == 0:
                self._fire(
                    "on_batch_end", step=step, epoch=epoch,
                    global_step=self.global_step,
                    metrics={"loss": actual_loss},
                )

                # MLflow step logging (non-blocking)
                self._mlflow_log_metric("train_loss", actual_loss, self.global_step)

            # Mid-epoch validation (step 단위 또는 소수 에폭)
            if (
                val_every_n > 0
                and self.val_loader is not None
                and (step + 1) % val_every_n == 0
                and (step + 1) < steps_in_epoch  # 에폭 마지막 step은 아래에서 처리
            ):
                self._run_validation(epoch)
                self.model.train()

        # mid-epoch 모드: 에폭 끝에서도 1회 검증 (마지막 구간 커버)
        if val_every_n > 0 and self.val_loader is not None:
            self._run_validation(epoch)
            self.model.train()

        return total_loss / max(num_batches, 1)

    def _compute_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.loss_fn is not None:
            outputs = self.model.forward(batch)
            logits = outputs.get("logits", outputs.get("output"))
            labels = batch.get("labels", batch.get("label"))
            if logits is None:
                raise ValueError("model.forward()가 'logits' 키를 반환하지 않았습니다")
            if labels is None:
                raise ValueError("배치에 'labels' 키가 없습니다")
            return self.loss_fn(logits, labels)
        else:
            return self.model.training_step(batch)

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict[str, float]:
        self.model.eval()
        all_metrics: dict[str, list[float]] = {}
        use_fallback = not hasattr(self.model, "validation_step")

        for batch in self.val_loader:
            batch = self._move_to_device(batch)
            if use_fallback:
                metrics = self._validate_fallback(batch)
            else:
                try:
                    metrics = self.model.validation_step(batch)
                except NotImplementedError:
                    use_fallback = True
                    metrics = self._validate_fallback(batch)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        # MLflow epoch logging
        for k, v in avg_metrics.items():
            self._mlflow_log_metric(f"val_{k}" if not k.startswith("val_") else k, v, epoch)

        self.model.train()
        return avg_metrics

    def _validate_fallback(self, batch: dict[str, Any]) -> dict[str, float]:
        """Fallback validation when model lacks validation_step."""
        device_type = self.device.type if self.device.type != "mps" else "cpu"
        with autocast(device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            outputs = self.model(batch)

        # Try outputs["loss"], outputs.loss, then loss_fn(logits, labels)
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif self.loss_fn is not None:
            logits = outputs.get("logits", outputs.get("output")) if isinstance(outputs, dict) else getattr(outputs, "logits", None)
            labels = batch.get("labels", batch.get("label"))
            if logits is not None and labels is not None:
                loss = self.loss_fn(logits, labels)
            else:
                logger.warning("_validate_fallback: logits 또는 labels를 찾을 수 없습니다")
                return {}
        else:
            logger.warning("_validate_fallback: loss를 계산할 수 없습니다")
            return {}

        return {"loss": loss.item() if isinstance(loss, torch.Tensor) else float(loss)}

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

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

        logger.info(f"체크포인트에서 재개: {ckpt_path}")

        # Model weights: adapter_model.safetensors → model.safetensors → model.pt
        adapter_path = ckpt_path / "adapter_model.safetensors"
        safetensors_path = ckpt_path / "model.safetensors"
        model_pt_path = ckpt_path / "model.pt"

        target = getattr(self.model, "module", self.model)

        if adapter_path.exists():
            # LoRA / PEFT adapter
            if hasattr(target, "load_adapter"):
                target.load_adapter(str(ckpt_path))
                logger.info("LoRA adapter loaded from %s", ckpt_path)
            else:
                logger.warning(
                    "adapter_model.safetensors found but model has no load_adapter method"
                )
        elif safetensors_path.exists():
            try:
                from safetensors.torch import load_file

                state_dict = load_file(safetensors_path)
                target.load_state_dict(state_dict)
            except ImportError:
                logger.warning("safetensors not installed, cannot load model.safetensors")
        elif model_pt_path.exists():
            state_dict = torch.load(model_pt_path, map_location="cpu", weights_only=True)
            target.load_state_dict(state_dict)

        # Optimizer
        opt_path = ckpt_path / "optimizer.pt"
        if opt_path.exists():
            self.optimizer.load_state_dict(
                torch.load(opt_path, map_location="cpu", weights_only=True)
            )

        # Scheduler
        sched_path = ckpt_path / "scheduler.pt"
        if sched_path.exists() and self.scheduler is not None:
            self.scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=True)
            )

        # GradScaler
        scaler_path = ckpt_path / "scaler.pt"
        if scaler_path.exists() and self.scaler.is_enabled():
            self.scaler.load_state_dict(
                torch.load(scaler_path, map_location="cpu", weights_only=True)
            )

        # Trainer state
        state_path = ckpt_path / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.start_epoch = state.get("epoch", 0)
            self.global_step = state.get("global_step", 0)

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

    # ── Checkpoint helpers ──

    def _find_best_checkpoint(self) -> Path | None:
        """best 또는 latest symlink가 가리키는 체크포인트 디렉토리를 반환한다."""
        ckpt_dir = Path(self.settings.config.storage.checkpoint_dir)
        for name in ("best", "latest"):
            link = ckpt_dir / name
            if link.exists():
                return link.resolve()
        return None

    def _export_and_log_model(self, checkpoint_dir: Path) -> None:
        """체크포인트에서 모델 artifact를 MLflow에 등록한다.

        LoRA 학습이면 adapter만, full finetuning이면 전체 모델을 저장한다.
        merge는 수행하지 않는다 — merge는 mdp export / mdp serve 시점에 on-demand.
        """
        import json
        import shutil
        import tempfile

        import mlflow

        try:
            recipe = self.settings.recipe
            model = self.model
            # DDP/FSDP에서 원본 모델 추출
            target = getattr(model, "module", model)

            with tempfile.TemporaryDirectory() as tmp:
                output_dir = Path(tmp)

                # 모델 가중치: PEFT면 adapter만, 아니면 전체
                has_adapter = hasattr(target, "save_pretrained") and hasattr(target, "peft_config")
                if has_adapter:
                    # adapter만 저장 (~50MB)
                    target.save_pretrained(output_dir)
                elif hasattr(target, "save_pretrained"):
                    # HF 모델 전체 저장
                    target.save_pretrained(output_dir)
                else:
                    from safetensors.torch import save_file
                    save_file(target.state_dict(), output_dir / "model.safetensors")

                # tokenizer 저장
                tokenizer_config = recipe.data.tokenizer
                if tokenizer_config:
                    pretrained = tokenizer_config.get("pretrained") if isinstance(tokenizer_config, dict) else getattr(tokenizer_config, "pretrained", None)
                    if pretrained:
                        try:
                            from transformers import AutoTokenizer
                            AutoTokenizer.from_pretrained(pretrained).save_pretrained(output_dir)
                        except Exception as e:
                            logger.warning(f"토크나이저 저장 실패 (무시): {e}")

                # recipe.yaml 복사
                recipe_src = checkpoint_dir / "recipe.yaml"
                if recipe_src.exists():
                    shutil.copy(recipe_src, output_dir / "recipe.yaml")

                # serving_meta.json
                meta = {
                    "model_class": type(target).__name__,
                    "head_replaced": recipe.head is not None,
                    "has_adapter": has_adapter,
                }
                (output_dir / "serving_meta.json").write_text(json.dumps(meta, indent=2))

                mlflow.log_artifacts(tmp, "model")
                logger.info("모델 artifact를 MLflow에 등록: model/")

        except Exception as e:
            logger.warning(f"모델 artifact 등록 실패 (학습 결과는 유효합니다): {e}")

    # ── MLflow ──

    def _start_mlflow_run(self) -> Any:
        """Create an MLflow run context (rank-0 only). Returns nullcontext() on failure."""
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

            run_kwargs = {}
            if hasattr(mlflow_cfg, "start_run") and isinstance(mlflow_cfg.start_run, dict):
                run_kwargs = mlflow_cfg.start_run

            return mlflow.start_run(**run_kwargs)
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
        """Run 시작 시 실험 재현에 필요한 하이퍼파라미터를 기록한다."""
        try:
            import mlflow

            recipe = self.settings.recipe
            params: dict[str, Any] = {
                "task": recipe.task,
                "model_class": recipe.model.class_path,
                "pretrained": recipe.model.pretrained or "none",
                "dataset_source": recipe.data.source,
                "batch_size": recipe.data.dataloader.batch_size,
                "epochs": self.epochs or 0,
                "max_steps": self.max_steps or 0,
                "precision": recipe.training.precision,
                "gradient_accumulation_steps": self.grad_accum_steps,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }

            # Adapter
            if recipe.adapter is not None:
                params["adapter_method"] = recipe.adapter.method
                if recipe.adapter.r is not None:
                    params["adapter_r"] = recipe.adapter.r

            # Strategy
            dist = self.settings.config.compute.distributed
            if isinstance(dist, dict) and dist.get("strategy"):
                params["strategy"] = dist["strategy"]

            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"MLflow params 로깅 실패: {e}")

    def _log_mlflow_summary(
        self, training_duration: float, stopped_reason: str,
    ) -> None:
        """Run 종료 시 최종 메트릭과 config snapshot을 기록한다."""
        try:
            import mlflow
            from mdp.utils.sanitize import sanitize_config

            mlflow.log_metrics(
                {
                    "training_duration_seconds": training_duration,
                    "total_steps": self.global_step,
                    **{f"final_{k}": v for k, v in self.last_metrics.items()},
                }
            )
            mlflow.set_tag("stopped_reason", stopped_reason)

            config_dict = sanitize_config(self.settings.model_dump())
            mlflow.log_dict(config_dict, "config/settings.json")

            # Best 체크포인트를 artifact로 등록
            best_ckpt = self._find_best_checkpoint()
            if best_ckpt:
                mlflow.log_artifacts(str(best_ckpt), "checkpoint")

                # 서빙 가능 모델 생성 + artifact 등록
                self._export_and_log_model(best_ckpt)

        except Exception as e:
            logger.warning(f"MLflow summary 로깅 실패 (학습 결과는 유효합니다): {e}")
