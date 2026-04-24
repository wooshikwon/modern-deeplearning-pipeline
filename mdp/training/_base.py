"""BaseTrainer — Trainer / RLTrainer 공통 추상 기반 클래스.

spec-training-restructure U2 에서 신설. Trainer(SFT) 와 RLTrainer 가 복제해 온
OOM / system logging / MLflow wrapper shim 을 단일 위치로 통합한다.

책임:
- 공통 shim: ``_move_to_device``, ``_should_stop``, ``_estimate_total_steps``
- OOM / memory_history wrapper: ``_progress_log`` free function 호출을 래핑
- System logging wrapper: step-progress · run-banner (LR 조회는 abstract method 로 위임)
- MLflow lifecycle wrapper: start / params / summary / peak-memory
- Checkpoint state 훅: ``_checkpoint_state()`` / ``_load_checkpoint_state()``
  — Trainer / RLTrainer 가 각각 override 하여 구현한다.

MRO: ``BaseTrainer(object)`` 만 상속 — 추가 mixin 없음. 기존 테스트의
``Trainer._dump_oom_summary(stub)`` 같은 직접 호출 패턴은 상속을 통해 그대로
호환된다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any

import torch


from mdp.training._progress_log import (
    dump_oom_summary,
    fmt_eta,
    log_run_banner,
    log_step_progress,
    maybe_dump_memory_snapshot,
    maybe_start_memory_history,
)

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Trainer / RLTrainer 공통 추상 기반 클래스."""

    # ── 서브클래스 제공 속성 (타입 선언만, 초기화는 서브클래스 __init__) ──────────
    device: torch.device
    callbacks: list
    _stop_requested: bool
    # training-loop configuration (set before train() is called)
    max_steps: int | None
    epochs: int | None
    grad_accum_steps: int
    train_loader: Any  # torch.utils.data.DataLoader
    global_step: int
    _recipe_dict: dict[str, Any]
    _is_main_process: bool
    settings: Any  # TrainerSettings; typed as Any to avoid circular import

    # ── 추상 메서드 ───────────────────────────────────────────────────────────

    @abstractmethod
    def _optimizer_for_progress_log(self) -> torch.optim.Optimizer | None:
        """step-progress 로그에서 LR 을 읽을 단일 optimizer 를 반환한다.

        - Trainer(SFT): ``self.optimizer``
        - RLTrainer: ``self.optimizers.get("policy")``

        LR 조회 실패 가능성을 caller 가 고민하지 않아도 되도록 서브클래스가
        None-safe 하게 반환해야 한다.
        """
        ...

    @abstractmethod
    def _collect_mlflow_params(self) -> None:
        """Run 시작 시 MLflow 에 정적 파라미터를 기록한다.

        두 trainer 모두 ``log_static_params(self.settings.recipe, self.settings)``
        를 호출하는 동일 패턴이지만, MLflow 기록 외에 trainer 별 부가 동작이
        있을 수 있으므로 서브클래스에서 구현한다.
        """
        ...

    @abstractmethod
    def _checkpoint_state(self) -> dict:
        """현재 학습 상태를 dict 로 직렬화한다.

        반환 dict 는 ``_load_checkpoint_state`` 에 그대로 전달될 수 있어야 한다.
        Trainer / RLTrainer 가 각각 override 하여 실제 직렬화 로직을 구현한다.
        """
        ...

    # ── 공통 구현 ────────────────────────────────────────────────────────────

    def _load_checkpoint_state(self, state: dict) -> None:
        """``_checkpoint_state`` 의 역방향 — state dict 로 학습 상태를 복원한다.

        BaseTrainer 의 기본 구현은 no-op. Trainer / RLTrainer 가 override 하여
        실제 복원 로직을 구현한다. 서브클래스 override 없이 직접 호출하면 아무 동작도 하지 않는다.
        """

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """배치 텐서를 ``self.device`` 로 이동한다."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _should_stop(self) -> bool:
        """EarlyStopping 콜백이나 signal 로 인해 학습을 중단해야 하는지 반환한다."""
        if self._stop_requested:
            return True
        return any(getattr(cb, "should_stop", False) for cb in self.callbacks)

    def _estimate_total_steps(self) -> int:
        """설정에서 총 예상 step 수를 계산한다.

        서브클래스의 ``__init__`` 이 완료되기 전 (스케줄러 생성 시) 호출되므로
        ``self.max_steps``, ``self.epochs``, ``self.grad_accum_steps``,
        ``self.train_loader`` 가 이미 세팅되어 있다고 가정한다.
        """
        if self.max_steps:
            return self.max_steps
        steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
        return int(steps_per_epoch * (self.epochs or 1))

    # ── OOM / memory_history shim ────────────────────────────────────────────

    def _dump_oom_summary(self) -> None:
        """OOM 발생 시 모든 rank 의 memory 상태를 rank-0 에 집계한다.

        ``_progress_log.dump_oom_summary`` 로 위임. 5초 타임아웃 내 all-gather
        후 local fallback (cycle 1 review 2-2).
        """
        dump_oom_summary(logger=logger)

    def _maybe_start_memory_history(self) -> bool:
        """``monitoring.memory_history=True`` 면 tensor-level snapshot 수집을 켠다.

        ``_progress_log.maybe_start_memory_history`` 로 위임. rank-0 전용.
        """
        return maybe_start_memory_history(
            recipe_dict=self._recipe_dict, logger=logger,
        )

    def _maybe_dump_memory_snapshot(self, active: bool) -> None:
        """``_maybe_start_memory_history`` 가 성공했을 때에만 snapshot 을 dump 한다."""
        maybe_dump_memory_snapshot(active=active, logger=logger)

    # ── System logging shim ──────────────────────────────────────────────────

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        """ETA 초를 ``HH:MM:SS`` 또는 ``MM:SS`` 로 포맷한다.

        ``_progress_log.fmt_eta`` 로 위임. 음수·inf·NaN 은 ``"--:--"``.
        """
        return fmt_eta(seconds)

    def _log_step_progress(
        self,
        loss: float,
        grad_norm: float | None,
        *,
        start_time: float,
        max_steps: int,
    ) -> None:
        """rank-0 text step-progress 한 줄을 로거로 흘린다.

        caller 는 (a) rank-0 guard, (b) log_every_n_steps 타이밍을 모두 처리했다
        는 전제로 호출한다.

        LR 은 ``_optimizer_for_progress_log()`` 로 조회한다:
        - Trainer(SFT): 단일 ``self.optimizer.param_groups[0]["lr"]``
        - RLTrainer: ``self.optimizers["policy"].param_groups[0]["lr"]``

        """
        try:
            opt = self._optimizer_for_progress_log()
            current_lr = (
                opt.param_groups[0]["lr"]
                if opt is not None and opt.param_groups
                else 0.0
            )
        except Exception:  # noqa: BLE001
            current_lr = 0.0

        log_step_progress(
            logger=logger,
            global_step=self.global_step,
            max_steps=max_steps,
            loss=loss,
            current_lr=current_lr,
            grad_norm=grad_norm,
            start_time=start_time,
        )

    def _log_run_banner(
        self,
        kind: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Start / End 배너를 rank-0 에서 한 번만 출력한다.

        ``kind`` 는 ``"start"`` 또는 ``"end"``. ``is_json_mode()`` 이면 skip.

        ``algorithm_label`` 은 서브클래스에서 ``_algorithm_label()`` 를 override
        해 주입한다:
        - Trainer(SFT): ``recipe.task``
        - RLTrainer: ``type(self.algorithm).__name__``
        """
        strategy = getattr(self, "strategy", None)
        strategy_name = type(strategy).__name__ if strategy is not None else "NoStrategy"
        _get_label = getattr(self, "_algorithm_label", None)
        algorithm_label = _get_label() if callable(_get_label) else "unknown"

        peak_memory_gib = None
        if kind == "end":
            peak_summary = self._peak_memory_summary_extra() or {}
            peak_memory_gib = peak_summary.get("peak_memory_gb")

        log_run_banner(
            logger=logger,
            kind=kind,
            is_main_process=self._is_main_process,
            settings=self.settings,
            algorithm_label=algorithm_label,
            strategy_name=strategy_name,
            max_steps=self.max_steps,
            epochs=self.epochs,
            global_step=self.global_step,
            peak_memory_gib=peak_memory_gib,
            extra=extra,
        )

    def _algorithm_label(self) -> str:
        """배너에 쓸 algorithm 식별자를 반환한다.

        - Trainer(SFT): override 하여 ``recipe.task`` 를 반환
        - RLTrainer: override 하여 ``type(self.algorithm).__name__`` 를 반환
        """
        return "unknown"

    # ── MLflow lifecycle shim ────────────────────────────────────────────────

    def _start_mlflow_run(self) -> Any:
        """MLflow run context 를 시작한다. 실패 시 nullcontext() 반환 (rank-0 only)."""
        try:
            import mlflow

            mlflow_cfg = self.settings.config.mlflow
            if mlflow_cfg is None:
                return nullcontext()

            if hasattr(mlflow_cfg, "tracking_uri") and mlflow_cfg.tracking_uri:
                mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
            experiment_name = (
                getattr(mlflow_cfg, "experiment_name", None)
                or getattr(mlflow_cfg, "experiment", None)
            )
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            run_kwargs = {}
            if hasattr(mlflow_cfg, "start_run") and isinstance(mlflow_cfg.start_run, dict):
                run_kwargs = mlflow_cfg.start_run
            return mlflow.start_run(**run_kwargs)
        except Exception as e:
            logger.warning(f"MLflow run 시작 실패: {e}")
            return nullcontext()

    def _log_mlflow_params(self) -> None:
        """``_collect_mlflow_params`` 로 위임 — 서브클래스 구현 호출."""
        self._collect_mlflow_params()

    @abstractmethod
    def _log_mlflow_summary(
        self,
        training_duration: float,
        stopped_reason: str,
        **kwargs: Any,
    ) -> None:
        """Run 종료 시 summary 로깅. 서브클래스가 override 해 구현한다.

        BaseTrainer 는 인터페이스 선언만 하며, 실제 로직은 각 trainer 고유의
        ``log_summary`` 호출 방식이 달라 서브클래스에 유지된다.
        """
        ...

    def _peak_memory_summary_extra(self) -> dict[str, float] | None:
        """rank 0 의 CUDA peak memory 를 GiB 로 반환한다.

        CUDA 미가용 또는 예외 시 ``None`` 반환 — summary 에 영향 없음.
        Trainer / RLTrainer 에 동일 구현이 있었으므로 BaseTrainer 로 올렸다.
        """
        try:
            if not torch.cuda.is_available():
                return None
            peak_bytes = torch.cuda.max_memory_allocated()
            if peak_bytes <= 0:
                return None
            return {"peak_memory_gb": peak_bytes / (1024**3)}
        except Exception as e:  # noqa: BLE001
            logger.debug("peak_memory_gb 집계 실패(무시): %s", e)
            return None
