"""Shared utilities for SFT Trainer and RLTrainer.

Device detection, distributed strategy creation, expert parallelism setup,
and backward/optimizer step logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)


def aggregate_checkpoint_stats(
    callbacks: list,
) -> tuple[int, Path | None, str]:
    """duck-typed saved_checkpoints 집계. Trainer/RLTrainer 공용.

    Trainer._log_mlflow_summary / RLTrainer._log_mlflow_summary 양쪽이 동일한
    방식으로 self.callbacks를 순회하며 "저장 개수 합산 + 첫 best 경로 선정 +
    zero-warning용 monitor hint"를 뽑아내는 블록을 6개 사이트(메인 집계 2 +
    결과 dict fallback 재집계 2 + monitor hint 2)에 걸쳐 복제해 왔다. 한 쪽을
    고치면 다른 쪽도 반드시 고쳐야 하는 강결합이므로, spec §3.2의 duck typing
    규칙을 단일 헬퍼로 확립한다.

    동작 규칙:
    - ``hasattr(cb, "saved_checkpoints")``만을 기준으로 사용한다 — isinstance 검사
      대신 duck typing을 써서 ModelCheckpoint 서브클래스, 다중 인스턴스
      (예: Critic + Policy RL 구성), frozen 모델용 callback(빈 리스트 → 0 기여)을
      자연스럽게 포함한다.
    - ``best_path``는 **규칙 A** (첫 non-empty best)를 따른다. ``best_models``는
      ``[(metric_value, path_str), ...]`` worst-first 정렬이므로 마지막 요소가
      best이며, 여러 콜백이 있으면 첫 매칭 콜백의 best를 채택한다.
    - ``monitor_hint``는 ``saved_checkpoints`` 속성을 가진 콜백 중 ``monitor``를
      가진 것만 CSV로 합쳐 반환한다. 빈 경우 안내 문자열로 대체하여 호출자가
      그대로 로그에 넣을 수 있게 한다.

    Args:
        callbacks: Trainer/RLTrainer의 ``self.callbacks`` 리스트.

    Returns:
        ``(total, best_path, monitor_hint)``:
        - ``total``: 모든 콜백의 ``saved_checkpoints`` 길이 합산
        - ``best_path``: 규칙 A로 선정된 best 경로 (없으면 ``None``)
        - ``monitor_hint``: count=0 warning에 그대로 쓸 수 있는 monitor 이름 CSV
          (없으면 ``"(no ModelCheckpoint configured)"``)
    """
    total = 0
    best_path: Path | None = None
    monitor_names: list[str] = []
    for cb in callbacks:
        if not hasattr(cb, "saved_checkpoints"):
            continue
        total += len(cb.saved_checkpoints)
        if hasattr(cb, "monitor"):
            monitor_names.append(cb.monitor)
        if best_path is None:
            best_models = getattr(cb, "best_models", None)
            if best_models:
                best_path = Path(best_models[-1][1])
    monitor_hint = ", ".join(monitor_names) or "(no ModelCheckpoint configured)"
    return total, best_path, monitor_hint


def setup_amp(
    precision: str, device: torch.device,
) -> tuple[bool, torch.dtype, GradScaler]:
    """AMP/GradScaler를 설정한다. Trainer/RLTrainer 공용."""
    scaler_device = device.type
    if precision == "fp16":
        amp_enabled = True
        amp_dtype = torch.float16
        if device.type == "mps":
            logger.warning("MPS에서 fp16은 GradScaler를 지원하지 않습니다. bf16을 권장합니다.")
            scaler = GradScaler(scaler_device, enabled=False)
        else:
            scaler = GradScaler(scaler_device, enabled=True)
    elif precision == "bf16":
        amp_enabled = True
        amp_dtype = torch.bfloat16
        scaler = GradScaler(scaler_device, enabled=False)
    else:  # fp32
        amp_enabled = False
        amp_dtype = torch.float32
        scaler = GradScaler(scaler_device, enabled=False)
    return amp_enabled, amp_dtype, scaler


def load_callbacks_from_file(path: str) -> list[dict[str, Any]]:
    """콜백 YAML 파일을 읽어 list[dict] 설정을 반환한다.

    파일 형식은 Recipe의 callbacks 섹션과 동일:
    ``[{_component_: Name, ...}, ...]``
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"콜백 파일은 리스트여야 합니다 (실제: {type(raw).__name__}). "
            "예: [{_component_: EarlyStopping, patience: 3}]"
        )
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "_component_" not in item:
            raise ValueError(
                f"콜백 항목 [{i}]에 _component_ 키가 필요합니다: {item}"
            )
    return raw


def create_callbacks(configs: list[dict[str, Any]], resolver: Any) -> list:
    """Recipe의 callbacks 설정에서 콜백 리스트를 생성한다."""
    callbacks = []
    for cfg in configs:
        try:
            callbacks.append(resolver.resolve(cfg))
        except Exception as e:
            logger.warning("콜백 생성 실패: %s", e)
    return callbacks


def detect_device() -> torch.device:
    """Detect the best available device (CUDA > MPS > CPU).

    분산 학습(torchrun)에서는 LOCAL_RANK 환경 변수를 읽어 각 rank의
    전용 GPU를 반환한다. LOCAL_RANK가 없으면 cuda:0(단일 GPU)을 반환.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def auto_strategy(**kwargs: Any) -> Any:
    """GPU 수에 따라 적절한 분산 전략을 자동 선택한다.

    멀티 GPU → DDPStrategy, 단일 GPU/CPU → None.
    aliases.yaml에서 ``auto`` 로 등록되어 ``_component_: auto``로 사용한다.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return None
    from mdp.training.strategies.ddp import DDPStrategy

    return DDPStrategy(**kwargs)


def create_strategy(settings: Settings, resolver: Any) -> Any:
    """Config.compute.distributed에서 분산 전략을 생성한다. None이면 전략 없음."""
    dist_config = settings.config.compute.distributed
    if dist_config is None:
        return None
    if not isinstance(dist_config, dict):
        return None

    strategy_name = dist_config.get("strategy", "auto")
    if strategy_name == "none":
        return None

    # strategy 값이 이미 _component_ dict이면 직접 resolve
    if isinstance(strategy_name, dict):
        return resolver.resolve(strategy_name)

    # 문자열이면 aliases.yaml에서 조회
    strategy_kwargs = {
        k: v for k, v in dist_config.items()
        if k not in ("strategy", "moe")
    }
    return resolver.resolve({"_component_": strategy_name, **strategy_kwargs})


def create_expert_parallel(settings: Settings) -> Any:
    """Create ExpertParallel from distributed.moe config, or return None."""
    dist_config = settings.config.compute.distributed
    if dist_config is None or not isinstance(dist_config, dict):
        return None
    moe_config = dist_config.get("moe")
    if moe_config is None or not moe_config.get("enabled", False):
        return None

    from mdp.training.strategies.moe import ExpertParallel

    return ExpertParallel(
        ep_size=moe_config.get("ep_size", moe_config.get("expert_parallel_size", 1)),
        expert_module_pattern=moe_config.get("expert_module_pattern", "experts"),
    )


def backward_and_step(
    losses: dict[str, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, Any | None],
    scaler: GradScaler,
    trainable_models: dict[str, nn.Module],
    grad_accum_steps: int,
    at_accum_boundary: bool,
    grad_clip_norm: float | None = None,
    force_step: bool = False,
) -> bool | None:
    """Shared backward + optimizer step.

    Returns:
        True: optimizer step executed.
        False: backward done, not at accumulation boundary.
        None: NaN/Inf detected, gradients cleared, caller should skip.
    """
    # NaN/Inf guard
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            logger.warning("NaN/Inf loss detected in '%s', skipping step", name)
            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)
            return None

    # Backward with accumulation scaling
    accum = 1 if force_step else grad_accum_steps
    for loss in losses.values():
        scaler.scale(loss / accum).backward()

    # Optimizer step at accumulation boundary or force
    if force_step or at_accum_boundary:
        for name, opt in optimizers.items():
            scaler.unscale_(opt)
            if grad_clip_norm is not None and name in trainable_models:
                clip_grad_norm_(trainable_models[name].parameters(), grad_clip_norm)
            scaler.step(opt)
            sched = schedulers.get(name)
            if sched is not None:
                sched.step()
        scaler.update()

        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        return True

    return False
