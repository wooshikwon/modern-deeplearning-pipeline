"""Training progress logging free functions shared by Trainer (SFT) and RLTrainer.

spec-system-logging-cleanup U4/U5 는 두 trainer 에 다음 6 helper 를 대칭으로
심었다:

- ``fmt_eta`` — ETA 초를 ``HH:MM:SS`` / ``MM:SS`` / ``--:--`` 포맷으로.
- ``log_step_progress`` — rank-0 한 줄 step-progress.
- ``log_run_banner`` — Start/End 배너.
- ``dump_oom_summary`` — OOM 발생 시 rank 별 memory 상태 집계 로그.
- ``maybe_start_memory_history`` — ``torch.cuda.memory._record_memory_history`` on/off.
- ``maybe_dump_memory_snapshot`` — snapshot pickle dump.

두 trainer 의 실질 차이는 (a) step-progress 에서 LR 조회 경로 (SFT 는 단일
optimizer, RL 은 dict["policy"]) 와 (b) start-banner 의 algorithm 필드 (SFT 는
recipe.task, RL 은 ``type(algorithm).__name__``) 뿐이다. 그 외 포맷·try/except
흡수·문서화는 문자 단위 동일하다 (cycle 1 review 2-1).

본 모듈은 이 6 helper 를 **free function** 으로 추출한다. BaseTrainer / 서브클래스는
얇은 bound method shim 을 상속하여 기존 테스트(``Trainer._dump_oom_summary(stub)``
등 직접 호출 패턴) 호환을 깨지 않는다. shim 은 caller-specific 상태(optimizer
lookup, algorithm name)만 해석하여 함수로 위임한다.

OOM summary 의 ``all_gather_object`` 는 다른 rank 가 이미 OOM 으로 사망한 상황
에서 NCCL collective 가 최대 수분간 hang 할 수 있다 (cycle 1 review 2-2). 본
모듈은 ``concurrent.futures.ThreadPoolExecutor`` 로 all-gather 를 timeout-wrap
하고, timeout 발생 시 local info 만으로 fallback 한다. backend-agnostic 하며
NCCL/gloo 모두에서 동일하게 동작한다.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "dump_oom_summary",
    "fmt_eta",
    "log_run_banner",
    "log_step_progress",
    "maybe_dump_memory_snapshot",
    "maybe_start_memory_history",
]


# ``all_gather_object`` timeout 상수 (초). 다른 rank 가 생존해 있을 때도 수집은
# 네트워크 latency 수준으로 끝나므로 5 초면 충분하다. 한 rank 가 OOM 으로 죽은
# 상황에서는 이 시간 내 응답이 없으므로 timeout → local fallback.
_OOM_ALL_GATHER_TIMEOUT_SEC: float = 5.0


# ---------------------------------------------------------------------------
# ETA 포맷
# ---------------------------------------------------------------------------


def fmt_eta(seconds: float | None) -> str:
    """ETA duration 을 ``HH:MM:SS`` (>1h) / ``MM:SS`` (<1h) 로 포맷.

    음수·inf·NaN 은 ``"--:--"`` 로 표기해 로그 라인이 파싱 불가능 상태로
    오염되는 것을 피한다. ``_log_step_progress`` 가 소비하며, 자체는 side
    effect 가 없는 순수 함수.
    """
    try:
        if (
            seconds is None
            or seconds != seconds  # NaN
            or seconds < 0
            or seconds == float("inf")
        ):
            return "--:--"
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    except Exception:  # noqa: BLE001 — format helper must never raise
        return "--:--"


# ---------------------------------------------------------------------------
# Step progress 로깅
# ---------------------------------------------------------------------------


def log_step_progress(
    *,
    logger: logging.Logger,
    global_step: int,
    max_steps: int,
    loss: float,
    current_lr: float,
    grad_norm: float | None,
    start_time: float,
) -> None:
    """Rank-0 한 줄 step-progress.

    caller 는 (a) rank-0 guard, (b) log_every_n_steps 타이밍을 모두 처리했다는
    전제로 호출한다. LR 조회는 trainer 별로 다르므로 호출부에서 결정된 값을
    ``current_lr`` 로 전달한다.
    """
    steps_done = max(global_step, 1)
    percent = 100.0 * global_step / max(max_steps, 1)
    elapsed = max(time.time() - start_time, 1e-9)
    throughput = steps_done / elapsed  # step/s
    remaining = max(max_steps - global_step, 0)
    eta_sec = remaining / max(throughput, 1e-9)

    grad_str = f"{grad_norm:.2f}" if grad_norm is not None else "--"
    logger.info(
        "[step %d/%d | %.1f%%] loss=%.4f lr=%.2e grad_norm=%s "
        "throughput=%.2f step/s ETA=%s",
        global_step, max_steps, percent,
        loss, current_lr, grad_str, throughput, fmt_eta(eta_sec),
    )


# ---------------------------------------------------------------------------
# Run banner (Start / End)
# ---------------------------------------------------------------------------


def log_run_banner(
    *,
    logger: logging.Logger,
    kind: str,
    is_main_process: bool,
    settings: Any,
    algorithm_label: str,
    strategy_name: str,
    max_steps: int | None,
    epochs: float | None,
    global_step: int,
    peak_memory_gib: float | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Start / End 배너를 rank-0 에서 한 번만 출력한다.

    ``kind`` 는 ``"start"`` 또는 ``"end"``. ``is_json_mode()`` 이면 구조화
    stdout 을 깨뜨리지 않기 위해 출력 자체를 건너뛴다.

    ``algorithm_label`` 은 trainer-specific:
      - RLTrainer: ``type(self.algorithm).__name__``
      - Trainer(SFT): ``recipe.task`` (algorithm 클래스가 없으므로)

    ``peak_memory_gib`` 는 end 배너에서만 사용 — caller 가 이미
    ``_peak_memory_summary_extra`` 등으로 계산한 값을 넘긴다 (중복 집계 금지).
    """
    # JSON 모드에서는 구조화 출력만 stdout 으로 흘러야 하므로 텍스트 배너 출력 금지.
    try:
        from mdp.cli.output import is_json_mode
        if is_json_mode():
            return
    except Exception:  # noqa: BLE001 — CLI 진입 경로 밖에서 호출되는 테스트 방어
        pass

    if not is_main_process:
        return

    recipe = getattr(settings, "recipe", None)
    config = getattr(settings, "config", None)

    if kind == "start":
        precision = getattr(recipe.training, "precision", "fp32") if recipe else "?"
        max_steps_val = max_steps if max_steps is not None else "-"
        epochs_val = f"{epochs:.2f}" if epochs is not None else "-"
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        bs_per_rank = (
            getattr(recipe.data.dataloader, "batch_size", "?") if recipe else "?"
        )
        experiment = (
            getattr(config.mlflow, "experiment_name", None)
            if config and getattr(config, "mlflow", None) is not None
            else None
        ) or "-"
        run_id = (extra or {}).get("run_id") or "-"

        lines = [
            "=" * 62,
            f"MDP Run Started | algorithm={algorithm_label} strategy={strategy_name} "
            f"precision={precision}",
            f"max_steps={max_steps_val} epochs={epochs_val} "
            f"world_size={world_size} bs_per_rank={bs_per_rank}",
            f"experiment={experiment} run_id={run_id}",
            "=" * 62,
        ]
        for line in lines:
            logger.info(line)
        return

    if kind == "end":
        extras = extra or {}
        stopped_reason = extras.get("stopped_reason", "-")
        duration = extras.get("duration", 0.0)
        checkpoints_saved = extras.get("checkpoints_saved", 0)
        final_loss = extras.get("final_loss")
        total_steps = extras.get("total_steps", global_step)

        peak_str = (
            f"{peak_memory_gib:.2f} GiB" if peak_memory_gib is not None else "n/a"
        )

        final_loss_str = (
            f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "-"
        )
        lines = [
            "=" * 62,
            f"MDP Run Ended   | stopped_reason={stopped_reason} "
            f"duration={duration:.1f}s peak_memory={peak_str}",
            f"checkpoints_saved={checkpoints_saved} final_loss={final_loss_str} "
            f"total_steps={total_steps}",
            "=" * 62,
        ]
        for line in lines:
            logger.info(line)


# ---------------------------------------------------------------------------
# OOM summary
# ---------------------------------------------------------------------------


def _all_gather_with_timeout(
    local_info: dict,
    timeout_sec: float,
    logger: logging.Logger,
) -> list[dict | None]:
    """``dist.all_gather_object`` 를 별도 daemon thread 에서 실행, timeout 시
    local fallback.

    다른 rank 가 이미 OOM 으로 사망한 상황에서 NCCL collective 는 기본 timeout
    (수분~수십분) 까지 hang 한다. 본 래퍼는 daemon thread + Future 로 timeout 을
    bound 하여 "최대 ``timeout_sec`` 초만 기다리고 local info 만 반환" 하는
    backend-agnostic 방어책을 제공한다.

    ``ThreadPoolExecutor`` 를 context manager 로 쓰지 않는다 — ``__exit__`` 이
    worker 종료까지 대기하기 때문에 timeout 이후에도 main 경로가 hang 된다.
    대신 ``shutdown(wait=False)`` 로 즉시 반환하고, daemon 속성으로 worker 가
    인터프리터 종료 시 강제 종료되도록 한다.
    """
    try:
        import torch.distributed as dist
    except Exception as e:  # noqa: BLE001
        logger.debug("torch.distributed import 실패: %s", e)
        return [local_info]

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return [local_info]

    world_size = dist.get_world_size()

    def _gather() -> list[dict | None]:
        output: list[dict | None] = [None] * world_size
        dist.all_gather_object(output, local_info)
        return output

    pool: concurrent.futures.ThreadPoolExecutor | None = None
    try:
        # daemon thread 로 띄워 인터프리터 종료 시 자연 종료. thread_name_prefix
        # 는 관측성용.
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mdp-oom-gather",
        )
        future = pool.submit(_gather)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "OOM summary all_gather_object timeout (%.1fs) — "
                "local info fallback",
                timeout_sec,
            )
            return [local_info]
    except Exception as e:  # noqa: BLE001 — 어떤 예외든 local fallback
        logger.debug("all_gather for OOM summary skipped: %s", e)
        return [local_info]
    finally:
        if pool is not None:
            # wait=False 로 worker 완료 대기 없이 즉시 반환. hang 된 worker 는
            # 인터프리터 종료 시 daemon thread 로 자연 종료.
            pool.shutdown(wait=False)


def dump_oom_summary(
    *,
    logger: logging.Logger,
    timeout_sec: float = _OOM_ALL_GATHER_TIMEOUT_SEC,
) -> None:
    """OOM 발생 시 모든 rank 의 memory 상태를 rank-0 에 집계하여 logger.error
    로 흘린다.

    단일 프로세스 환경에서는 local info 만 출력. Distributed 환경에서는
    ``dist.all_gather_object`` 로 모든 rank 상태를 수집하되, timeout
    (기본 5 초) 안에 완료되지 않으면 local fallback 하여 hang 을 차단한다
    (cycle 1 review 2-2).

    이미 일부 rank 가 OOM 으로 죽어 NCCL collective 가 예외를 던지는 경로도
    흡수 — 어떤 경우든 rank-0 는 최대 ``timeout_sec`` 초 내에 자기 상태를
    기록하고 run 을 이어간다.
    """
    if not torch.cuda.is_available():
        return

    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
    except Exception:  # noqa: BLE001 — mem_get_info 는 일부 환경에서 미지원
        free_gib = float("nan")

    local_info = {
        "rank": int(os.environ.get("RANK", "0")),
        "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
        "free_gib": free_gib,
    }

    gathered = _all_gather_with_timeout(local_info, timeout_sec, logger)

    # rank-0 만 출력 — 비-rank-0 에서 집계 자체를 건너뛰어 불필요한 문자열
    # 포맷 비용을 피한다.
    if int(os.environ.get("RANK", "0")) != 0:
        return

    valid = [info for info in gathered if info is not None]
    lines = ["", "FATAL: torch.OutOfMemoryError — rank-level memory summary:"]
    for info in sorted(valid, key=lambda x: x["rank"]):
        free_fmt = (
            f"{info['free_gib']:5.2f} GiB"
            if info["free_gib"] == info["free_gib"]  # NaN 체크
            else "  n/a  "
        )
        lines.append(
            f"  rank {info['rank']}: "
            f"allocated={info['allocated_gib']:6.2f} GiB | "
            f"reserved={info['reserved_gib']:6.2f} GiB | "
            f"free={free_fmt}"
        )
    low_free = [
        info for info in valid
        if info["free_gib"] == info["free_gib"] and info["free_gib"] < 1.0
    ]
    if low_free:
        ranks = [info["rank"] for info in low_free]
        lines.append(f"  → OOM suspected on rank(s): {ranks}")
    logger.error("\n".join(lines))


# ---------------------------------------------------------------------------
# memory_history on/off
# ---------------------------------------------------------------------------


def maybe_start_memory_history(
    *,
    recipe_dict: Any,
    logger: logging.Logger,
) -> bool:
    """``monitoring.memory_history=True`` 면 tensor-level snapshot 수집을 켠다.

    rank-0 만 켠다 — multi-rank 저장 시 동일 pickle 파일을 여러 rank 가
    덮어쓰는 경합을 피한다. 예외가 나면 warning 후 False 반환하여 학습은
    계속 진행되도록 한다.

    Returns:
        실제로 memory_history 가 활성되었는지 여부. caller 는 이 반환값을
        받아 ``maybe_dump_memory_snapshot`` 실행을 게이트한다.
    """
    mon_cfg = recipe_dict.get("monitoring", {}) if isinstance(recipe_dict, dict) else {}
    if not mon_cfg.get("memory_history", False):
        return False
    if not torch.cuda.is_available():
        return False
    if int(os.environ.get("RANK", "0")) != 0:
        return False

    try:
        torch.cuda.memory._record_memory_history(
            max_entries=1_000_000, stacks="python",
        )
        logger.info(
            "monitoring.memory_history enabled — _record_memory_history started",
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("memory_history start failed: %s", e)
        return False


def maybe_dump_memory_snapshot(
    *,
    active: bool,
    logger: logging.Logger,
) -> None:
    """``maybe_start_memory_history`` 가 성공했을 때에만 snapshot 을 파일로 dump."""
    if not active:
        return
    try:
        snap_dir = Path("storage/memory_profiles")
        snap_dir.mkdir(parents=True, exist_ok=True)

        # run_id: MLflow active run 우선, 없으면 timestamp.
        run_id: str
        try:
            import mlflow
            _run = mlflow.active_run()
            if _run is not None and hasattr(_run, "info"):
                run_id = _run.info.run_id
            else:
                run_id = f"run_{int(time.time())}"
        except Exception:  # noqa: BLE001 — mlflow 미설치·미초기화 등
            run_id = f"run_{int(time.time())}"

        snap_path = snap_dir / f"{run_id}.pickle"
        torch.cuda.memory._dump_snapshot(str(snap_path))
        logger.info("memory_history snapshot saved to %s", snap_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("memory snapshot save failed: %s", e)
