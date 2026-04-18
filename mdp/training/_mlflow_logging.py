"""MLflow 로깅 공용 헬퍼 모듈.

Trainer·RLTrainer가 MLflow를 직접 호출하지 않고 이 모듈의 함수를 경유하도록
만들어 "어떤 값이 param(static) / metric(dynamic) / tag·artifact(summary)로
가야 하는가"라는 분류를 한 곳에서 결정한다. 본 모듈은 spec-logging-consistency
§"U1: 공용 로깅 헬퍼 모듈 신설"의 구현체이며, U2·U3·U5는 여기서 공개한
심볼만 소비한다.

원칙 요약:

- **원칙 1 (값의 변동성이 저장소를 결정)**: Recipe·Settings 선언값은 params로,
  scheduler로 변하는 값은 metrics로, run 종료 시점에 확정되는 집계는 tag/metric/
  artifact로 분리한다.
- **원칙 2 (static 값은 이중 기록이 아닌 역할 분리)**: ``log_static_params``는
  optimizer 인스턴스 상태를 param으로 내보내지 않는다. warmup step 0 스냅샷이
  param에 박혀 사용자를 오도하던 구조적 결함(weighted-ntp Phase 3 사례)을
  해소한다.
- **원칙 3 (multi-group slash 네이밍)**: ``collect_optimizer_state``가
  param_groups 전체를 순회하며 아래 키 규약을 따른다:

  | 축 | 키 형태 |
  |---|---|
  | single-group, single-optimizer | ``learning_rate`` |
  | single-group, multi-optimizer  | ``learning_rate/{opt_name}`` |
  | multi-group, single-optimizer  | ``learning_rate/group_{idx}`` (param_group에 ``name`` 있으면 ``learning_rate/{name}``) |
  | multi-group, multi-optimizer   | ``learning_rate/{opt_name}/group_{idx}`` (name 있으면 ``/{name}``) |

모든 함수는 ``mlflow.active_run()``이 ``None``이면 no-op으로 동작한다. DDP
rank 가드는 caller의 ``_is_main_process``에 위임한다. 본 모듈은 thread-safe일
필요 없다(rank-0에서만 호출).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import torch

if TYPE_CHECKING:
    from mdp.settings.schema import Recipe, Settings


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def collect_optimizer_state(
    optimizers: Mapping[str, torch.optim.Optimizer],
) -> dict[str, float]:
    """``param_groups`` 전체를 순회하며 learning_rate·momentum·weight_decay 수집.

    반환 dict 키는 원칙 3의 slash 네이밍 규약을 엄격히 따른다. 반환값을 그대로
    ``mlflow.log_metrics``에 전달할 수 있도록 값은 모두 ``float``.

    momentum·weight_decay는 해당 param_group에 키가 실제로 존재하는 경우에만
    포함한다. SGD는 momentum을 갖지만 Adam은 갖지 않으므로, optimizer 종류를
    caller가 몰라도 자연스럽게 분기된다.

    Args:
        optimizers: ``{name: Optimizer}`` 형태의 dict. 단일 optimizer라도
            ``{"policy": opt}`` 같이 감싸서 전달. 이 규약은 RLTrainer의 기존
            ``self.optimizers`` 시그니처와 동일하여 Trainer 쪽에서만 얇은
            래퍼 메서드가 필요하다.

    Returns:
        metric 키·값 dict. Optimizer 개수와 param_group 수에 따라 접두/접미가
        붙는다.
    """
    # multi-optimizer 여부는 dict 크기로 판단한다.
    multi_optimizer = len(optimizers) > 1

    result: dict[str, float] = {}
    for opt_name, opt in optimizers.items():
        groups = opt.param_groups
        multi_group = len(groups) > 1

        for idx, group in enumerate(groups):
            # group suffix — multi-group에서만 붙는다. name 우선.
            group_name: str | None = group.get("name") if isinstance(group, Mapping) else None
            if multi_group:
                group_suffix = f"/{group_name}" if group_name else f"/group_{idx}"
            else:
                group_suffix = ""

            # optimizer prefix — multi-optimizer에서만 붙는다.
            opt_prefix = f"/{opt_name}" if multi_optimizer else ""

            # key = learning_rate[/{opt_name}][/{group_name_or_idx}]
            lr_key = f"learning_rate{opt_prefix}{group_suffix}"
            result[lr_key] = float(group["lr"])

            if "momentum" in group:
                result[f"momentum{opt_prefix}{group_suffix}"] = float(group["momentum"])
            if "weight_decay" in group:
                result[f"weight_decay{opt_prefix}{group_suffix}"] = float(group["weight_decay"])

    return result


def log_static_params(
    recipe: "Recipe",
    settings: "Settings",
) -> None:
    """Run 시작 시 한 번 호출하여 Recipe·Settings 선언값을 ``log_params``로 기록.

    원칙 2에 따라 optimizer 인스턴스 상태는 **읽지 않는다**. 학습률은
    ``learning_rate_init`` 키로 기록되며 출처는:

    - RL: ``recipe.rl.models["policy"]["optimizer"]["lr"]``
    - SFT: ``recipe.optimizer["lr"]``

    해당 필드가 모두 부재하는 경우에만 param에서 생략한다(fallback으로 optimizer
    인스턴스를 읽는 경로는 본 모듈에서는 지원하지 않는다 — 그 경로는 caller가
    명시적으로 optimizer를 전달할 때만 동작해야 하는 별도 관심사이며, 기본
    시그니처에 포함하지 않는 이유는 spec §"위험 시나리오 4"의 정책 결정).

    ``mlflow.active_run()``이 ``None``이면 모든 쓰기는 생략된다.

    Args:
        recipe: ``Settings.recipe``. task / model / adapter / training / optimizer
            / data 등을 소스로 사용.
        settings: ``Settings`` 전체. ``config.compute.distributed.strategy``를
            strategy param 산출에 사용.
    """
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        is_rl = recipe.rl is not None
        params: dict[str, Any] = {"task": recipe.task}

        if is_rl:
            policy_spec = recipe.rl.models.get("policy", {})
            algo = recipe.rl.algorithm
            if isinstance(algo, Mapping):
                params["algorithm"] = algo.get("_component_", "unknown")
            params["policy_class"] = policy_spec.get("_component_", "unknown")
            params["pretrained"] = policy_spec.get("pretrained", "none")
            # RL learning_rate_init: recipe.rl.models.policy.optimizer.lr
            policy_optimizer = policy_spec.get("optimizer") or {}
            if "lr" in policy_optimizer:
                params["learning_rate_init"] = policy_optimizer["lr"]
            # RL adapter는 policy_spec 안쪽 adapter
            adapter = policy_spec.get("adapter")
        else:
            params["model_class"] = recipe.model.get("_component_", "unknown")
            params["pretrained"] = recipe.model.get("pretrained", "none")
            # SFT learning_rate_init: recipe.optimizer.lr
            sft_optimizer = recipe.optimizer or {}
            if isinstance(sft_optimizer, Mapping) and "lr" in sft_optimizer:
                params["learning_rate_init"] = sft_optimizer["lr"]
            adapter = recipe.adapter

        # 데이터·학습 공통 파라미터
        params["dataset_source"] = recipe.data.dataset.get("source", "unknown")
        params["batch_size"] = recipe.data.dataloader.batch_size
        params["epochs"] = recipe.training.epochs if recipe.training.epochs is not None else 0
        params["max_steps"] = recipe.training.max_steps if recipe.training.max_steps is not None else 0
        params["precision"] = recipe.training.precision
        params["gradient_accumulation_steps"] = recipe.training.gradient_accumulation_steps

        # Adapter (SFT: recipe.adapter; RL: policy_spec.adapter)
        if adapter is not None and isinstance(adapter, Mapping):
            params["adapter_component"] = adapter.get("_component_", "unknown")
            if adapter.get("r") is not None:
                params["adapter_r"] = adapter["r"]

        # Strategy
        dist = settings.config.compute.distributed
        if isinstance(dist, Mapping) and dist.get("strategy"):
            s = dist["strategy"]
            params["strategy"] = s.get("_component_", s) if isinstance(s, Mapping) else s

        mlflow.log_params(params)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow params 로깅 실패: %s", e)


def log_step_metrics(
    optimizers: Mapping[str, torch.optim.Optimizer],
    step: int,
    extra: Mapping[str, float] | None = None,
) -> None:
    """Step 경계(매 optimizer.step 이후)에서 호출. optimizer state + extra 병합.

    ``collect_optimizer_state`` 반환 dict에 ``extra``를 병합해 한 번의
    ``mlflow.log_metrics`` 호출로 내보낸다. ``extra``의 키가 ``learning_rate``
    같은 자동 수집 키와 충돌하면 ``extra``가 우선한다(caller 의도 존중).

    ``mlflow.active_run()`` 없으면 no-op.
    """
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        merged = collect_optimizer_state(optimizers)
        if extra:
            merged.update({k: float(v) for k, v in extra.items()})
        if merged:
            mlflow.log_metrics(merged, step=step)
    except Exception as e:  # noqa: BLE001
        logger.debug("MLflow step metrics 로깅 실패: %s", e, exc_info=True)


def log_epoch_metrics(
    optimizers: Mapping[str, torch.optim.Optimizer],
    epoch: int,
    extra: Mapping[str, float] | None = None,
) -> None:
    """Epoch 경계에서 호출. ``step=epoch``으로 MLflow에 전달.

    MLflow 축 관점에서는 step-축과 epoch-축이 공존하므로, 이 함수는 caller가
    epoch 축으로 별도 축을 쓰겠다고 결정했을 때 사용한다. Trainer 쪽은
    ``log_step_metrics``와 혼용하지 않도록 의식한다(spec §"log_epoch_metrics").

    ``mlflow.active_run()`` 없으면 no-op.
    """
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        merged = collect_optimizer_state(optimizers)
        if extra:
            merged.update({k: float(v) for k, v in extra.items()})
        if merged:
            mlflow.log_metrics(merged, step=epoch)
    except Exception as e:  # noqa: BLE001
        logger.debug("MLflow epoch metrics 로깅 실패: %s", e, exc_info=True)


def log_summary(
    *,
    training_duration_seconds: float,
    total_steps: int,
    stopped_reason: str,
    final_metrics: Mapping[str, float] | None,
    checkpoint_stats: tuple[int, Path | None, str] | None,
    sanitized_config: Mapping[str, Any] | None,
    artifact_dirs: Iterable[tuple[Path, str]] = (),
    extra: Mapping[str, float] | None = None,
) -> None:
    """Run 종료 직전 호출. Trainer·RLTrainer 양쪽의 summary 경로 통합.

    keyword-only 시그니처로 caller의 인자 순서 혼동을 방지한다. 각 인자의
    역할:

    - ``training_duration_seconds`` / ``total_steps``: ``log_metrics``로 기록.
    - ``final_metrics``: 비어 있지 않으면 ``{f"final_{k}": v}`` 형태로
      ``log_metrics`` 추가 호출.
    - ``stopped_reason``: ``set_tag("stopped_reason", ...)``.
    - ``checkpoint_stats``: ``(count, best_path, monitor_hint)`` 튜플. count는
      ``set_tag("checkpoints_saved", ...)``, best_path가 ``None`` 아니면
      ``set_tag("best_checkpoint", str(best_path))``. ``aggregate_checkpoint_stats``
      반환 튜플을 그대로 넘길 수 있도록 설계.
    - ``sanitized_config``: 비어 있지 않으면 ``log_dict(..., "config.json")``.
    - ``artifact_dirs``: ``(src_dir, artifact_path_prefix)`` 반복. 각 튜플에 대해
      ``log_artifacts(str(src_dir), artifact_path=prefix)``.
    - ``extra``: run-summary에 추가 기록할 key→float dict. prefix 없이 그대로
      ``log_metrics``에 병합된다 (예: ``{"peak_memory_gb": 68.4}``). caller 책임
      네이밍 — spec-logging-consistency 원칙 3(slash 네이밍은 group-level에만
      적용)에 어긋나지 않도록 key는 flat string 권장.

    ``mlflow.active_run()`` 없으면 모든 쓰기 생략.
    """
    try:
        import mlflow

        if mlflow.active_run() is None:
            return

        summary_metrics: dict[str, float] = {
            "training_duration_seconds": float(training_duration_seconds),
            "total_steps": float(total_steps),
        }
        if final_metrics:
            summary_metrics.update({f"final_{k}": float(v) for k, v in final_metrics.items()})
        if extra:
            summary_metrics.update({str(k): float(v) for k, v in extra.items()})
        mlflow.log_metrics(summary_metrics)

        mlflow.set_tag("stopped_reason", stopped_reason)

        if checkpoint_stats is not None:
            count, best_path, _monitor_hint = checkpoint_stats
            mlflow.set_tag("checkpoints_saved", str(count))
            if best_path is not None:
                mlflow.set_tag("best_checkpoint", str(best_path))

        if sanitized_config:
            mlflow.log_dict(dict(sanitized_config), "config.json")

        for src_dir, artifact_prefix in artifact_dirs:
            mlflow.log_artifacts(str(src_dir), artifact_path=artifact_prefix)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow summary 로깅 실패: %s", e)
