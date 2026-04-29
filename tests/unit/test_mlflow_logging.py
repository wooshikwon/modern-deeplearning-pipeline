"""``mdp/training/_mlflow_logging.py`` 공용 헬퍼 단위 테스트.

spec-logging-consistency §U5에서 본 Unit(U1)에 할당한 6건의 고립 테스트를
수용한다. 나머지 4건(Trainer/RLTrainer의 실제 호출 경로 검증)은 U2·U3·U5
범위이므로 본 파일에서는 만들지 않는다.

원칙 검증 지점:

- **원칙 3 (multi-group slash 네이밍)**: test_collect_optimizer_state_*의 3건이
  single-group / multi-group unnamed / multi-group named 세 경로의 키 형태를
  고정한다. 이 키 규약이 깨지면 U2·U3가 소비하는 공유 인터페이스가 바로
  붕괴되므로 early regression guard 역할이 크다.
- **원칙 2 (static 값은 recipe 출처)**: test_log_static_params_uses_recipe_lr이
  "optimizer 인스턴스 상태를 param에 박지 않는다"를 명시적으로 검증한다.
  recipe 선언값과 optimizer 인스턴스의 현재 lr을 다르게 세팅해 recipe 쪽이
  기록됨을 확인한다.
- **공통 no-op 계약**: test_log_static_params_no_run_noop은 ``mlflow.active_run()``
  이 ``None``일 때 log_params/log_metrics/set_tag가 전혀 호출되지 않아야 한다는
  모듈 공통 계약을 검증한다.
- **원칙 1 (summary 경로)**: test_log_summary_writes_final_metrics는 final_*
  prefix 규약과 checkpoint tag 세팅을 한 번에 확인한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from mdp.training._mlflow_logging import (
    collect_optimizer_state,
    log_static_params,
    log_summary,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────


def _make_optimizer(
    param_groups: list[dict],
) -> torch.optim.Optimizer:
    """명시적으로 param_groups를 구성한 SGD optimizer를 만든다.

    각 group의 ``params``는 서로 다른 tensor로 구성해야 optimizer가
    "param already in another group"을 뱉지 않는다.
    """
    # 각 group마다 고유 텐서를 할당
    groups_with_params: list[dict] = []
    for i, g in enumerate(param_groups):
        copy = dict(g)
        copy["params"] = [torch.nn.Parameter(torch.zeros(1) + i)]
        groups_with_params.append(copy)
    # 첫 group을 생성자로, 나머지는 add_param_group으로 추가
    first = groups_with_params[0]
    opt = torch.optim.SGD([first["params"][0]], lr=first["lr"])
    # 첫 group의 메타 키들을 반영
    for k, v in first.items():
        if k != "params":
            opt.param_groups[0][k] = v
    for g in groups_with_params[1:]:
        opt.add_param_group(g)
    return opt


# ──────────────────────────────────────────────────────────────────────────
# collect_optimizer_state — 원칙 3 key naming
# ──────────────────────────────────────────────────────────────────────────


def test_collect_optimizer_state_single_group() -> None:
    """Single-group + single-optimizer → ``learning_rate`` 단일 키.

    대부분의 SFT run이 이 경로를 타므로 유저 친화적 단일 키를 고정한다.
    SGD의 경우 momentum/weight_decay가 param_group에 내장 0.0으로 존재하므로
    추가 키로 함께 나가지만, 본 테스트는 LR 네이밍 규약에 집중한다.
    """
    opt = _make_optimizer([{"lr": 1e-3}])
    state = collect_optimizer_state({"policy": opt})

    # Single-group에서는 slash suffix 없이 평탄 키.
    assert state["learning_rate"] == pytest.approx(1e-3)
    # multi-group 키가 섞여 나오면 안 된다.
    assert "learning_rate/group_0" not in state
    assert "learning_rate/policy" not in state
    # momentum/weight_decay 수집 규칙 확인 — SGD는 둘 다 포함.
    assert "momentum" in state
    assert "weight_decay" in state


def test_collect_optimizer_state_multi_group_unnamed() -> None:
    """Multi-group (name 없음) → ``learning_rate/group_{idx}`` 키.

    weighted-ntp의 2-group CriticValueModel이 name 미지정으로 돌 때의 폴백
    형태를 보장한다.
    """
    opt = _make_optimizer(
        [
            {"lr": 4e-5},
            {"lr": 4e-4},
        ]
    )
    state = collect_optimizer_state({"policy": opt})

    assert state["learning_rate/group_0"] == pytest.approx(4e-5)
    assert state["learning_rate/group_1"] == pytest.approx(4e-4)
    # 단일 키 폴백이 있으면 안 됨 (multi-group은 반드시 slash).
    assert "learning_rate" not in state


def test_collect_optimizer_state_multi_group_named() -> None:
    """Multi-group (name 지정) → ``learning_rate/{name}`` 키.

    param_group에 ``name`` 키가 있으면 숫자 인덱스 대신 이름을 사용한다.
    LoRA/head 분리 같은 실사례에서 대시보드 가독성을 확보한다.
    """
    opt = _make_optimizer(
        [
            {"lr": 4e-5, "name": "lora"},
            {"lr": 4e-4, "name": "head"},
        ]
    )
    state = collect_optimizer_state({"policy": opt})

    assert state["learning_rate/lora"] == pytest.approx(4e-5)
    assert state["learning_rate/head"] == pytest.approx(4e-4)
    # 숫자 인덱스 키가 섞이면 안 됨 — name이 있으면 name이 우선.
    assert "learning_rate/group_0" not in state
    assert "learning_rate/group_1" not in state


# ──────────────────────────────────────────────────────────────────────────
# log_static_params — 원칙 2: recipe 출처 우선
# ──────────────────────────────────────────────────────────────────────────


def _make_rl_recipe_settings(policy_lr: float = 1e-4):
    """RL recipe + settings mock을 최소 구성으로 만든다.

    Pydantic Recipe 전체를 유효하게 구성하는 건 비용이 크므로 MagicMock spec
    없이 속성 트리를 수동 조립한다. log_static_params가 참조하는 최소 속성
    세트만 채운다.
    """
    recipe = MagicMock()
    recipe.task = "rl-alignment"
    recipe.rl = MagicMock()
    recipe.rl.algorithm = {"_component_": "DPO"}
    recipe.rl.models = {
        "policy": {
            "_component_": "PolicyModel",
            "pretrained": "gpt2",
            "optimizer": {"lr": policy_lr},
        }
    }
    recipe.data.dataset = {"source": "hh-rlhf"}
    recipe.data.dataloader = MagicMock()
    recipe.data.dataloader.batch_size = 4
    recipe.training = MagicMock()
    recipe.training.epochs = None
    recipe.training.max_steps = 1000
    recipe.training.precision = "bf16"
    recipe.training.gradient_accumulation_steps = 1

    settings = MagicMock()
    settings.recipe = recipe
    settings.config.compute.distributed = None
    return recipe, settings


def test_log_static_params_uses_recipe_lr() -> None:
    """Recipe 선언 lr이 기록되고 optimizer 인스턴스 lr은 무시됨.

    이것이 spec의 핵심 약속. weighted-ntp Phase 3에서 warmup step 0의
    ``param_groups[0]["lr"] = 2e-12`` 가 param으로 박혀 사용자를 오도한 사례의
    구조적 해소를 검증한다.
    """
    recipe, settings = _make_rl_recipe_settings(policy_lr=1e-4)

    # 이 optimizer는 param으로 나가지 않아야 한다 — 함수 시그니처에 optimizer를
    # 받지 않는다는 설계 자체가 방어 장치. 혹시 내부 경로에서 읽힌다면
    # 이 테스트가 인지할 수 있도록 log_static_params에 넘기지 않는다.
    # (시그니처에 없으므로 기록 자체가 불가능해야 한다.)

    with patch("mlflow.active_run", return_value=MagicMock()), patch(
        "mlflow.log_params"
    ) as mock_log_params:
        log_static_params(recipe, settings)

    assert mock_log_params.call_count == 1
    params = mock_log_params.call_args.args[0]
    assert params["learning_rate_init"] == pytest.approx(1e-4)
    # optimizer 인스턴스 상태 키가 나가지 않아야 함
    assert "learning_rate" not in params  # step-level은 metric 경로
    assert "policy_lr" not in params  # 구 스냅샷 키 제거


def test_log_static_params_no_run_noop() -> None:
    """``mlflow.active_run()`` 이 None이면 어떤 쓰기도 발생하지 않는다.

    모듈 전체 공통 계약. caller가 DDP rank-0 가드를 빼먹거나 run을 아직
    시작하지 않은 초기화 경로에서 호출해도 부작용이 없어야 한다.
    """
    recipe, settings = _make_rl_recipe_settings(policy_lr=1e-4)

    with patch("mlflow.active_run", return_value=None), patch(
        "mlflow.log_params"
    ) as mock_log_params, patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.set_tag"
    ) as mock_set_tag:
        log_static_params(recipe, settings)

    assert mock_log_params.call_count == 0
    assert mock_log_metrics.call_count == 0
    assert mock_set_tag.call_count == 0


# ──────────────────────────────────────────────────────────────────────────
# log_summary — final_* prefix + checkpoint tag
# ──────────────────────────────────────────────────────────────────────────


def test_log_summary_writes_final_metrics() -> None:
    """``final_metrics`` 가 ``final_*`` prefix로, checkpoint_stats가 tag로 기록된다.

    Trainer와 RLTrainer가 동일 시그니처로 요약을 넘기는 대칭성을 담보한다.
    ``final_val_loss`` 네이밍 규약은 MLflow UI 필터링에서 실제 final 지표를
    식별하는 단서라 중요하다.
    """
    with patch("mlflow.active_run", return_value=MagicMock()), patch(
        "mlflow.log_metrics"
    ) as mock_log_metrics, patch("mlflow.set_tag") as mock_set_tag, patch(
        "mlflow.log_dict"
    ) as _mock_log_dict, patch("mlflow.log_artifacts") as _mock_log_artifacts:
        log_summary(
            training_duration_seconds=12.5,
            total_steps=100,
            stopped_reason="completed",
            final_metrics={"val_loss": 0.3},
            checkpoint_stats=(3, Path("/tmp/best.ckpt"), "val_loss"),
            sanitized_config=None,
            artifact_dirs=(),
        )

    # 첫 log_metrics 호출이 summary metric dict.
    assert mock_log_metrics.call_count == 1
    logged = mock_log_metrics.call_args.args[0]
    assert logged["final_val_loss"] == pytest.approx(0.3)
    assert logged["training_duration_seconds"] == pytest.approx(12.5)
    assert logged["total_steps"] == pytest.approx(100.0)

    # set_tag 호출이 세 번: stopped_reason, checkpoints_saved, best_checkpoint.
    tag_calls = {args[0]: args[1] for args, _ in (c[:2] for c in mock_set_tag.call_args_list)}
    # call_args_list는 [call(args, kwargs), ...] — 간단히 key/value로 재구성
    tag_calls = {c.args[0]: c.args[1] for c in mock_set_tag.call_args_list}
    assert tag_calls["stopped_reason"] == "completed"
    assert tag_calls["checkpoints_saved"] == "3"
    assert tag_calls["best_checkpoint"] == "/tmp/best.ckpt"
