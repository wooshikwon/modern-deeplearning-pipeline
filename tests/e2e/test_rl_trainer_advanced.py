"""RLTrainer 고급 통합 경로 테스트 — spec-lr-warmup-configurable U4.

Trainer-측 warmup factor e2e는 `test_trainer_advanced.py`가 담당한다. 본 파일은
RLTrainer 경유 경로에 특화된 두 시나리오를 커버한다:

- `test_rl_trainer_warmup_per_model_factors`: `recipe.rl.models.policy.scheduler`와
  `recipe.rl.models.critic.scheduler`가 서로 다른 `warmup_start_factor`를 가질 때,
  RLTrainer가 per-model로 독립된 LinearLR 인스턴스를 빌드하는지 검증.
- `test_trainer_rl_trainer_symmetric_factors`: 동일 scheduler config dict를 Trainer와
  RLTrainer(단일 trainable 모델)에 각각 전달했을 때 resulting LinearLR의 핵심 인자
  (`start_factor`·`end_factor`)가 bit-identical. 공용 헬퍼(`mdp/training/_schedulers.py`)
  경유로 두 trainer 경로가 같은 지점을 찍어야 한다.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    RLSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from mdp.training.rl_trainer import RLTrainer
from mdp.training.trainer import Trainer
from tests.e2e.conftest import make_test_settings
from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


# ── 테스트용 RL 모델 (test_rl_integration.TinyLM과 최소 duplicate) ──


class _TinyLM(nn.Module):
    """RL 경로 테스트용 최소 LM — forward + generate만 제공."""

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.config = type("Config", (), {"pad_token_id": 0})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):  # noqa: D401
        return type("Out", (), {"logits": self.head(self.embed(input_ids))})()


def _pref_batches(n: int, bs: int, seq: int = 8, vocab: int = 32) -> list[dict]:
    return [
        {
            "chosen_input_ids": torch.randint(1, vocab, (bs, seq)),
            "chosen_attention_mask": torch.ones(bs, seq, dtype=torch.long),
            "chosen_labels": torch.randint(1, vocab, (bs, seq)),
            "rejected_input_ids": torch.randint(1, vocab, (bs, seq)),
            "rejected_attention_mask": torch.ones(bs, seq, dtype=torch.long),
            "rejected_labels": torch.randint(1, vocab, (bs, seq)),
        }
        for _ in range(n)
    ]


def _linear_warmup_from_sequential(sched) -> LinearLR:
    """SequentialLR → 내부 LinearLR 추출. PyTorch private 경로 최소화 위해 타입만 확인."""
    assert isinstance(sched, SequentialLR), (
        f"warmup 활성 시 SequentialLR이어야 합니다. got={type(sched).__name__}"
    )
    linear = sched._schedulers[0]
    assert isinstance(linear, LinearLR)
    return linear


def _rl_settings_with_per_model_factors(
    *,
    policy_start: float,
    critic_start: float,
    warmup_ratio: float = 0.5,
    max_steps: int = 3,
) -> Settings:
    """policy·critic 각 trainable 모델이 독립된 scheduler config를 갖는 RLSettings.

    custom algorithm(SimpleWeightedCELoss)을 사용하여 causal 배치 경로를 탄다 —
    DPO는 preference 배치 전용이라 critic을 trainable로 둘 수 없고, 둘 다 trainable
    하려면 optimizer가 붙은 구성이 필요하다.
    """
    model_component = "tests.e2e.test_rl_trainer_advanced._TinyLM"
    recipe = Recipe(
        name="rl-warmup-per-model",
        task="text_generation",
        rl=RLSpec(
            algorithm={
                "_component_": "tests.e2e.test_rl_custom_algorithm.SimpleWeightedCELoss",
                "weight_scale": 1.0,
            },
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                    "scheduler": {
                        "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                        "T_max": 2,
                        "warmup_ratio": warmup_ratio,
                        "warmup_start_factor": policy_start,
                        "warmup_end_factor": 1.0,
                    },
                },
                "critic": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 5e-4},
                    "scheduler": {
                        "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                        "T_max": 2,
                        "warmup_ratio": warmup_ratio,
                        "warmup_start_factor": critic_start,
                        "warmup_end_factor": 1.0,
                    },
                },
            },
        ),
        data=DataSpec(
            dataset={
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": "/tmp/fake",
                "split": "train",
            },
            collator={
                "_component_": "mdp.data.collators.PreferenceCollator",
                "tokenizer": "gpt2",
                "max_length": 2048,
            },
        ),
        training=TrainingSpec(max_steps=max_steps),
        metadata=MetadataSpec(author="test", description="warmup per-model test"),
    )
    return Settings(recipe=recipe, config=Config())


def test_rl_trainer_warmup_per_model_factors() -> None:
    """per-model scheduler config가 서로 다른 warmup factor를 독립적으로 반영.

    policy와 critic에 서로 다른 `warmup_start_factor`를 지정하면 `rl_trainer.
    schedulers["policy"]`와 `rl_trainer.schedulers["critic"]`는 각각 독립된
    LinearLR 인스턴스를 갖고, 두 인자가 섞이지 않아야 한다.
    """
    settings = _rl_settings_with_per_model_factors(
        policy_start=0.1,
        critic_start=0.05,
    )
    models = {"policy": _TinyLM(), "critic": _TinyLM()}
    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(_pref_batches(5, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    policy_linear = _linear_warmup_from_sequential(trainer.schedulers["policy"])
    critic_linear = _linear_warmup_from_sequential(trainer.schedulers["critic"])

    # per-model 독립성: 두 LinearLR 객체가 다른 인스턴스이고 값이 섞이지 않음.
    assert policy_linear is not critic_linear
    assert math.isclose(policy_linear.start_factor, 0.1, rel_tol=1e-12)
    assert math.isclose(critic_linear.start_factor, 0.05, rel_tol=1e-12)
    # end_factor는 두 모델 동일 지정했지만, 그 역시 독립적으로 전파됐는지 확인.
    assert policy_linear.end_factor == 1.0
    assert critic_linear.end_factor == 1.0

    # optimizer 바인딩도 per-model 독립이어야 한다. LinearLR이 올바른 optimizer를
    # 참조해야 scheduler.step()이 해당 모델 param_groups에만 영향을 준다.
    assert policy_linear.optimizer is trainer.optimizers["policy"]
    assert critic_linear.optimizer is trainer.optimizers["critic"]


def test_trainer_rl_trainer_symmetric_factors() -> None:
    """같은 scheduler config를 Trainer와 RLTrainer(policy)에 각각 전달하면 LinearLR 인자가 일치.

    공용 헬퍼(`mdp/training/_schedulers.create_scheduler_with_warmup`) 경유로 두 trainer가
    동일 지점을 찍어야 대칭성이 구조적으로 보장된다. total_iters는 양쪽의
    `_estimate_total_steps()` 결과에 따라 달라질 수 있으므로, **factor 쌍**(= Recipe에서
    직접 제어 가능한 값)의 일치에 방점을 둔다.
    """
    sched_config = {
        "_component_": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 2,
        "warmup_ratio": 0.5,
        "warmup_start_factor": 0.1,
        "warmup_end_factor": 1.0,
    }

    # ── Trainer 쪽 ──
    trainer_settings = make_test_settings(
        epochs=2,
        scheduler=dict(sched_config),  # 사본 전달 — 실제 파이프라인과 동일
    )
    trainer = Trainer(
        settings=trainer_settings,
        model=TinyVisionModel(num_classes=2, hidden_dim=16),
        train_loader=ListDataLoader(make_vision_batches(4, 4, 2, 8)),
    )
    trainer_linear = _linear_warmup_from_sequential(trainer.scheduler)

    # ── RLTrainer 쪽 (single-trainable: policy만) ──
    model_component = "tests.e2e.test_rl_trainer_advanced._TinyLM"
    recipe = Recipe(
        name="rl-symmetric",
        task="text_generation",
        rl=RLSpec(
            algorithm={
                "_component_": "tests.e2e.test_rl_custom_algorithm.SimpleWeightedCELoss",
                "weight_scale": 1.0,
            },
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                    "scheduler": dict(sched_config),
                },
                "critic": {"_component_": model_component},  # frozen, scheduler 없음
            },
        ),
        data=DataSpec(
            dataset={
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": "/tmp/fake",
                "split": "train",
            },
            collator={
                "_component_": "mdp.data.collators.PreferenceCollator",
                "tokenizer": "gpt2",
                "max_length": 2048,
            },
        ),
        training=TrainingSpec(max_steps=3),
        metadata=MetadataSpec(author="test", description="warmup symmetric test"),
    )
    rl_settings = Settings(recipe=recipe, config=Config())
    rl_trainer = RLTrainer(
        settings=rl_settings,
        models={"policy": _TinyLM(), "critic": _TinyLM()},
        train_loader=ListDataLoader(_pref_batches(5, 4)),
    )
    rl_linear = _linear_warmup_from_sequential(rl_trainer.schedulers["policy"])

    # Factor 쌍: 사용자가 Recipe에서 직접 지정한 값. bit-identical해야 한다.
    assert trainer_linear.start_factor == rl_linear.start_factor
    assert trainer_linear.end_factor == rl_linear.end_factor
    # 기본값 유지 사이드 확인 — Recipe에 명시 안 된 값이 우회 경로로 오염되지 않는지.
    assert math.isclose(trainer_linear.start_factor, 0.1, rel_tol=1e-12)
    assert math.isclose(trainer_linear.end_factor, 1.0, rel_tol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# spec-logging-consistency (U5): RLTrainer 측 MLflow 로깅 규약 3종 e2e 검증.
#
# 본 블록의 세 테스트는 spec §U5의 남은 핵심 약속을 회귀 방어한다:
#   1. test_rl_trainer_logs_multi_group_lr — weighted-ntp CriticValueModel의
#      2-group optimizer(LoRA lr + head lr) 사례가 `learning_rate/group_0` ·
#      `learning_rate/group_1`(또는 name이 있으면 `/lora` · `/head`) 두 키로
#      MLflow metric에 기록됨을 보장. 과거 `param_groups[0]` 하드코딩이 두 번째
#      그룹을 **어디에도 기록하지 않던 결함**의 직접 회귀 테스트.
#   2. test_rl_trainer_final_metrics_symmetric_to_trainer — U3에서 보완한
#      `final_metrics=self.last_metrics` 전달로 Trainer↔RLTrainer `final_*` 블록
#      대칭이 복구됐음을 회귀 방어. `self.last_metrics`에 값을 주입한 상태로
#      `_log_mlflow_summary`를 호출해 `final_val_loss` 키가 metric에 나타나는지
#      직접 확인.
#   3. test_no_policy_lr_param — 하위 호환 삭제의 명시적 방어. 새 run의 params에
#      `policy_lr` 키가 **없음**을 assert하고, 대신 `learning_rate_init`이 recipe
#      선언값과 일치함을 확인한다.
#
# 모든 테스트는 MLflow 자체는 `unittest.mock.patch`로 캡처(네트워크 없음).
# ─────────────────────────────────────────────────────────────────────


def _dpo_rl_trainer_for_logging(
    *,
    policy_lr: float = 1e-3,
    max_steps: int = 2,
    n_batches: int = 5,
):
    """MLflow 로깅 e2e 테스트용 최소 DPO RLTrainer를 구성한다.

    `test_rl_integration._dpo_settings`의 축약형. 단일 trainable policy +
    frozen reference로 구성하여 `self.optimizers`가 `{"policy": ...}` 단일 엔트리.
    multi-group 테스트에서는 호출 직후 `trainer.optimizers["policy"]`를 수동으로
    교체한다.
    """
    model_component = "tests.e2e.test_rl_trainer_advanced._TinyLM"
    recipe = Recipe(
        name="rl-logging-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": policy_lr},
                },
                "reference": {"_component_": model_component},
            },
        ),
        data=DataSpec(
            dataset={
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": "/tmp/fake",
                "split": "train",
            },
            collator={
                "_component_": "mdp.data.collators.PreferenceCollator",
                "tokenizer": "gpt2",
                "max_length": 2048,
            },
        ),
        training=TrainingSpec(max_steps=max_steps),
        metadata=MetadataSpec(author="test", description="mlflow logging test"),
    )
    config = Config()
    config.job.resume = "disabled"
    settings = Settings(recipe=recipe, config=config)

    from mdp.training.rl_trainer import RLTrainer

    trainer = RLTrainer(
        settings=settings,
        models={"policy": _TinyLM(), "reference": _TinyLM()},
        train_loader=ListDataLoader(_pref_batches(n_batches, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    return trainer


def test_rl_trainer_logs_multi_group_lr() -> None:
    """2-group optimizer → `learning_rate/group_0`·`learning_rate/group_1` 두 키가 기록.

    weighted-ntp `CriticValueModel` 패턴의 직접 회귀 방어. RLTrainer 초기화 후
    `trainer.optimizers["policy"]`를 LoRA-style 2-group SGD로 교체하여 multi-group
    경로가 MLflow metric에 그대로 흐르는지 검증한다. Recipe 기반 optimizer 생성은
    모든 모델 parameter에 동일 LR을 적용하므로 multi-group을 만들 수 없어,
    교체 후 검증이 실사례 구성에 가장 가깝다.
    """
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch

    trainer = _dpo_rl_trainer_for_logging(policy_lr=1e-4, max_steps=2, n_batches=5)

    # policy optimizer를 2-group SGD로 교체. LoRA(4e-5) + head(4e-4) 구성 모사.
    policy_model = trainer.trainable["policy"]
    embed_params = list(policy_model.embed.parameters())
    head_params = list(policy_model.head.parameters())
    # embed와 head param을 각각 독립 group으로 묶어 2-group 구성을 강제.
    multi_opt = torch.optim.SGD(
        [
            {"params": embed_params, "lr": 4e-5},
            {"params": head_params, "lr": 4e-4},
        ]
    )
    trainer.optimizers["policy"] = multi_opt
    # 해당 모델의 scheduler도 교체하거나 비워야 일관. 기존 schedulers 엔트리가
    # 이전 optimizer를 참조하므로 삭제.
    trainer.schedulers.pop("policy", None)
    trainer.scheduler_intervals.pop("policy", None)

    with patch.object(trainer, "_start_mlflow_run", return_value=nullcontext()), patch(
        "mlflow.active_run", return_value=MagicMock()
    ), patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.log_params"
    ), patch("mlflow.set_tag"), patch("mlflow.log_dict"), patch(
        "mlflow.log_artifacts"
    ):
        trainer.train()

    # ── 검증: step-level log_metrics 호출 중 learning_rate/group_0·/group_1 모두 발견.
    step_calls = [
        c for c in mock_log_metrics.call_args_list
        if c.kwargs.get("step") is not None
    ]
    assert len(step_calls) > 0, "step-level log_metrics 호출이 없습니다."

    seen_keys: set[str] = set()
    for call in step_calls:
        seen_keys.update(call.args[0].keys())

    assert "learning_rate/group_0" in seen_keys, (
        f"multi-group LR 키 `learning_rate/group_0`이 기록되지 않았습니다. "
        f"seen={sorted(seen_keys)}"
    )
    assert "learning_rate/group_1" in seen_keys, (
        f"multi-group LR 키 `learning_rate/group_1`이 기록되지 않았습니다. "
        f"seen={sorted(seen_keys)}"
    )
    # single-group 폴백 키는 공존하면 안 된다 — multi-group이면 반드시 slash.
    assert "learning_rate" not in seen_keys, (
        f"multi-group 경로에서 single-group 폴백 키가 섞였습니다: seen={sorted(seen_keys)}"
    )

    # 값 검증: group_0은 4e-5, group_1은 4e-4 (SGD + scheduler 없음이므로 불변).
    first_call_dict = step_calls[0].args[0]
    assert first_call_dict["learning_rate/group_0"] == pytest.approx(4e-5, rel=1e-9)
    assert first_call_dict["learning_rate/group_1"] == pytest.approx(4e-4, rel=1e-9)


def test_rl_trainer_final_metrics_symmetric_to_trainer() -> None:
    """RLTrainer run 종료 시 `final_*` 블록이 MLflow metric에 기록된다.

    U3가 복구한 Trainer↔RLTrainer 대칭성의 회귀 방어. `self.last_metrics`에
    `{"val_loss": 0.3}`을 주입한 상태로 `_log_mlflow_summary`를 호출하면
    `log_summary`가 `final_val_loss=0.3`을 metric에 포함시킨다. 과거 RLTrainer는
    이 블록 전체가 빠져 있어 Trainer 비대칭이었다.
    """
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch

    trainer = _dpo_rl_trainer_for_logging(policy_lr=1e-3, max_steps=1, n_batches=3)

    # last_metrics를 주입. `_log_mlflow_summary`가 이를 `final_metrics`로 전달.
    trainer.last_metrics = {"val_loss": 0.3}

    with patch.object(trainer, "_start_mlflow_run", return_value=nullcontext()), patch(
        "mlflow.active_run", return_value=MagicMock()
    ), patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.log_params"
    ), patch("mlflow.set_tag"), patch("mlflow.log_dict"), patch(
        "mlflow.log_artifacts"
    ):
        trainer.train()

    # ── 검증: log_metrics 호출 중 `final_val_loss` 키가 나온 호출 존재.
    summary_calls = [
        c for c in mock_log_metrics.call_args_list
        if "final_val_loss" in c.args[0]
    ]
    assert len(summary_calls) == 1, (
        f"`final_val_loss`를 포함한 log_metrics 호출이 정확히 1회여야 합니다. "
        f"찾은 수: {len(summary_calls)}. 전체 호출: {mock_log_metrics.call_args_list}"
    )
    summary_dict = summary_calls[0].args[0]
    assert summary_dict["final_val_loss"] == pytest.approx(0.3)
    # Trainer와 동일하게 summary 호출에 training_duration_seconds·total_steps도 함께.
    assert "training_duration_seconds" in summary_dict
    assert "total_steps" in summary_dict
    # step 인자 없음(summary는 축을 두지 않음) — `log_summary`의 계약.
    assert summary_calls[0].kwargs.get("step") is None


def test_no_policy_lr_param() -> None:
    """새 run의 params에 `policy_lr` 키가 존재하지 않고, `learning_rate_init`이
    recipe 선언값과 일치함을 확인한다.

    하위 호환 삭제(원칙 2 위반 스냅샷 제거)의 명시적 회귀 방어. 과거
    `rl_trainer._log_mlflow_params`가 `optimizer.param_groups[0]["lr"]` 스냅샷을
    `policy_lr` 이름으로 박던 경로가 완전히 사라졌고, 대신 recipe에 선언된
    `policy.optimizer.lr`이 `learning_rate_init`으로 기록된다.
    """
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch

    declared_lr = 2e-4  # recipe 선언값 — warmup step 0 스냅샷과 다른 값이어야 변별력
    trainer = _dpo_rl_trainer_for_logging(
        policy_lr=declared_lr, max_steps=1, n_batches=3
    )

    with patch.object(trainer, "_start_mlflow_run", return_value=nullcontext()), patch(
        "mlflow.active_run", return_value=MagicMock()
    ), patch("mlflow.log_params") as mock_log_params, patch(
        "mlflow.log_metrics"
    ), patch("mlflow.set_tag"), patch("mlflow.log_dict"), patch(
        "mlflow.log_artifacts"
    ):
        trainer.train()

    # log_params는 run 시작 시 1회 호출되어야 한다.
    assert mock_log_params.call_count >= 1, (
        f"`log_static_params` 경유 log_params가 호출되지 않았습니다. "
        f"call_count={mock_log_params.call_count}"
    )

    # 첫 번째 호출의 params dict를 검사.
    params = mock_log_params.call_args_list[0].args[0]

    # 구 키 부재 확인.
    assert "policy_lr" not in params, (
        f"구 호환 키 `policy_lr`이 여전히 기록됩니다: params={params}"
    )
    assert "learning_rate" not in params, (
        f"optimizer 인스턴스 스냅샷 키 `learning_rate`가 param에 박혔습니다. "
        f"이는 원칙 2 위반: params={params}"
    )

    # 새 키 존재 + 값 일치.
    assert "learning_rate_init" in params, (
        f"`learning_rate_init` 키가 없습니다. params={params}"
    )
    assert params["learning_rate_init"] == pytest.approx(declared_lr, rel=1e-9), (
        f"`learning_rate_init`이 recipe 선언값({declared_lr})과 다릅니다: "
        f"{params['learning_rate_init']}"
    )


# ─────────────────────────────────────────────────────────────────────
# spec-logging-consistency fix cycle 1 (1-1): RLTrainer step-level 로깅
# 타이밍이 grad_accum 경계에만 발화하는지 회귀 방어.
#
# 과거 결함: `log_step_metrics` 호출이 모든 batch마다 발화하여, `grad_accum_steps > 1`
# 환경에서 같은 `self.global_step` 값으로 여러 entry가 누적되었다(MLflow UI 곡선 왜곡).
# 수정: 호출을 `if batch_idx % self.grad_accum_steps == 0:` 블록 내부로 이동.
# 본 테스트는 `grad_accum_steps=2`로 DPO 경로를 돌리고 step-level `log_metrics`
# 호출의 step 인자가 **중복 없이 단조 증가**함을 검증한다. Trainer(`trainer.py` L556)가
# 이미 경계 내부에서 발화하던 것과 완전 대칭(원칙 4).
# ─────────────────────────────────────────────────────────────────────


def test_rl_trainer_step_logging_at_grad_accum_boundary_only() -> None:
    """`grad_accum_steps > 1`에서 step-level log_metrics 호출이 경계에서만 1회 발화.

    `grad_accum_steps=2`, `max_steps=2`, 4 batch로 오프라인 DPO 경로를 돌린다.
    global_step은 boundary(batch_idx=1, 3)에서 2회 증가(1, 2)하고, step-level
    `log_step_metrics`도 그에 맞춰 **정확히 2회** 발화해야 한다. 과거 결함 경로에서는
    4 batch × 1회 = 4회 발화하며 step=1이 중복 기록됐다.
    """
    from contextlib import nullcontext
    from unittest.mock import MagicMock, patch

    # grad_accum_steps=2, max_steps=2 → 총 4 batch를 소비한 뒤 종료.
    model_component = "tests.e2e.test_rl_trainer_advanced._TinyLM"
    recipe = Recipe(
        name="rl-step-log-timing",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": {
                    "_component_": model_component,
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "reference": {"_component_": model_component},
            },
        ),
        data=DataSpec(
            dataset={
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": "/tmp/fake",
                "split": "train",
            },
            collator={
                "_component_": "mdp.data.collators.PreferenceCollator",
                "tokenizer": "gpt2",
                "max_length": 2048,
            },
        ),
        training=TrainingSpec(max_steps=2, gradient_accumulation_steps=2),
        metadata=MetadataSpec(author="test", description="step-log timing"),
    )
    config = Config()
    config.job.resume = "disabled"
    settings = Settings(recipe=recipe, config=config)

    trainer = RLTrainer(
        settings=settings,
        models={"policy": _TinyLM(), "reference": _TinyLM()},
        train_loader=ListDataLoader(_pref_batches(4, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    # Sanity — RLTrainer가 gradient_accumulation_steps을 실제로 수용했는지 확인.
    assert trainer.grad_accum_steps == 2

    with patch.object(trainer, "_start_mlflow_run", return_value=nullcontext()), patch(
        "mlflow.active_run", return_value=MagicMock()
    ), patch("mlflow.log_metrics") as mock_log_metrics, patch(
        "mlflow.log_params"
    ), patch("mlflow.set_tag"), patch("mlflow.log_dict"), patch(
        "mlflow.log_artifacts"
    ):
        trainer.train()

    # step-level 호출만 필터 — `log_summary`는 step 인자 없이 호출되므로 분리된다.
    step_calls = [
        c for c in mock_log_metrics.call_args_list
        if c.kwargs.get("step") is not None
    ]

    # step-level 호출이 최소 1회는 있어야 한다(기본 sanity).
    assert len(step_calls) >= 1, (
        f"step-level log_metrics 호출이 없습니다. 전체: {mock_log_metrics.call_args_list}"
    )

    # 각 호출의 step 인자 추출 → 중복 없음 + 단조 증가.
    # 과거 결함에서는 step=1이 4 batch 중 여러 번 기록돼 중복이 발생했다.
    step_indices = [c.kwargs["step"] for c in step_calls]
    assert len(step_indices) == len(set(step_indices)), (
        f"step-level 로깅에 중복 step이 있습니다(grad_accum 경계 밖 발화 재발 의심). "
        f"steps={step_indices}"
    )

    # 추가 의미 검증: 호출 수가 optimizer step 수(= global_step 증가 횟수)와 정확히
    # 일치해야 한다. max_steps=2에 도달할 때까지 boundary가 2번 발화하므로 2회.
    assert len(step_calls) == 2, (
        f"grad_accum_steps=2, max_steps=2에서 step-level 호출은 정확히 2회여야 합니다. "
        f"got={len(step_calls)}, steps={step_indices}"
    )
