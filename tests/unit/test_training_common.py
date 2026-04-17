"""`mdp/training/_common.py` 공용 헬퍼 단위 테스트.

`aggregate_checkpoint_stats`는 Trainer / RLTrainer의 `_log_mlflow_summary`와
결과 dict fallback 재집계에서 공용으로 사용된다. spec-trainer-robustness-fixes
cycle 1의 2-1 중복 제거로 도입되었으며, duck typing 규칙(hasattr 기반),
best_path 선정 규칙(규칙 A — 첫 non-empty best), zero-warning용 monitor_hint
조립 규칙을 한 번에 결정한다.

이 테스트는 헬퍼 자체의 계약을 고립 검증한다. `_log_mlflow_summary`가 헬퍼를
호출하는 구조이므로 `test_checkpoint_monitor.py`의 기존
`test_log_mlflow_summary_aggregates_multi_checkpoint_callbacks` 등은 그대로
통과하면서 회귀를 이중 방어한다.
"""

from __future__ import annotations

from pathlib import Path

from mdp.training._common import aggregate_checkpoint_stats


class _DummyCallback:
    """saved_checkpoints 속성 기반 duck-typed 가짜 콜백.

    `monitor`와 `best_models`는 선택적으로 설정할 수 있어 step-only 경로
    (best_models 없음)와 validation-based 경로(best_models 채워짐)를 모두
    시뮬레이션한다.
    """

    def __init__(
        self,
        saved_checkpoints: list[Path] | None = None,
        monitor: str | None = None,
        best_models: list[tuple[float, str]] | None = None,
    ) -> None:
        if saved_checkpoints is not None:
            self.saved_checkpoints = saved_checkpoints
        if monitor is not None:
            self.monitor = monitor
        if best_models is not None:
            self.best_models = best_models


class _OtherCallback:
    """saved_checkpoints를 **갖지 않는** 비관련 콜백.

    EarlyStopping이나 LRScheduler 콜백처럼 체크포인트 저장과 무관한 콜백이
    혼재할 때 집계에서 자연스럽게 제외되는지 검증한다.
    """

    def __init__(self) -> None:
        self.something_else = True


def test_aggregate_checkpoint_stats_empty() -> None:
    """빈 콜백 리스트에서 total=0, best_path=None, monitor_hint=안내 문자열."""
    total, best_path, monitor_hint = aggregate_checkpoint_stats([])
    assert total == 0
    assert best_path is None
    assert monitor_hint == "(no ModelCheckpoint configured)"


def test_aggregate_checkpoint_stats_multi_callbacks(tmp_path) -> None:
    """복수 ModelCheckpoint 인스턴스(Critic + Policy RL 구성)의 합산과 monitor CSV.

    - 첫 콜백: 2개 체크포인트 + best_models 채워짐 → best_path 여기서 채택
    - 둘째 콜백: 1개 체크포인트, best_models 없음 → total에만 기여
    - 셋째 콜백: saved_checkpoints 속성 없음 → 집계에서 제외
    """
    critic_paths = [tmp_path / "critic-1", tmp_path / "critic-2"]
    policy_paths = [tmp_path / "policy-1"]

    callbacks = [
        _DummyCallback(
            saved_checkpoints=critic_paths,
            monitor="val_critic_loss",
            best_models=[
                (0.9, str(critic_paths[0])),  # worst
                (0.7, str(critic_paths[1])),  # best (마지막 요소)
            ],
        ),
        _DummyCallback(
            saved_checkpoints=policy_paths,
            monitor="val_reward",
        ),
        _OtherCallback(),  # saved_checkpoints 없음 → 제외
    ]

    total, best_path, monitor_hint = aggregate_checkpoint_stats(callbacks)

    assert total == 3  # 2 + 1 + 0
    assert best_path == Path(str(critic_paths[1]))  # 첫 콜백의 최신 best
    assert monitor_hint == "val_critic_loss, val_reward"


def test_aggregate_checkpoint_stats_best_path_first_non_empty(tmp_path) -> None:
    """규칙 A: 첫 non-empty best를 가진 콜백이 best_path를 제공한다.

    첫 콜백에 best_models가 비어 있으면(step-only 저장 경로) 건너뛰고,
    두 번째 콜백의 best를 채택해야 한다. 이 규칙은 weighted-ntp TAW recipe처럼
    `every_n_steps`만 쓰는 콜백과 validation 기반 콜백이 공존할 때 필요하다.
    """
    step_only_paths = [tmp_path / "step-ckpt"]
    val_paths = [tmp_path / "val-ckpt"]

    callbacks = [
        _DummyCallback(
            saved_checkpoints=step_only_paths,
            monitor="val_loss",
            # best_models 없음 또는 빈 리스트 — step-only 경로 시뮬레이션
            best_models=[],
        ),
        _DummyCallback(
            saved_checkpoints=val_paths,
            monitor="val_acc",
            best_models=[(0.95, str(val_paths[0]))],
        ),
    ]

    total, best_path, monitor_hint = aggregate_checkpoint_stats(callbacks)

    assert total == 2
    # 첫 콜백의 best_models=[]이므로 두 번째 콜백에서 best_path 채택
    assert best_path == Path(str(val_paths[0]))
    assert monitor_hint == "val_loss, val_acc"
