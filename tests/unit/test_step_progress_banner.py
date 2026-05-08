"""Step progress + Run banner 단위 테스트 (spec-system-logging-cleanup §U4).

RLTrainer / Trainer 의 다음 helper 를 고립 검증한다:

- ``_fmt_eta(seconds)`` — HH:MM:SS 또는 MM:SS 포맷, 예외 입력은 ``--:--``.
- ``_log_step_progress(loss, grad_norm, start_time, max_steps)`` — rank-0 한 줄.
- ``_log_run_banner("start" | "end", extra=...)`` — rank-0 only, is_json_mode 가드.

각 helper 를 실제 Trainer 인스턴스 없이 ``SimpleNamespace`` 기반 stub 으로
바운드 호출해 fixture 복잡도를 낮춘다 (실제 Trainer 조립은 device / strategy /
dataloader 의존이 무거움). Helper 들의 시그니처는 ``self`` 만 읽고 쓰므로
``Trainer.<method>.__get__(stub)`` 경로가 그대로 동작한다.
"""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mdp.training.rl_trainer import RLTrainer
from mdp.training.trainer import Trainer


_RL_LOGGER = "mdp.training.rl_trainer"
_SFT_LOGGER = "mdp.training.trainer"
_BASE_LOGGER = "mdp.training._base"


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture
def caplog_both(caplog):
    """RL / SFT / BaseTrainer logger 모두 DEBUG 이상 수집.

    spec-training-restructure U2 이후 _log_step_progress / _log_run_banner
    는 BaseTrainer 로 이동됐으므로 mdp.training._base 로거도 수집한다.
    """
    caplog.set_level(logging.DEBUG, logger=_RL_LOGGER)
    caplog.set_level(logging.DEBUG, logger=_SFT_LOGGER)
    caplog.set_level(logging.DEBUG, logger=_BASE_LOGGER)
    return caplog


def _make_rl_stub(
    *,
    is_main: bool = True,
    global_step: int = 0,
    max_steps: int | None = 10,
    epochs: float | None = 1.0,
    lr: float = 1e-4,
    recipe_dict: dict | None = None,
    algorithm_name: str = "DPO",
    strategy_name: str = "NoStrategy",
) -> SimpleNamespace:
    """RLTrainer helper 호출에 필요한 최소 stub."""

    # Optimizer: param_groups 에 단일 group, lr 만 필요.
    optimizer = SimpleNamespace(param_groups=[{"lr": lr}])

    # Recipe 서브 구조 — 배너가 읽는 필드만.
    training = SimpleNamespace(precision="bf16")
    dataloader = SimpleNamespace(batch_size=32)
    data = SimpleNamespace(dataloader=dataloader)
    recipe = SimpleNamespace(training=training, data=data, task="causal_lm")

    # Config.mlflow.experiment_name
    mlflow_cfg = SimpleNamespace(experiment_name="weighted-ntp")
    config = SimpleNamespace(mlflow=mlflow_cfg)
    settings = SimpleNamespace(recipe=recipe, config=config)

    # algorithm 클래스 이름 stub — type(x).__name__ 로 읽힘.
    algo_class = type(algorithm_name, (), {})
    algorithm = algo_class()

    # strategy: None 아니면 클래스 이름 stub
    if strategy_name == "NoStrategy":
        strategy = None
    else:
        strategy = type(strategy_name, (), {})()

    stub = SimpleNamespace(
        _is_main_process=is_main,
        global_step=global_step,
        max_steps=max_steps,
        epochs=epochs,
        optimizers={"policy": optimizer},
        algorithm=algorithm,
        strategy=strategy,
        settings=settings,
        _recipe_dict=recipe_dict or {"monitoring": {"log_every_n_steps": 5}},
        # helper 들은 self._fmt_eta / self._peak_memory_summary_extra 를 호출.
        # staticmethod / bound method 이므로 함수 참조를 stub 에 꽂아준다.
        _fmt_eta=RLTrainer._fmt_eta,
        _peak_memory_summary_extra=lambda: None,  # CUDA 미가용 환경 시뮬레이션
    )
    # spec-training-restructure U2: _algorithm_label 은 BaseTrainer._log_run_banner 가
    # getattr 로 조회한다. SimpleNamespace 는 메서드를 갖지 않으므로 lambda 로 주입.
    stub._algorithm_label = lambda: type(algorithm).__name__
    # _optimizer_for_progress_log 도 BaseTrainer._log_step_progress 가 getattr 로 조회.
    stub._optimizer_for_progress_log = lambda: optimizer
    return stub


def _make_sft_stub(
    *,
    is_main: bool = True,
    global_step: int = 0,
    max_steps: int | None = 10,
    epochs: float | None = 1.0,
    lr: float = 1e-4,
    recipe_dict: dict | None = None,
    strategy_name: str = "NoStrategy",
) -> SimpleNamespace:
    """Trainer helper 호출에 필요한 최소 stub."""
    optimizer = SimpleNamespace(param_groups=[{"lr": lr}])

    training = SimpleNamespace(precision="bf16")
    dataloader = SimpleNamespace(batch_size=32)
    data = SimpleNamespace(dataloader=dataloader)
    recipe = SimpleNamespace(training=training, data=data, task="causal_lm")

    mlflow_cfg = SimpleNamespace(experiment_name="sft-exp")
    config = SimpleNamespace(mlflow=mlflow_cfg)
    settings = SimpleNamespace(recipe=recipe, config=config)

    if strategy_name == "NoStrategy":
        strategy = None
    else:
        strategy = type(strategy_name, (), {})()

    stub = SimpleNamespace(
        _is_main_process=is_main,
        global_step=global_step,
        max_steps=max_steps,
        epochs=epochs,
        optimizer=optimizer,
        strategy=strategy,
        settings=settings,
        _recipe_dict=recipe_dict or {"monitoring": {"log_every_n_steps": 5}},
        _fmt_eta=Trainer._fmt_eta,
        _peak_memory_summary_extra=lambda: None,
    )
    # spec-training-restructure U2: BaseTrainer._log_run_banner / _log_step_progress 가
    # getattr 로 조회하는 추상 메서드 구현체를 lambda 로 주입.
    stub._algorithm_label = lambda: getattr(recipe, "task", "sft")
    stub._optimizer_for_progress_log = lambda: optimizer
    return stub


# ──────────────────────────────────────────────────────────────────────────
# _fmt_eta
# ──────────────────────────────────────────────────────────────────────────


class TestFmtEta:
    """HH:MM:SS (>1h) / MM:SS (<1h) / 비정상 입력 --:--"""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "00:00"),
            (59, "00:59"),
            (65, "01:05"),
            (3599, "59:59"),
            (3600, "01:00:00"),
            (3725, "01:02:05"),
        ],
    )
    def test_valid_eta_formats(self, seconds: int, expected: str) -> None:
        assert RLTrainer._fmt_eta(seconds) == expected
        assert Trainer._fmt_eta(seconds) == expected

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), None])
    def test_invalid_eta_returns_placeholder(self, bad) -> None:
        assert RLTrainer._fmt_eta(bad) == "--:--"
        assert Trainer._fmt_eta(bad) == "--:--"


# ──────────────────────────────────────────────────────────────────────────
# _log_step_progress
# ──────────────────────────────────────────────────────────────────────────


class TestLogStepProgress:
    """rank-0 가드 + log_every_n_steps 간격 + 로그 포맷."""

    def test_format_contains_all_fields(self, caplog_both) -> None:
        stub = _make_rl_stub(global_step=5, max_steps=10, lr=1.94e-4)
        RLTrainer._log_step_progress(
            stub,
            loss=0.7523,
            grad_norm=2.13,
            start_time=time.time() - 10.0,  # 10초 경과
            max_steps=10,
        )
        msgs = [r.getMessage() for r in caplog_both.records]
        assert any("[step 5/10 | 50.0%]" in m for m in msgs), msgs
        combined = "\n".join(msgs)
        assert "loss=0.7523" in combined
        assert "lr=1.94e-04" in combined
        assert "grad_norm=2.13" in combined
        assert "throughput=" in combined
        assert "ETA=" in combined

    def test_grad_norm_none_renders_dashes(self, caplog_both) -> None:
        stub = _make_rl_stub(global_step=10, max_steps=10, lr=5e-5)
        RLTrainer._log_step_progress(
            stub, loss=0.4, grad_norm=None,
            start_time=time.time() - 1.0, max_steps=10,
        )
        msgs = "\n".join(r.getMessage() for r in caplog_both.records)
        assert "grad_norm=--" in msgs

    def test_sft_trainer_symmetric_output(self, caplog_both) -> None:
        """Trainer 쪽 _log_step_progress 도 동일 포맷을 낸다."""
        stub = _make_sft_stub(global_step=4, max_steps=20, lr=2e-4)
        Trainer._log_step_progress(
            stub, loss=1.23, grad_norm=0.5,
            start_time=time.time() - 2.0, max_steps=20,
        )
        msgs = [r.getMessage() for r in caplog_both.records]
        assert any("[step 4/20 | 20.0%]" in m for m in msgs), msgs


# ──────────────────────────────────────────────────────────────────────────
# log_every_n_steps 간격 발화 통합 검증
# ──────────────────────────────────────────────────────────────────────────


class TestLogEveryNStepsGating:
    """log_every_n_steps=5 조건에서 step 5, 10, ... 에만 progress 라인이 출력.

    `_log_step_progress` 자체는 caller 가 타이밍을 결정하므로 여기서는 wrapper
    루프를 손으로 돌려 "step % n == 0 또는 final" 규칙 준수를 확인한다.
    """

    def test_step_5_and_final_emit(self, caplog_both) -> None:
        """max_steps=10, log_every_n=5 → 출력되는 step 값은 {5, 10} 두 번."""
        n = 5
        max_steps = 10
        emitted_steps: list[int] = []

        start = time.time()
        for step in range(1, max_steps + 1):
            stub = _make_rl_stub(global_step=step, max_steps=max_steps)
            # caller 의 gating 로직을 그대로 재현
            if step > 0 and (step % n == 0 or step >= max_steps):
                RLTrainer._log_step_progress(
                    stub, loss=0.5, grad_norm=0.1,
                    start_time=start, max_steps=max_steps,
                )
                emitted_steps.append(step)

        assert emitted_steps == [5, 10], emitted_steps
        step_lines = [
            r.getMessage()
            for r in caplog_both.records
            if r.getMessage().startswith("[step ")
        ]
        assert len(step_lines) == 2, step_lines


# ──────────────────────────────────────────────────────────────────────────
# _log_run_banner
# ──────────────────────────────────────────────────────────────────────────


class TestLogRunBannerRLTrainer:
    """Start / End 배너 출력 + rank / JSON 가드."""

    def test_start_banner_contains_core_fields(self, caplog_both, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        stub = _make_rl_stub(
            max_steps=10, epochs=1.0,
            algorithm_name="WeightedNTP", strategy_name="DDPStrategy",
        )
        # is_json_mode 는 default False 이므로 그대로 진행.
        RLTrainer._log_run_banner(stub, "start", extra={"run_id": "abc123"})

        combined = "\n".join(r.getMessage() for r in caplog_both.records)
        assert "MDP Run Started" in combined
        assert "algorithm=WeightedNTP" in combined
        assert "strategy=DDPStrategy" in combined
        assert "precision=bf16" in combined
        assert "max_steps=10" in combined
        assert "world_size=4" in combined
        assert "bs_per_rank=32" in combined
        assert "experiment=weighted-ntp" in combined
        assert "run_id=abc123" in combined
        # 3 줄 메시지 + 2 구분선 = 5 줄 logger.info 호출
        banner_records = [
            r for r in caplog_both.records
            if "MDP Run" in r.getMessage() or r.getMessage().startswith("=" * 10)
        ]
        assert len(banner_records) >= 3

    def test_end_banner_contains_core_fields(self, caplog_both) -> None:
        stub = _make_rl_stub(global_step=10, max_steps=10)
        RLTrainer._log_run_banner(
            stub, "end",
            extra={
                "stopped_reason": "completed",
                "duration": 213.6,
                "checkpoints_saved": 0,
                "final_loss": 0.7519,
                "total_steps": 10,
            },
        )
        combined = "\n".join(r.getMessage() for r in caplog_both.records)
        assert "MDP Run Ended" in combined
        assert "stopped_reason=completed" in combined
        assert "duration=213.6s" in combined
        assert "checkpoints_saved=0" in combined
        assert "final_loss=0.7519" in combined
        assert "total_steps=10" in combined

    def test_banner_suppressed_in_json_mode(self, caplog_both) -> None:
        stub = _make_rl_stub()
        with patch("mdp.cli.output.is_json_mode", return_value=True):
            RLTrainer._log_run_banner(stub, "start", extra={"run_id": "abc"})
            RLTrainer._log_run_banner(stub, "end",
                                      extra={"stopped_reason": "completed",
                                             "duration": 1.0,
                                             "checkpoints_saved": 0,
                                             "final_loss": 0.1,
                                             "total_steps": 1})
        msgs = [r.getMessage() for r in caplog_both.records]
        assert not any("MDP Run" in m for m in msgs), msgs

    def test_banner_suppressed_on_non_rank0(self, caplog_both) -> None:
        stub = _make_rl_stub(is_main=False)
        RLTrainer._log_run_banner(stub, "start", extra={"run_id": "abc"})
        RLTrainer._log_run_banner(
            stub, "end",
            extra={"stopped_reason": "completed",
                   "duration": 1.0,
                   "checkpoints_saved": 0,
                   "final_loss": 0.1,
                   "total_steps": 1},
        )
        msgs = [r.getMessage() for r in caplog_both.records]
        assert not any("MDP Run" in m for m in msgs), msgs


class TestLogRunBannerTrainer:
    """SFT Trainer 쪽 대칭 배너."""

    def test_start_banner_uses_task_name(self, caplog_both, monkeypatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "2")
        stub = _make_sft_stub(strategy_name="FSDPStrategy")
        Trainer._log_run_banner(stub, "start", extra={"run_id": "xyz"})

        combined = "\n".join(r.getMessage() for r in caplog_both.records)
        # SFT Trainer 는 algorithm 이 없으므로 task 를 algorithm 슬롯에 쓴다.
        assert "algorithm=causal_lm" in combined
        assert "strategy=FSDPStrategy" in combined
        assert "world_size=2" in combined
        assert "run_id=xyz" in combined

    def test_end_banner(self, caplog_both) -> None:
        stub = _make_sft_stub(global_step=100)
        Trainer._log_run_banner(
            stub, "end",
            extra={"stopped_reason": "max_steps_reached",
                   "duration": 42.0,
                   "checkpoints_saved": 3,
                   "final_loss": 0.5,
                   "total_steps": 100},
        )
        combined = "\n".join(r.getMessage() for r in caplog_both.records)
        assert "MDP Run Ended" in combined
        assert "stopped_reason=max_steps_reached" in combined
        assert "checkpoints_saved=3" in combined


# ──────────────────────────────────────────────────────────────────────────
# rank-0 가드: step progress 도 caller 에서 is_main_process 로 걸리는지
# (실제 train loop 와 동일 조건 시뮬레이션)
# ──────────────────────────────────────────────────────────────────────────


class TestRank0GuardForProgress:
    """train loop 은 `if self._is_main_process: _log_step_progress(...)` 로 감싼다.

    본 테스트는 rank-0 이 아닌 context 에서 helper 가 직접 호출되지 않는다는
    계약을 rl_trainer.py / trainer.py 소스에서 텍스트 수준으로 확인한다.
    """

    def test_rl_trainer_guards_progress_with_main_process(self) -> None:
        from pathlib import Path
        src = (Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py").read_text()
        # train loop 안의 `_log_step_progress(...)` 호출 사이트 주변에 rank-0 가드가 있는지 확인.
        # helper 정의 자체(_log_step_progress 시그니처) 는 제외하기 위해, 호출 패턴만 찾는다.
        # 호출 지점 앞 ~1500 자 범위에 `_is_main_process` 가드가 있어야 한다
        # (직전의 `if self._is_main_process:` 블록이 step-progress 와 mlflow 기록을
        #  동시에 감싸기 때문에 필요한 윈도 크기).
        call_idx = src.find("self._log_step_progress(")
        assert call_idx != -1, "rl_trainer.py 에서 self._log_step_progress 호출이 없다"
        prefix = src[max(0, call_idx - 2000):call_idx]
        assert "_is_main_process" in prefix, (
            "rl_trainer.py 의 step-progress 호출이 rank-0 가드 안에 있지 않다."
        )

    def test_sft_trainer_guards_progress_with_main_process(self) -> None:
        from pathlib import Path
        src = (Path(__file__).parents[2] / "mdp" / "training" / "trainer.py").read_text()
        call_idx = src.find("self._log_step_progress(")
        assert call_idx != -1, "trainer.py 에서 self._log_step_progress 호출이 없다"
        prefix = src[max(0, call_idx - 2000):call_idx]
        assert "_is_main_process" in prefix, (
            "trainer.py 의 step-progress 호출이 rank-0 가드 안에 있지 않다."
        )
