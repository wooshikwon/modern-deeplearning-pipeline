"""SIGTERM/SIGINT handling tests for Trainer and RLTrainer (spec-trainer-robustness-fixes U1).

두 축에서 검증한다:

1. **구조 테스트 (AST)** — `train()` 메서드에 signal handler 설치·복원 코드가 존재하는지
   소스 레벨에서 확인한다. 전체 학습 루프를 돌리지 않고도 "코드가 올바르게 짜여 있음"을
   보장하는 값싼 회귀 방어선이다.

2. **동작 테스트 (instance-level)** — 실제 `Trainer`/`RLTrainer` 인스턴스를 최소 구성으로
   만들어 (`object.__new__`로 heavy init을 우회) 다음을 확인한다:
   - `_stop_requested=True`이면 `_should_stop()`이 True 반환
   - handler callable이 signum 인자를 받아 `_stop_requested`를 True로 세움
   - signum에 따라 `_stop_signal_name`이 "SIGTERM" / "SIGINT"로 설정됨

3. **end-to-end thread 테스트** — 별도 thread에서 `train()`을 실행하고
   `os.kill(os.getpid(), signal.SIGTERM/SIGINT)`로 self에 시그널을 보내
   stopped_reason이 `"signal_term"` / `"signal_int"`로 결과 dict에 전달되는지 확인.
   분산 환경 전제(torchrun이 모든 rank에 시그널 전파)는 본 테스트 범위 외.

참고: 본 테스트는 signal 설치 자체가 main thread 전역 상태를 건드리므로,
각 테스트 끝에서 handler를 명시적으로 복원해 다른 테스트로의 leak을 방지한다.
"""

from __future__ import annotations

import ast
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

_TRAINER = Path(__file__).parents[2] / "mdp" / "training" / "trainer.py"
_RL_TRAINER = Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py"


# ─────────────────────────────────────────────────────────────────────────────
# 구조 테스트 (AST) — Trainer
# ─────────────────────────────────────────────────────────────────────────────


def _get_method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == method_name:
                    return ast.get_source_segment(source, method) or ""
    raise AssertionError(f"{class_name}.{method_name}을 찾지 못함: {path}")


class TestTrainerSignalHandlerStructure:
    """Trainer.train()이 signal handler 설치·복원 코드를 포함하는지 AST로 확인."""

    def test_train_installs_sigterm_handler(self) -> None:
        src = _get_method_source(_TRAINER, "Trainer", "train")
        assert "signal.signal(signal.SIGTERM" in src, (
            "Trainer.train()에 signal.signal(signal.SIGTERM, ...) 호출이 없다. "
            "SIGTERM 미처리 시 finally 블록이 실행되지 않아 MLflow zombie run이 발생한다."
        )

    def test_train_installs_sigint_handler(self) -> None:
        src = _get_method_source(_TRAINER, "Trainer", "train")
        assert "signal.signal(signal.SIGINT" in src, (
            "Trainer.train()에 signal.signal(signal.SIGINT, ...) 호출이 없다."
        )

    def test_train_saves_original_handlers_before_install(self) -> None:
        """기존 handler를 저장하는 getsignal 호출이 signal.signal 호출 앞에 있어야 복원 가능."""
        src = _get_method_source(_TRAINER, "Trainer", "train")
        getsignal_sigterm = src.find("signal.getsignal(signal.SIGTERM)")
        set_sigterm = src.find("signal.signal(signal.SIGTERM")
        assert getsignal_sigterm != -1, "Trainer.train()에 SIGTERM 원본 handler 저장이 없다."
        assert getsignal_sigterm < set_sigterm, (
            "Trainer.train()에서 signal.getsignal(SIGTERM)이 signal.signal(SIGTERM)보다 뒤에 있다. "
            "저장 없이 덮어쓰면 복원이 불가능하다."
        )

    def test_train_restores_handlers_after_finally_body(self) -> None:
        """복원(signal.signal(..., original_*))은 finally 블록의 cleanup/on_train_end/summary
        호출이 끝난 뒤에 수행되어야 한다. 본 구조 테스트는 복원 코드의 존재와 위치만
        확인하며, 복원이 summary 로깅보다 뒤에 오는지 검사한다."""
        src = _get_method_source(_TRAINER, "Trainer", "train")
        summary_call = src.find("_log_mlflow_summary(training_duration, stopped_reason)")
        restore_sigterm = src.rfind("signal.signal(signal.SIGTERM, original_sigterm)")
        restore_sigint = src.rfind("signal.signal(signal.SIGINT, original_sigint)")
        assert restore_sigterm != -1, "Trainer.train()에 SIGTERM handler 복원 코드가 없다."
        assert restore_sigint != -1, "Trainer.train()에 SIGINT handler 복원 코드가 없다."
        assert restore_sigterm > summary_call, (
            "Trainer.train()에서 SIGTERM 복원이 _log_mlflow_summary보다 앞에 있다. "
            "finally 내부의 cleanup/on_train_end/summary가 끝난 뒤에 복원해야 한다."
        )

    def test_should_stop_honors_stop_requested_flag(self) -> None:
        """_should_stop()이 self._stop_requested를 OR하는 형태로 확장되어야 한다."""
        src = _get_method_source(_TRAINER, "Trainer", "_should_stop")
        assert "self._stop_requested" in src, (
            "Trainer._should_stop()이 self._stop_requested를 확인하지 않는다. "
            "signal handler가 flag를 세워도 루프가 break되지 않는다."
        )

    def test_signal_handler_preserves_first_signal_guard(self) -> None:
        """Trainer._signal_handler closure가 첫 시그널만 기록하도록 guard되어 있는지
        AST 수준에서 확인. 두 번째 시그널 도착 시 `_stop_signal_name`이 덮어쓰이지
        않아야 한다.
        (spec-trainer-robustness-fixes cycle 1 — 1-3 race 방어 회귀 방지)
        """
        src = _get_method_source(_TRAINER, "Trainer", "train")
        assert "if not self._stop_requested:" in src, (
            "Trainer._signal_handler closure에 `if not self._stop_requested` guard가 없다. "
            "SIGTERM → SIGINT 순차 수신 시 stopped_reason이 오분류된다."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 구조 테스트 (AST) — RLTrainer
# ─────────────────────────────────────────────────────────────────────────────


class TestRLTrainerSignalHandlerStructure:
    """RLTrainer.train()도 동일 패턴을 가져야 한다."""

    def test_train_installs_sigterm_handler(self) -> None:
        src = _get_method_source(_RL_TRAINER, "RLTrainer", "train")
        assert "signal.signal(signal.SIGTERM" in src, (
            "RLTrainer.train()에 signal.signal(SIGTERM, ...) 호출이 없다."
        )

    def test_train_installs_sigint_handler(self) -> None:
        src = _get_method_source(_RL_TRAINER, "RLTrainer", "train")
        assert "signal.signal(signal.SIGINT" in src, (
            "RLTrainer.train()에 signal.signal(SIGINT, ...) 호출이 없다."
        )

    def test_while_loop_condition_checks_stop_requested(self) -> None:
        """while self.global_step < max_steps 루프 조건에 and not self._stop_requested가 있어야
        시그널 수신 시 다음 iteration에서 break된다."""
        src = _get_method_source(_RL_TRAINER, "RLTrainer", "train")
        assert (
            "while self.global_step < max_steps and not self._stop_requested" in src
        ), (
            "RLTrainer.train()의 while 루프에 self._stop_requested 체크가 없다. "
            "signal 수신 후에도 루프가 max_steps까지 계속 돈다."
        )

    def test_stopped_reason_signal_branch_exists(self) -> None:
        """stopped_reason 결정 로직에 signal 수신을 우선 처리하는 분기가 있어야 한다."""
        src = _get_method_source(_RL_TRAINER, "RLTrainer", "train")
        assert "signal_term" in src and "signal_int" in src, (
            "RLTrainer.train()의 stopped_reason 결정 로직에 signal_term/signal_int 분기가 없다."
        )

    def test_should_stop_honors_stop_requested_flag(self) -> None:
        src = _get_method_source(_RL_TRAINER, "RLTrainer", "_should_stop")
        assert "self._stop_requested" in src, (
            "RLTrainer._should_stop()이 self._stop_requested를 확인하지 않는다."
        )

    def test_signal_handler_preserves_first_signal_guard(self) -> None:
        """RLTrainer._signal_handler closure가 첫 시그널만 기록하도록 guard되어 있는지
        AST 수준에서 확인. 두 번째 시그널 도착 시 `_stop_signal_name`이 덮어쓰이지
        않아야 한다 (Trainer와 동일 패턴).
        (spec-trainer-robustness-fixes cycle 1 — 1-3 race 방어 회귀 방지)
        """
        src = _get_method_source(_RL_TRAINER, "RLTrainer", "train")
        # closure 내부에 `if not self._stop_requested:` guard가 존재해야 한다.
        # (첫 시그널만 `_stop_signal_name = sig_name`으로 기록)
        assert "if not self._stop_requested:" in src, (
            "RLTrainer._signal_handler closure에 `if not self._stop_requested` guard가 없다. "
            "SIGTERM → SIGINT 순차 수신 시 stopped_reason이 오분류된다."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 동작 테스트 — Trainer._should_stop / 핸들러 로직 직접 호출
# ─────────────────────────────────────────────────────────────────────────────


def _make_bare_trainer() -> Any:
    """Heavy __init__을 우회한 최소 Trainer stub.

    optimizer/model/strategy 등 학습 루프 의존성 없이 _should_stop과 signal 관련
    필드만 테스트하기 위함. `__init__` 내부의 `create_callbacks`·`create_strategy`·
    `_create_optimizer` 같은 값비싼 구성요소를 전부 건너뛴다.
    """
    from mdp.training.trainer import Trainer

    t = object.__new__(Trainer)
    t.callbacks = []
    t._stop_requested = False
    t._stop_signal_name = None
    return t


def _make_bare_rl_trainer() -> Any:
    from mdp.training.rl_trainer import RLTrainer

    t = object.__new__(RLTrainer)
    t.callbacks = []
    t._stop_requested = False
    t._stop_signal_name = None
    return t


class _DummyCallback:
    """_should_stop()이 callbacks 쪽도 확인하는지 검증용."""

    def __init__(self, should_stop: bool = False) -> None:
        self.should_stop = should_stop


class TestTrainerShouldStopFlag:
    """Trainer._should_stop() 확장 동작."""

    def test_should_stop_false_initially(self) -> None:
        t = _make_bare_trainer()
        assert t._should_stop() is False

    def test_should_stop_true_when_stop_requested(self) -> None:
        t = _make_bare_trainer()
        t._stop_requested = True
        assert t._should_stop() is True

    def test_should_stop_true_when_callback_flags(self) -> None:
        """기존 callback 기반 early-stop 경로가 보존되는지 회귀 방지."""
        t = _make_bare_trainer()
        t.callbacks = [_DummyCallback(should_stop=True)]
        assert t._should_stop() is True


class TestRLTrainerShouldStopFlag:
    def test_should_stop_false_initially(self) -> None:
        t = _make_bare_rl_trainer()
        assert t._should_stop() is False

    def test_should_stop_true_when_stop_requested(self) -> None:
        t = _make_bare_rl_trainer()
        t._stop_requested = True
        assert t._should_stop() is True

    def test_should_stop_true_when_callback_flags(self) -> None:
        t = _make_bare_rl_trainer()
        t.callbacks = [_DummyCallback(should_stop=True)]
        assert t._should_stop() is True


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end — 실제 train() 실행 중 SIGTERM/SIGINT 주입
# ─────────────────────────────────────────────────────────────────────────────


class _TinyModel(nn.Module):
    """단일 Linear + training_step을 가진 초경량 모델."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"]
        return {"logits": self.linear(x)}

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch)["logits"]
        labels = batch["labels"]
        return nn.functional.cross_entropy(logits, labels)


class _SlowLoader:
    """매 배치 요청마다 짧게 sleep하여 signal이 중간에 도달할 여유를 만든다.

    len()은 큰 값으로 고정해 `sys.maxsize`급 loop를 돌린다. enumerate()는
    sleep 후 다음 배치를 yield하므로, 충분한 시간이면 signal handler가 flag를 세우고
    다음 iteration에서 break한다.
    """

    def __init__(self, num_batches: int = 10000, step_sleep: float = 0.05) -> None:
        self.num_batches = num_batches
        self.step_sleep = step_sleep
        self.sampler = None

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            time.sleep(self.step_sleep)
            yield {
                "x": torch.randn(2, 4),
                "labels": torch.randint(0, 2, (2,)),
            }


def _build_tiny_trainer() -> Any:
    """실제 학습 가능한 최소 Trainer.

    signal handler 설치 → SIGTERM 수신 → flag → 다음 step 경계에서 break → finally →
    stopped_reason=="signal_term" 전체 흐름을 검증하기 위한 목적.
    """
    from mdp.training.trainer import Trainer
    from tests.e2e.conftest import make_test_settings

    settings = make_test_settings(
        epochs=1000,  # 사실상 무한, signal로만 종료
        precision="fp32",
        val_check_interval=1.0,
        val_check_unit="epoch",
        name="signal-test",
    )

    model = _TinyModel()
    train_loader = _SlowLoader(num_batches=10000, step_sleep=0.05)

    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=train_loader,
        val_loader=None,
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False
    # MLflow 비활성화 — tracking URI·experiment 없이 nullcontext로 빠진다.
    # (Settings.config.mlflow == None 또는 _start_mlflow_run이 실패하면 nullcontext)
    trainer._is_main_process = False
    return trainer


def _save_handlers() -> tuple[Any, Any]:
    """테스트 leak 방지용: 현재 SIGTERM/SIGINT handler를 기록해 둔다."""
    return signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)


def _restore_handlers(sigterm: Any, sigint: Any) -> None:
    signal.signal(signal.SIGTERM, sigterm)
    signal.signal(signal.SIGINT, sigint)


@pytest.fixture
def preserve_signal_handlers():
    """테스트 전후로 SIGTERM/SIGINT handler를 보존한다."""
    saved = _save_handlers()
    yield
    _restore_handlers(*saved)


class TestTrainerSignalEndToEnd:
    """main thread에서 train()을 실행하고 별도 thread가 self에게 signal을 보내
    stopped_reason이 기대대로 설정되는지 검증한다.

    중요: Python `signal.signal()`은 **main thread에서만** 호출 가능하고,
    signal도 main thread에서만 배달된다. 따라서 실제 production 흐름(main thread가
    train() 호출)을 그대로 재현하려면 test 본체가 main thread에서 train()을 돌려야
    한다. signal 송신은 별도 thread에서 `os.kill(os.getpid(), sig)`로 수행한다.
    """

    @pytest.mark.parametrize(
        ("sig", "expected_reason", "expected_name"),
        [
            (signal.SIGTERM, "signal_term", "SIGTERM"),
            (signal.SIGINT, "signal_int", "SIGINT"),
        ],
    )
    def test_trainer_stopped_reason_reflects_signal(
        self,
        sig: signal.Signals,
        expected_reason: str,
        expected_name: str,
        preserve_signal_handlers: None,
    ) -> None:
        trainer = _build_tiny_trainer()

        # 별도 thread에서 지연 후 signal 송신.
        def _kill_after_delay() -> None:
            time.sleep(1.0)
            os.kill(os.getpid(), sig)

        killer = threading.Thread(target=_kill_after_delay, daemon=True)
        killer.start()

        result = trainer.train()
        killer.join(timeout=2.0)

        assert result["stopped_reason"] == expected_reason, (
            f"stopped_reason이 기대와 다름: got={result['stopped_reason']!r}, "
            f"expected={expected_reason!r}"
        )
        assert trainer._stop_signal_name == expected_name

    def test_signal_handler_preserves_first_signal(
        self, preserve_signal_handlers: None
    ) -> None:
        """첫 시그널만 기록되어야 한다 — SIGTERM → SIGINT 순차 수신 시
        stopped_reason이 "signal_term"으로 유지되는지 검증한다.

        race 시나리오: 외부가 SIGTERM을 보낸 뒤 사용자가 즉시 Ctrl+C를 눌러
        같은 step 경계 전에 SIGINT가 추가로 도달하면, 이전 구현은
        `_stop_signal_name`을 "SIGINT"로 덮어써 MLflow tag가 "signal_int"로
        오분류된다. 수정 후에는 `_stop_requested` 플래그로 guard하여 첫 시그널만
        기록된다.
        (spec-trainer-robustness-fixes cycle 1 — 1-3 race 방어 회귀 방지)
        """
        trainer = _build_tiny_trainer()

        # 두 시그널을 짧은 간격으로 연속 송신. SIGTERM이 먼저, SIGINT가 바로 뒤.
        # step_sleep=0.05이므로 두 신호가 같은 step 경계 전에 도달할 확률이 높다.
        def _kill_sequential() -> None:
            time.sleep(1.0)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.02)
            os.kill(os.getpid(), signal.SIGINT)

        killer = threading.Thread(target=_kill_sequential, daemon=True)
        killer.start()

        result = trainer.train()
        killer.join(timeout=3.0)

        # 첫 시그널(SIGTERM)이 보존되어야 한다.
        assert trainer._stop_signal_name == "SIGTERM", (
            "두 번째 시그널(SIGINT)이 `_stop_signal_name`을 덮어썼다. "
            f"got={trainer._stop_signal_name!r}"
        )
        assert result["stopped_reason"] == "signal_term", (
            "SIGTERM 선행 수신인데 stopped_reason이 'signal_term'이 아니다. "
            f"got={result['stopped_reason']!r}"
        )

    def test_trainer_restores_original_handlers(
        self, preserve_signal_handlers: None
    ) -> None:
        """train() 종료 후 SIGTERM/SIGINT handler가 원래 값으로 복원되는지."""

        # 테스트에서 식별 가능한 고유 sentinel handler 설치
        def _sentinel(signum: int, frame: Any) -> None:  # pragma: no cover
            pass

        signal.signal(signal.SIGTERM, _sentinel)
        signal.signal(signal.SIGINT, _sentinel)

        trainer = _build_tiny_trainer()

        def _kill_after_delay() -> None:
            time.sleep(1.0)
            os.kill(os.getpid(), signal.SIGTERM)

        killer = threading.Thread(target=_kill_after_delay, daemon=True)
        killer.start()
        trainer.train()
        killer.join(timeout=2.0)

        # 복원 확인
        assert signal.getsignal(signal.SIGTERM) is _sentinel, (
            "Trainer.train()이 원래 SIGTERM handler를 복원하지 않았다."
        )
        assert signal.getsignal(signal.SIGINT) is _sentinel, (
            "Trainer.train()이 원래 SIGINT handler를 복원하지 않았다."
        )
