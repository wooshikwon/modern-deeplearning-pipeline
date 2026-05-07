"""Duration semantics documentation + info log tests (spec-trainer-robustness-fixes U3).

본 모듈은 C-3 "epochs + max_steps 공존 시 semantics 미문서화" 문제가 실제로
해결되었는지 두 축에서 확인한다.

1. **스키마 문서화** — `TrainingSpec` 클래스 docstring과 `epochs`/`max_steps`
   Pydantic Field description이 "먼저 도달한 조건에서 종료" semantics를 명시하는지.
   Pydantic의 `model_fields["name"].description`으로 빈 문자열이 아닌 설명이
   존재함을 확인하고, 클래스 docstring에 두 필드 이름이 모두 등장하는지 검사한다.

2. **진입 로그** — `Trainer.train()` / `RLTrainer.train()`이 epochs·max_steps 둘 다
   set인 경우 진입 직후 INFO 로그를 1회 찍는지, 하나만 set인 경우에는
   찍지 않는지(노이즈 방지) 확인한다. 전체 학습 루프를 돌리지 않고, 별도 thread로
   SIGTERM을 보내 train()이 진입부 로그만 남기고 즉시 종료되도록 유도한다.

signal handler는 main thread만 설치 가능하므로 test 본체가 main thread에서
train()을 호출해야 한다. 기존 `test_trainer_signal_handling.py`의 패턴을 따른다.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


# ─────────────────────────────────────────────────────────────────────────────
# 스키마 문서화 테스트
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingSpecDocumentation:
    """TrainingSpec이 duration semantics를 제대로 문서화하고 있는지 확인."""

    def test_trainingspec_epochs_description_documented(self) -> None:
        """epochs 필드의 Pydantic description이 비어있지 않고 semantics를 언급해야 한다."""
        from mdp.settings.schema import TrainingSpec

        field = TrainingSpec.model_fields["epochs"]
        desc = field.description or ""
        assert desc.strip(), "TrainingSpec.epochs에 description이 없다."
        assert "먼저 도달" in desc or "whichever" in desc.lower(), (
            f"TrainingSpec.epochs.description에 'early-hit' semantics가 없음: {desc!r}"
        )

    def test_trainingspec_max_steps_description_documented(self) -> None:
        """max_steps 필드의 Pydantic description도 동일한 기준을 만족해야 한다."""
        from mdp.settings.schema import TrainingSpec

        field = TrainingSpec.model_fields["max_steps"]
        desc = field.description or ""
        assert desc.strip(), "TrainingSpec.max_steps에 description이 없다."
        assert "먼저 도달" in desc or "whichever" in desc.lower(), (
            f"TrainingSpec.max_steps.description에 'early-hit' semantics가 없음: {desc!r}"
        )

    def test_trainingspec_class_docstring_mentions_both(self) -> None:
        """클래스 docstring에 epochs / max_steps 두 필드 이름이 모두 등장하고
        "먼저 도달" semantics가 언급되어야 한다."""
        from mdp.settings.schema import TrainingSpec

        doc = TrainingSpec.__doc__ or ""
        assert "epochs" in doc, "TrainingSpec docstring에 'epochs'가 없다."
        assert "max_steps" in doc, "TrainingSpec docstring에 'max_steps'가 없다."
        assert "먼저 도달" in doc or "whichever" in doc.lower(), (
            "TrainingSpec docstring에 early-hit semantics 설명이 없다."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 진입 로그 테스트 — Trainer / RLTrainer에 걸쳐 공통으로 쓰이는 fixture·헬퍼
# ─────────────────────────────────────────────────────────────────────────────


class _TinyModel(nn.Module):
    """단일 Linear + forward-native loss를 가진 초경량 모델."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"]
        logits = self.linear(x)
        return {"logits": logits, "loss": nn.functional.cross_entropy(logits, batch["labels"])}


class _SlowLoader:
    """매 배치 요청마다 짧게 sleep하여 signal이 중간에 도달할 여유를 만든다.

    `test_trainer_signal_handling.py`의 동명 헬퍼와 동일한 역할.
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


def _build_tiny_trainer(epochs: float | None, max_steps: int | None) -> Any:
    """진입부 로그 검증용 최소 Trainer.

    `epochs`/`max_steps` 조합만 조정하며, SIGTERM으로 즉시 종료시켜 진입 직후 로그만
    포착한다. `test_trainer_signal_handling.py`의 `_build_tiny_trainer`와 유사하나
    duration 필드 제어를 위해 별도 구현.
    """
    from mdp.training.trainer import Trainer
    from tests.e2e.conftest import make_test_settings

    settings = make_test_settings(
        epochs=epochs if epochs is not None else 1000,
        max_steps=max_steps,
        precision="fp32",
        val_check_interval=1.0,
        val_check_unit="epoch",
        name="duration-log-test",
    )
    # epochs=None을 테스트하려면 make_test_settings를 거친 뒤 직접 설정해야 한다.
    # make_test_settings의 epochs 파라미터 기본 타입이 int라 None을 그대로 전달할 수 없기
    # 때문이다. Recipe/Training 재빌드 대신 training 객체만 교체한다.
    if epochs is None:
        settings.recipe.training.epochs = None

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
    trainer._is_main_process = False
    return trainer


def _save_handlers() -> tuple[Any, Any]:
    return signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)


def _restore_handlers(sigterm: Any, sigint: Any) -> None:
    signal.signal(signal.SIGTERM, sigterm)
    signal.signal(signal.SIGINT, sigint)


@pytest.fixture
def preserve_signal_handlers():
    """테스트 전후로 SIGTERM/SIGINT handler를 보존한다(다른 테스트로의 leak 방지)."""
    saved = _save_handlers()
    yield
    _restore_handlers(*saved)


def _kill_shortly(delay: float = 0.3) -> threading.Thread:
    """짧은 지연 후 self에게 SIGTERM을 보내는 daemon thread를 반환한다."""

    def _kill() -> None:
        time.sleep(delay)
        os.kill(os.getpid(), signal.SIGTERM)

    t = threading.Thread(target=_kill, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Trainer 진입 로그 테스트
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainerDurationLog:
    """Trainer.train()이 duration semantics 로그를 조건에 맞게 출력하는지 확인."""

    def test_trainer_logs_when_both_duration_fields_set(
        self,
        caplog: pytest.LogCaptureFixture,
        preserve_signal_handlers: None,
    ) -> None:
        """epochs·max_steps 둘 다 set이면 INFO 로그가 1회 찍혀야 한다."""
        trainer = _build_tiny_trainer(epochs=5.0, max_steps=20)

        killer = _kill_shortly(delay=0.3)
        with caplog.at_level(logging.INFO, logger="mdp.training.trainer"):
            trainer.train()
        killer.join(timeout=2.0)

        duration_logs = [
            r for r in caplog.records
            if "모두 지정됨" in r.message and "먼저 도달" in r.message
        ]
        assert len(duration_logs) >= 1, (
            "둘 다 set인 경우 duration 로그가 최소 1회 찍혀야 한다. "
            f"records={[r.message for r in caplog.records]!r}"
        )
        msg = duration_logs[0].getMessage()
        assert "epochs" in msg and "max_steps" in msg, (
            f"duration 로그에 필드 이름이 누락: {msg!r}"
        )

    def test_trainer_does_not_log_when_only_epochs_set(
        self,
        caplog: pytest.LogCaptureFixture,
        preserve_signal_handlers: None,
    ) -> None:
        """epochs만 set이면 의도가 명확하므로 duration 로그를 찍지 않아야 한다."""
        trainer = _build_tiny_trainer(epochs=3.0, max_steps=None)

        killer = _kill_shortly(delay=0.3)
        with caplog.at_level(logging.INFO, logger="mdp.training.trainer"):
            trainer.train()
        killer.join(timeout=2.0)

        duration_logs = [
            r for r in caplog.records
            if "모두 지정됨" in r.message
        ]
        assert not duration_logs, (
            "epochs만 set인 경우 duration 로그가 찍히면 안 된다(노이즈 방지). "
            f"got={[r.message for r in duration_logs]!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RLTrainer 진입 로그 테스트 — 대칭 검증
# ─────────────────────────────────────────────────────────────────────────────


class TestRLTrainerDurationLog:
    """RLTrainer.train()도 동일 조건에서 duration 로그를 찍는지 구조적으로 확인.

    RLTrainer의 전체 설정은 SFT Trainer보다 무겁고(여러 모델 + optimizer 사전 구성 필요)
    단일 테스트에서 재현하기엔 비용이 크다. 대신 소스 AST에서 진입부 로그 블록의 존재와
    조건식을 검사하여 Trainer와 대칭으로 구현되었음을 보장한다.
    """

    _RL_TRAINER = Path(__file__).parents[2] / "mdp" / "training" / "rl_trainer.py"

    def _get_train_source(self) -> str:
        import ast

        source = self._RL_TRAINER.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "RLTrainer":
                for method in node.body:
                    if (
                        isinstance(method, ast.FunctionDef)
                        and method.name == "train"
                    ):
                        return ast.get_source_segment(source, method) or ""
        raise AssertionError("RLTrainer.train()을 찾지 못함")

    def test_rl_trainer_logs_when_both_duration_fields_set(self) -> None:
        """RLTrainer.train()에 epochs·max_steps 둘 다 set일 때만 찍는 INFO 로그가
        signal handler 설치부 이후에 존재해야 한다."""
        src = self._get_train_source()

        # 조건식: 둘 다 not None일 때만 로그 (노이즈 방지)
        assert "self.epochs is not None and self.max_steps is not None" in src, (
            "RLTrainer.train()에 epochs·max_steps 둘 다 set인지 검사하는 조건이 없다."
        )
        # 로그 메시지 핵심 키워드
        assert "모두 지정됨" in src and "먼저 도달" in src, (
            "RLTrainer.train()의 duration 로그 메시지가 Trainer와 대칭이 아니다."
        )
        # signal handler 설치(signal.signal(signal.SIGINT, ...))가 로그보다 앞에 있어야 한다
        sigint_install = src.find("signal.signal(signal.SIGINT, _signal_handler)")
        duration_log = src.find("모두 지정됨")
        assert sigint_install != -1, "RLTrainer.train()에 SIGINT handler 설치가 없다."
        assert duration_log > sigint_install, (
            "RLTrainer.train()의 duration 로그가 signal handler 설치보다 앞에 있다. "
            "spec §3.3 요구: handler 설치 뒤, 학습 루프 시작 전."
        )
