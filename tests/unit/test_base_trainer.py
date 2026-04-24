"""BaseTrainer 단위 테스트 (spec-training-restructure).

BaseTrainer(ABC) 의 공통 shim 메서드를 구체적인 최소 서브클래스 stub 으로 검증한다.
실제 Trainer / RLTrainer 인스턴스 없이 mixin 계약만 고립 테스트한다.

대상 메서드:
- ``_move_to_device``
- ``_should_stop``
- ``_estimate_total_steps``
- ``_load_checkpoint_state`` (BaseTrainer 기본 no-op, Trainer/RLTrainer 는 override)
- ``_dump_oom_summary`` / ``_maybe_start_memory_history`` / ``_maybe_dump_memory_snapshot``
- ``_fmt_eta``
- ``_log_step_progress``
- ``_log_run_banner``
- ``_algorithm_label`` 기본 반환값
- ``_peak_memory_summary_extra``
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger(__name__)
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import pytest

from mdp.training._base import BaseTrainer


# ──────────────────────────────────────────────────────────────────────────
# Minimal concrete subclass for testing
# ──────────────────────────────────────────────────────────────────────────


class _ConcreteTrainer(BaseTrainer):
    """BaseTrainer 의 모든 abstract method 를 최소 구현한 테스트 전용 서브클래스."""

    def _optimizer_for_progress_log(self):
        return getattr(self, "optimizer", None)

    def _collect_mlflow_params(self) -> None:
        pass  # no-op for tests

    def _checkpoint_state(self) -> dict:
        return {}  # 최소 구현

    def _log_mlflow_summary(self, training_duration, stopped_reason, **kwargs):
        pass  # no-op for tests

    def _fire(self, hook_name: str, **extra_kwargs: Any) -> None:
        """테스트 전용 최소 _fire 구현 — BaseTrainer 가 더 이상 _fire 를 제공하지 않으므로
        각 서브클래스(Trainer, RLTrainer, 이 stub)가 독립 구현을 유지한다."""
        kwargs = dict(extra_kwargs)
        kwargs.setdefault("global_step", self.global_step)
        kwargs.setdefault("strategy", getattr(self, "strategy", None))
        kwargs.setdefault("recipe_dict", self._recipe_dict)
        kwargs.setdefault("scaler", getattr(self, "scaler", None))
        kwargs["trainer"] = self
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    if getattr(cb, "critical", False):
                        raise
                    logger.warning(f"콜백 {type(cb).__name__}.{hook_name} 실패: {e}")


def _make_trainer(**kwargs) -> _ConcreteTrainer:
    """테스트에 필요한 최소 상태를 가진 _ConcreteTrainer 인스턴스를 생성한다."""
    trainer = object.__new__(_ConcreteTrainer)

    # 공통 기본값
    defaults = dict(
        device=torch.device("cpu"),
        global_step=0,
        max_steps=None,
        epochs=1,
        grad_accum_steps=1,
        callbacks=[],
        _stop_requested=False,
        _is_main_process=True,
        strategy=None,
        _recipe_dict={},
        settings=SimpleNamespace(
            recipe=SimpleNamespace(task="test_task"),
            config=SimpleNamespace(mlflow=None),
            max_steps=None,
            epochs=1,
            grad_accum_steps=1,
        ),
        scaler=None,
    )
    defaults.update(kwargs)
    trainer.__dict__.update(defaults)
    return trainer


# ──────────────────────────────────────────────────────────────────────────
# _move_to_device
# ──────────────────────────────────────────────────────────────────────────


class TestMoveToDevice:
    def test_tensor_moved(self):
        trainer = _make_trainer(device=torch.device("cpu"))
        t = torch.tensor([1.0, 2.0])
        batch = {"ids": t, "label": "foo"}
        result = trainer._move_to_device(batch)
        assert result["ids"].device.type == "cpu"
        assert result["label"] == "foo"

    def test_non_tensor_passthrough(self):
        trainer = _make_trainer()
        batch = {"text": "hello", "count": 42}
        result = trainer._move_to_device(batch)
        assert result == batch


# ──────────────────────────────────────────────────────────────────────────
# _should_stop
# ──────────────────────────────────────────────────────────────────────────


class TestShouldStop:
    def test_returns_false_by_default(self):
        trainer = _make_trainer(_stop_requested=False, callbacks=[])
        assert trainer._should_stop() is False

    def test_stop_requested_flag(self):
        trainer = _make_trainer(_stop_requested=True, callbacks=[])
        assert trainer._should_stop() is True

    def test_callback_should_stop(self):
        cb = SimpleNamespace(should_stop=True)
        trainer = _make_trainer(_stop_requested=False, callbacks=[cb])
        assert trainer._should_stop() is True

    def test_callback_should_stop_false(self):
        cb = SimpleNamespace(should_stop=False)
        trainer = _make_trainer(_stop_requested=False, callbacks=[cb])
        assert trainer._should_stop() is False

    def test_callback_without_should_stop_attr(self):
        cb = SimpleNamespace()  # should_stop 없음
        trainer = _make_trainer(_stop_requested=False, callbacks=[cb])
        assert trainer._should_stop() is False


# ──────────────────────────────────────────────────────────────────────────
# _estimate_total_steps
# ──────────────────────────────────────────────────────────────────────────


class TestEstimateTotalSteps:
    def test_max_steps_takes_priority(self):
        loader = MagicMock()
        loader.__len__ = MagicMock(return_value=100)
        trainer = _make_trainer(max_steps=50, epochs=10, grad_accum_steps=1, train_loader=loader)
        assert trainer._estimate_total_steps() == 50

    def test_computed_from_loader(self):
        loader = MagicMock()
        loader.__len__ = MagicMock(return_value=100)
        trainer = _make_trainer(max_steps=None, epochs=3, grad_accum_steps=2, train_loader=loader)
        # 100 // 2 = 50 steps/epoch * 3 epochs = 150
        assert trainer._estimate_total_steps() == 150

    def test_epochs_defaults_to_1_when_none(self):
        loader = MagicMock()
        loader.__len__ = MagicMock(return_value=40)
        trainer = _make_trainer(max_steps=None, epochs=None, grad_accum_steps=1, train_loader=loader)
        assert trainer._estimate_total_steps() == 40


# ──────────────────────────────────────────────────────────────────────────
# _load_checkpoint_state (BaseTrainer 기본 no-op)
# ──────────────────────────────────────────────────────────────────────────


class TestLoadCheckpointState:
    def test_no_op_stub_returns_none(self):
        trainer = _make_trainer()
        result = trainer._load_checkpoint_state({"step": 5})
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# _algorithm_label 기본값
# ──────────────────────────────────────────────────────────────────────────


class TestAlgorithmLabel:
    def test_default_returns_unknown(self):
        trainer = _make_trainer()
        assert trainer._algorithm_label() == "unknown"


# ──────────────────────────────────────────────────────────────────────────
# _fmt_eta (staticmethod)
# ──────────────────────────────────────────────────────────────────────────


class TestFmtEta:
    def test_seconds_format(self):
        assert _ConcreteTrainer._fmt_eta(65) == "01:05"

    def test_hours_format(self):
        assert _ConcreteTrainer._fmt_eta(3725) == "01:02:05"

    def test_negative_returns_placeholder(self):
        assert _ConcreteTrainer._fmt_eta(-1) == "--:--"

    def test_nan_returns_placeholder(self):
        import math
        assert _ConcreteTrainer._fmt_eta(math.nan) == "--:--"


# ──────────────────────────────────────────────────────────────────────────
# _peak_memory_summary_extra
# ──────────────────────────────────────────────────────────────────────────


class TestPeakMemorySummaryExtra:
    def test_returns_none_when_cuda_unavailable(self):
        trainer = _make_trainer()
        with patch("torch.cuda.is_available", return_value=False):
            result = trainer._peak_memory_summary_extra()
        assert result is None

    def test_returns_none_when_peak_bytes_zero(self):
        trainer = _make_trainer()
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.max_memory_allocated", return_value=0):
            result = trainer._peak_memory_summary_extra()
        assert result is None

    def test_returns_dict_with_peak_memory_gb(self):
        trainer = _make_trainer()
        gib = 1024 ** 3 * 4  # 4 GiB in bytes
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.max_memory_allocated", return_value=gib):
            result = trainer._peak_memory_summary_extra()
        assert result is not None
        assert abs(result["peak_memory_gb"] - 4.0) < 1e-6


# ──────────────────────────────────────────────────────────────────────────
# _dump_oom_summary shim
# ──────────────────────────────────────────────────────────────────────────


class TestDumpOomSummary:
    def test_delegates_to_progress_log(self):
        trainer = _make_trainer()
        with patch("mdp.training._base.dump_oom_summary") as mock_dump:
            trainer._dump_oom_summary()
        mock_dump.assert_called_once()


# ──────────────────────────────────────────────────────────────────────────
# _maybe_start_memory_history shim
# ──────────────────────────────────────────────────────────────────────────


class TestMaybeStartMemoryHistory:
    def test_delegates_to_progress_log(self):
        trainer = _make_trainer(_recipe_dict={"monitoring": {}})
        with patch("mdp.training._base.maybe_start_memory_history", return_value=False) as mock_fn:
            result = trainer._maybe_start_memory_history()
        mock_fn.assert_called_once()
        assert result is False


# ──────────────────────────────────────────────────────────────────────────
# _log_mlflow_summary @abstractmethod 계약
# ──────────────────────────────────────────────────────────────────────────


class TestLogMlflowSummaryContract:
    def test_missing_implementation_blocks_instantiation(self):
        """_log_mlflow_summary 를 구현하지 않은 서브클래스는 인스턴스화 불가.

        @abstractmethod 로 선언되어 있으므로 ABC 가 인스턴스화 시점에 검출한다.
        미구현 시 TypeError 가 발생하는 것이 NotImplementedError 런타임 폭발보다
        빠른 정적 보증이다.
        """
        class _IncompleteTrainer(BaseTrainer):
            def _optimizer_for_progress_log(self):
                return None
            def _collect_mlflow_params(self) -> None:
                pass
            def _checkpoint_state(self) -> dict:
                return {}
            # _log_mlflow_summary 미구현

        with pytest.raises(TypeError, match="_log_mlflow_summary"):
            object.__new__(_IncompleteTrainer)  # __new__ 단계에서 ABC 검사

    def test_concrete_implementation_not_abstract(self):
        """_ConcreteTrainer (override 완료) 는 인스턴스화가 가능하다."""
        # _make_trainer 가 정상 동작하면 구현이 완전한 것이다.
        trainer = _make_trainer()
        assert trainer is not None
