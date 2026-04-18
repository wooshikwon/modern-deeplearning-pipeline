"""``monitoring.memory_history`` recipe option 단위 테스트 (§U5).

검증 항목:

1. ``memory_history=False`` (기본값) → ``torch.cuda.memory._record_memory_history``
   가 호출되지 않는다.
2. ``memory_history=True`` → start 시 ``_record_memory_history``, dump 시
   ``_dump_snapshot`` 둘 다 mock 으로 호출 관찰.
3. RANK != "0" 인 프로세스는 활성화하지 않는다 (multi-rank 중복 dump 회피).
4. ``_record_memory_history`` 예외 → warning + 학습 계속.
5. ``_dump_snapshot`` 예외 → warning + 학습 계속.

테스트는 SimpleNamespace stub 에 helper 를 bound method 로 호출해 Trainer
조립 의존을 제거한다 (``_maybe_start_memory_history`` / ``_maybe_dump_memory_snapshot``
은 ``self`` 의 ``_recipe_dict`` 하나만 읽으므로 충분).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from mdp.training.rl_trainer import RLTrainer
from mdp.training.trainer import Trainer


_RL_LOGGER = "mdp.training.rl_trainer"
_SFT_LOGGER = "mdp.training.trainer"


@pytest.fixture
def caplog_both(caplog):
    caplog.set_level(logging.DEBUG, logger=_RL_LOGGER)
    caplog.set_level(logging.DEBUG, logger=_SFT_LOGGER)
    return caplog


def _make_stub(memory_history: bool) -> SimpleNamespace:
    return SimpleNamespace(
        _recipe_dict={
            "monitoring": {
                "log_every_n_steps": 10,
                "memory_history": memory_history,
                "verbose": False,
            }
        }
    )


# ──────────────────────────────────────────────────────────────────────────
# _maybe_start_memory_history — 비활성 경로
# ──────────────────────────────────────────────────────────────────────────


class TestMemoryHistoryDisabled:
    """memory_history=False (기본값) 에서는 _record_memory_history 호출 없음."""

    def test_memory_history_disabled_does_not_call_record(self, monkeypatch):
        """기본값 False → start 호출 자체 없음."""
        monkeypatch.setenv("RANK", "0")
        stub = _make_stub(memory_history=False)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda.memory, "_record_memory_history") as mock_record:
            active_rl = RLTrainer._maybe_start_memory_history(stub)
            active_sft = Trainer._maybe_start_memory_history(stub)

        assert active_rl is False
        assert active_sft is False
        mock_record.assert_not_called()

    def test_dump_skipped_when_inactive(self, monkeypatch):
        """active=False 이면 _dump_snapshot 호출 없음."""
        stub = _make_stub(memory_history=True)  # config 와 무관 — active flag 만 사용
        with patch.object(torch.cuda.memory, "_dump_snapshot") as mock_dump:
            RLTrainer._maybe_dump_memory_snapshot(stub, active=False)
            Trainer._maybe_dump_memory_snapshot(stub, active=False)
        mock_dump.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
# _maybe_start_memory_history — 활성 경로
# ──────────────────────────────────────────────────────────────────────────


class TestMemoryHistoryEnabled:
    """memory_history=True + rank-0 → start + dump 호출 확인."""

    def test_memory_history_enabled_starts_and_dumps(self, tmp_path, monkeypatch, caplog_both):
        """True → _record_memory_history + _dump_snapshot 둘 다 mock 으로 호출."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.chdir(tmp_path)  # storage/memory_profiles 는 cwd 기준

        stub = _make_stub(memory_history=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda.memory, "_record_memory_history") as mock_record, \
             patch.object(torch.cuda.memory, "_dump_snapshot") as mock_dump, \
             patch("mlflow.active_run", return_value=None):
            active = RLTrainer._maybe_start_memory_history(stub)
            assert active is True
            mock_record.assert_called_once()
            # stacks="python", max_entries=1_000_000 계약 확인
            call_kwargs = mock_record.call_args.kwargs
            assert call_kwargs.get("stacks") == "python"
            assert call_kwargs.get("max_entries") == 1_000_000

            RLTrainer._maybe_dump_memory_snapshot(stub, active=True)
            mock_dump.assert_called_once()
            # dump 경로가 storage/memory_profiles/*.pickle 형태
            dump_path = mock_dump.call_args.args[0]
            assert "memory_profiles" in dump_path
            assert dump_path.endswith(".pickle")

        # 활성화 로그 확인
        msgs = [r.getMessage() for r in caplog_both.records]
        assert any("_record_memory_history started" in m for m in msgs)
        assert any("memory_history snapshot saved to" in m for m in msgs)

    def test_sft_trainer_symmetric_start_and_dump(self, tmp_path, monkeypatch):
        """Trainer 도 동일 경로."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.chdir(tmp_path)

        stub = _make_stub(memory_history=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda.memory, "_record_memory_history") as mock_record, \
             patch.object(torch.cuda.memory, "_dump_snapshot") as mock_dump, \
             patch("mlflow.active_run", return_value=None):
            active = Trainer._maybe_start_memory_history(stub)
            assert active is True
            mock_record.assert_called_once()
            Trainer._maybe_dump_memory_snapshot(stub, active=True)
            mock_dump.assert_called_once()

    def test_dump_uses_mlflow_run_id_when_available(self, tmp_path, monkeypatch):
        """MLflow active_run 이 있으면 그 run_id 로 pickle 파일명 생성."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.chdir(tmp_path)

        mock_run = MagicMock()
        mock_run.info.run_id = "abc123def456"

        stub = _make_stub(memory_history=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda.memory, "_record_memory_history"), \
             patch.object(torch.cuda.memory, "_dump_snapshot") as mock_dump, \
             patch("mlflow.active_run", return_value=mock_run):
            RLTrainer._maybe_dump_memory_snapshot(stub, active=True)

        dump_path = mock_dump.call_args.args[0]
        assert "abc123def456.pickle" in dump_path


# ──────────────────────────────────────────────────────────────────────────
# rank-0 전용 분기
# ──────────────────────────────────────────────────────────────────────────


class TestMemoryHistoryRank0Only:
    """비-rank-0 에서는 활성화하지 않는다 (pickle 중복 저장 회피)."""

    def test_memory_history_rank0_only(self, monkeypatch):
        """RANK=2 이면 active=False — record 호출 없음."""
        monkeypatch.setenv("RANK", "2")
        stub = _make_stub(memory_history=True)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda.memory, "_record_memory_history") as mock_record:
            active_rl = RLTrainer._maybe_start_memory_history(stub)
            active_sft = Trainer._maybe_start_memory_history(stub)

        assert active_rl is False
        assert active_sft is False
        mock_record.assert_not_called()

    def test_cuda_unavailable_returns_false(self, monkeypatch):
        """CUDA 미가용 환경에서는 memory_history=True 여도 활성화 안 한다."""
        monkeypatch.setenv("RANK", "0")
        stub = _make_stub(memory_history=True)
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda.memory, "_record_memory_history") as mock_record:
            active_rl = RLTrainer._maybe_start_memory_history(stub)
            active_sft = Trainer._maybe_start_memory_history(stub)

        assert active_rl is False
        assert active_sft is False
        mock_record.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────
# 실패 graceful — warning 후 학습 계속
# ──────────────────────────────────────────────────────────────────────────


class TestMemoryHistoryFailureGraceful:
    """_record_memory_history · _dump_snapshot 실패가 학습 루프를 깨뜨리지 않는다."""

    def test_memory_history_start_failure_graceful(self, monkeypatch, caplog_both):
        """_record_memory_history 가 예외 → warning + active=False."""
        monkeypatch.setenv("RANK", "0")
        stub = _make_stub(memory_history=True)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(
                 torch.cuda.memory, "_record_memory_history",
                 side_effect=RuntimeError("record_memory_history unsupported"),
             ):
            active = RLTrainer._maybe_start_memory_history(stub)

        assert active is False
        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.WARNING]
        assert any("memory_history start failed" in m for m in msgs)

    def test_memory_history_dump_failure_graceful(self, tmp_path, monkeypatch, caplog_both):
        """_dump_snapshot 예외 → warning + 학습 계속 (예외 흡수)."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.chdir(tmp_path)

        stub = _make_stub(memory_history=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(
                 torch.cuda.memory, "_dump_snapshot",
                 side_effect=RuntimeError("dump failed"),
             ), \
             patch("mlflow.active_run", return_value=None):
            # 예외가 helper 밖으로 빠져나오지 않아야 한다.
            RLTrainer._maybe_dump_memory_snapshot(stub, active=True)

        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.WARNING]
        assert any("memory snapshot save failed" in m for m in msgs)

    def test_sft_trainer_dump_failure_symmetric(self, tmp_path, monkeypatch, caplog_both):
        """SFT Trainer 도 동일하게 예외 흡수."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.chdir(tmp_path)

        stub = _make_stub(memory_history=True)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(
                 torch.cuda.memory, "_dump_snapshot",
                 side_effect=RuntimeError("dump failed"),
             ), \
             patch("mlflow.active_run", return_value=None):
            Trainer._maybe_dump_memory_snapshot(stub, active=True)

        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.WARNING]
        assert any("memory snapshot save failed" in m for m in msgs)
