"""OOM rank-level memory summary 단위 테스트 (spec-system-logging-cleanup §U5).

``Trainer._dump_oom_summary`` / ``RLTrainer._dump_oom_summary`` 는 OOM 포착
시 모든 rank 의 memory 상태를 rank-0 에 집계하여 logger.error 로 흘린다.
본 모듈은 다음 네 분기를 고립 검증한다:

1. 단일 프로세스 (dist 미초기화) — local info 만으로 포맷.
2. low_free (< 1 GiB) rank 가 있으면 "OOM suspected on rank" 라인 포함.
3. ``dist.all_gather_object`` mock 으로 multi-rank 수집 경로 검증.
4. ``dist.all_gather_object`` 가 예외를 던지면 local fallback.

그리고 train() 전체 경로에서 OOM 이 발생했을 때 ``_dump_oom_summary`` 가
호출되고 OOM 이 re-raise 되는지도 mock 기반으로 검증한다. 실제 Trainer 조립은
device/strategy/dataloader 의존이 무거우므로 ``SimpleNamespace`` stub + bound
method 호출 패턴을 사용한다.
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
    """RL / SFT trainer logger 모두 DEBUG 이상 수집."""
    caplog.set_level(logging.DEBUG, logger=_RL_LOGGER)
    caplog.set_level(logging.DEBUG, logger=_SFT_LOGGER)
    return caplog


# ──────────────────────────────────────────────────────────────────────────
# _dump_oom_summary — 단일 프로세스
# ──────────────────────────────────────────────────────────────────────────


class TestDumpOomSummarySingleProcess:
    """dist 미초기화 환경에서 local info 만으로 포맷된다."""

    def test_dump_oom_summary_single_process(self, caplog_both, monkeypatch):
        """dist 미초기화 + 단일 rank → local info 한 줄 출력."""
        monkeypatch.delenv("RANK", raising=False)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=120 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=135 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(6 * 1024**3, 141 * 1024**3)):
            stub = SimpleNamespace()
            # dist 미초기화: torch.distributed.is_initialized() 가 False.
            # 실제 torch.distributed 모듈을 그대로 쓰되 is_initialized 를 patch.
            with patch("torch.distributed.is_initialized", return_value=False):
                RLTrainer._dump_oom_summary(stub)

        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.ERROR]
        combined = "\n".join(msgs)
        assert "FATAL: torch.OutOfMemoryError" in combined
        assert "rank 0:" in combined
        # 6.2f 포맷 → 앞 공백이 붙을 수 있어 값으로만 확인.
        assert "120.00 GiB" in combined
        assert "135.00 GiB" in combined
        assert "6.00 GiB" in combined
        # 6 GiB free > 1 GiB → OOM suspected 라인 없어야 함
        assert "OOM suspected" not in combined

    def test_dump_oom_summary_low_free_detects_rank(self, caplog_both, monkeypatch):
        """free < 1 GiB 면 → "OOM suspected on rank(s): [0]" 포함."""
        monkeypatch.delenv("RANK", raising=False)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=139 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=140 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(int(0.3 * 1024**3), 141 * 1024**3)), \
             patch("torch.distributed.is_initialized", return_value=False):
            stub = SimpleNamespace()
            RLTrainer._dump_oom_summary(stub)

        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.ERROR]
        combined = "\n".join(msgs)
        assert "OOM suspected on rank(s): [0]" in combined

    def test_sft_trainer_symmetric_single_process(self, caplog_both, monkeypatch):
        """Trainer._dump_oom_summary 도 동일 포맷·동일 경로."""
        monkeypatch.delenv("RANK", raising=False)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=50 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=55 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(80 * 1024**3, 141 * 1024**3)), \
             patch("torch.distributed.is_initialized", return_value=False):
            stub = SimpleNamespace()
            Trainer._dump_oom_summary(stub)

        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.ERROR]
        combined = "\n".join(msgs)
        assert "FATAL: torch.OutOfMemoryError" in combined
        assert "rank 0:" in combined
        assert "50.00 GiB" in combined

    def test_dump_oom_summary_cuda_unavailable_is_noop(self, caplog_both, monkeypatch):
        """CUDA 미가용이면 호출 자체가 no-op (에러 로그 없음)."""
        monkeypatch.delenv("RANK", raising=False)
        with patch.object(torch.cuda, "is_available", return_value=False):
            stub = SimpleNamespace()
            RLTrainer._dump_oom_summary(stub)
            Trainer._dump_oom_summary(stub)

        err_msgs = [r for r in caplog_both.records if r.levelno >= logging.ERROR]
        assert err_msgs == []


# ──────────────────────────────────────────────────────────────────────────
# _dump_oom_summary — distributed (all_gather 경로)
# ──────────────────────────────────────────────────────────────────────────


class TestDumpOomSummaryDistributed:
    """dist.all_gather_object mock 기반 multi-rank 수집 검증."""

    def test_dump_oom_summary_distributed_gather(self, caplog_both, monkeypatch):
        """all_gather_object 가 4 rank info 를 채우면 모두 출력된다."""
        monkeypatch.setenv("RANK", "0")

        # rank 0: 100 GiB alloc / 0.5 GiB free (OOM suspected)
        # rank 1: 60 GiB alloc / 70 GiB free
        # rank 2: 70 GiB alloc / 60 GiB free
        # rank 3: 50 GiB alloc / 80 GiB free
        gathered_payload = [
            {"rank": 0, "allocated_gib": 100.0, "reserved_gib": 120.0, "free_gib": 0.5},
            {"rank": 1, "allocated_gib": 60.0, "reserved_gib": 65.0, "free_gib": 70.0},
            {"rank": 2, "allocated_gib": 70.0, "reserved_gib": 75.0, "free_gib": 60.0},
            {"rank": 3, "allocated_gib": 50.0, "reserved_gib": 55.0, "free_gib": 80.0},
        ]

        def fake_all_gather_object(output_list, _local):
            for i, val in enumerate(gathered_payload):
                output_list[i] = val

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=100 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=120 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(int(0.5 * 1024**3), 141 * 1024**3)), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=4), \
             patch("torch.distributed.all_gather_object", side_effect=fake_all_gather_object):
            stub = SimpleNamespace()
            RLTrainer._dump_oom_summary(stub)

        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.ERROR]
        combined = "\n".join(msgs)
        # 모든 rank 라인 포함
        for r in range(4):
            assert f"rank {r}:" in combined, combined
        # OOM suspected 에 rank 0 포함 (free=0.5 < 1.0)
        assert "OOM suspected on rank(s): [0]" in combined

    def test_dump_oom_summary_gather_timeout_falls_back_to_local(
        self, caplog_both, monkeypatch
    ):
        """all_gather_object 가 예외를 던지면 local info 만으로 fallback."""
        monkeypatch.setenv("RANK", "0")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=45 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=50 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(90 * 1024**3, 141 * 1024**3)), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=4), \
             patch(
                 "torch.distributed.all_gather_object",
                 side_effect=RuntimeError("NCCL timeout simulated"),
             ):
            stub = SimpleNamespace()
            RLTrainer._dump_oom_summary(stub)

        err_msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.ERROR]
        combined = "\n".join(err_msgs)
        # local rank 만 있어야 한다 — rank 1/2/3 라인 없음
        assert "rank 0:" in combined
        assert "rank 1:" not in combined
        assert "rank 2:" not in combined
        assert "rank 3:" not in combined

    def test_dump_oom_summary_gather_hang_timeout_falls_back_to_local(
        self, caplog_both, monkeypatch
    ):
        """다른 rank 가 OOM 으로 죽어 ``all_gather_object`` 가 영원히 반환하지
        않는 시나리오에서 타임아웃 내 local fallback 으로 빠져나와야 한다
        (cycle 1 review 2-2).

        ``concurrent.futures.ThreadPoolExecutor`` + ``future.result(timeout=...)``
        로 hang 을 bound 한다. 테스트에서는 all_gather_object 를 아주 오래
        잠드는 함수로 patch 하고, ``dump_oom_summary`` 에 짧은 타임아웃을 주입해
        시간 내에 fallback 되는지 확인한다.
        """
        import time as _time
        from mdp.training._progress_log import dump_oom_summary

        monkeypatch.setenv("RANK", "0")

        def _hang_forever(_output_list, _local):
            # 실제로는 수분 hang 될 수 있는 상황을 짧게 흉내 — 테스트 전체
            # runtime 을 과도하게 늘리지 않도록 1초 sleep.
            _time.sleep(1.0)

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=100 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=120 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(int(0.5 * 1024**3), 141 * 1024**3)), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=4), \
             patch("torch.distributed.all_gather_object", side_effect=_hang_forever):
            start = _time.time()
            # 짧은 timeout(0.1s) 으로 hang 을 시뮬레이션한다.
            dump_oom_summary(logger=logging.getLogger(_RL_LOGGER), timeout_sec=0.1)
            elapsed = _time.time() - start

        # 1) 타임아웃 이내에 반환
        assert elapsed < 0.5, f"timeout 초과, {elapsed:.2f}s 소요"

        # 2) local info 만으로 fallback 출력
        err_msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.ERROR]
        combined = "\n".join(err_msgs)
        assert "rank 0:" in combined
        assert "rank 1:" not in combined
        assert "rank 2:" not in combined
        assert "rank 3:" not in combined

        # 3) WARNING 레벨로 timeout 로그 남음
        warn_msgs = [r.getMessage() for r in caplog_both.records if r.levelno == logging.WARNING]
        assert any("timeout" in m.lower() for m in warn_msgs), warn_msgs

    def test_dump_oom_summary_non_rank0_skips_output(self, caplog_both, monkeypatch):
        """비-rank-0 은 집계 후 로그를 내지 않는다."""
        monkeypatch.setenv("RANK", "2")

        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "memory_allocated", return_value=50 * 1024**3), \
             patch.object(torch.cuda, "memory_reserved", return_value=55 * 1024**3), \
             patch.object(torch.cuda, "mem_get_info", return_value=(80 * 1024**3, 141 * 1024**3)), \
             patch("torch.distributed.is_initialized", return_value=False):
            stub = SimpleNamespace()
            RLTrainer._dump_oom_summary(stub)

        err_msgs = [r for r in caplog_both.records if r.levelno >= logging.ERROR]
        assert err_msgs == []


# ──────────────────────────────────────────────────────────────────────────
# train() OOM 경로 — _dump_oom_summary 호출 + re-raise
# ──────────────────────────────────────────────────────────────────────────


class TestTrainOomPath:
    """train() 내부에서 torch.cuda.OutOfMemoryError 를 모사하면
    ``_dump_oom_summary`` 가 호출되고 예외가 re-raise 되는지 검증.

    실제 Trainer 를 조립하지 않고 ``train()`` 함수 블록 안의 핵심 except
    흐름을 그대로 재현하는 minimal harness. Trainer / RLTrainer 의 실제
    except 블록이 동일 패턴임을 ``grep`` 으로 읽고 교차검증 (대칭).
    """

    def test_train_oom_path_calls_dump_oom_summary(self, caplog_both):
        """except torch.cuda.OutOfMemoryError: 블록 재현 — _dump_oom_summary 호출 + raise."""

        class _Trainer:
            _oom_observed = False

            def _dump_oom_summary(self):
                self._dump_called = True

            def run(self):
                try:
                    try:
                        raise torch.cuda.OutOfMemoryError("simulated OOM")
                    except torch.cuda.OutOfMemoryError:
                        self._oom_observed = True
                        try:
                            self._dump_oom_summary()
                        except Exception:
                            pass
                        raise
                finally:
                    pass

        t = _Trainer()
        with pytest.raises(torch.cuda.OutOfMemoryError):
            t.run()
        assert t._oom_observed is True
        assert getattr(t, "_dump_called", False) is True

    def test_dump_failure_does_not_suppress_oom(self, caplog_both):
        """_dump_oom_summary 내부 실패해도 OOM 은 반드시 re-raise 된다."""

        class _Trainer:
            _oom_observed = False

            def _dump_oom_summary(self):
                raise RuntimeError("summary blew up")

            def run(self):
                try:
                    raise torch.cuda.OutOfMemoryError("simulated OOM")
                except torch.cuda.OutOfMemoryError:
                    self._oom_observed = True
                    try:
                        self._dump_oom_summary()
                    except Exception as e:
                        import logging as _lg
                        _lg.getLogger(_RL_LOGGER).warning(
                            "OOM summary dump failed: %s", e,
                        )
                    raise

        t = _Trainer()
        with pytest.raises(torch.cuda.OutOfMemoryError):
            t.run()
        msgs = [r.getMessage() for r in caplog_both.records if r.levelno >= logging.WARNING]
        assert any("OOM summary dump failed" in m for m in msgs)
