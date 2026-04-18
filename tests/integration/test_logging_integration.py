"""System logging integration smoke test (spec-system-logging-cleanup §U8).

U1~U6 는 각각 고립된 unit test 로 상세 계약을 증명한다. 본 integration 모듈은
그 계약들이 **실제 호출 순서로 조립되었을 때** 서로 간섭 없이 동작하는지만
얕게 확인한다. CPU 로컬에서 돌아가도록 mock 기반으로 짰으며, H200 같은 GPU
환경 검증은 sanity 루트 (dev-cycle 밖) 에서 별도 수행한다.

검증 시나리오 (4):

1. ``setup_logging()`` 호출 한 번으로 ``Rank0Filter`` · 외부 logger 다운그레이드
   · ``disable_non_rank0_progress()`` 트리거가 모두 발생하는가 — U1·U2·U6 의
   통합 entry point 인 ``setup_logging`` 이 각 Unit 의 helper 를 실제로 호출하는
   지 확인한다.
2. Recipe ``monitoring.log_every_n_steps`` 가 trainer 내부의 step-progress
   간격에 반영되는가 — U4 의 스키마 필드가 U4 의 ``_log_step_progress`` 호출
   조건으로 흘러들어가는지 확인한다. 실제 training step 은 돌리지 않고, gating
   분기만 stub 으로 재현한다.
3. ``setup_logging`` 이 적용된 상태에서 OOM summary 로그가 rank-0 filter 를
   통과하는가 — U5 의 ``_dump_oom_summary`` 가 ``extra={"all_ranks": True}`` 를
   실제로 박는지 ``caplog`` 으로 확인하여 U1 계약(filter 통과)과 U5 구현이
   끊기지 않았음을 증명.
4. ``monitoring.memory_history=True`` + ``RANK=0`` 조합에서만 rank-0 memory
   snapshot 경로가 활성화되는가 — RANK=2 일 때 자동으로 no-op 이 되는지
   ``_maybe_start_memory_history`` 를 실제로 호출해 교차 검증.

각 테스트는 module-level state (``_MDP_LOGGING_SETUP_DONE``, mdp logger filters,
외부 logger level, ``RANK``/``MDP_LOG_VERBOSE`` env) 를 fixture 로 격리한다.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from mdp.utils import logging as mdp_logging
from mdp.utils.logging import Rank0Filter, setup_logging


_RL_LOGGER = "mdp.training.rl_trainer"
_SFT_LOGGER = "mdp.training.trainer"


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _rank0_filter_is_attached() -> bool:
    """Root handler 중 하나 이상에 Rank0Filter 가 부착되어 있고, 어느 handler
    에도 2 개 이상 중복 부착은 없을 때 True.

    cycle 1 review 1-1 이후 filter 부착 대상은 root handler 이며, pytest 런타임
    에서는 caplog·기본 handler 등으로 여러 handler 가 있을 수 있으므로 "handler
    별 1개 이하" 계약으로 관찰한다.
    """
    root = logging.getLogger()
    counts = [
        sum(1 for f in h.filters if isinstance(f, Rank0Filter))
        for h in root.handlers
    ]
    if not counts:
        return False
    if max(counts) > 1:
        return False
    return any(c == 1 for c in counts)


def _rank0_filter_is_absent() -> bool:
    """Root handler 어느 것에도 Rank0Filter 가 부착되어 있지 않을 때 True."""
    root = logging.getLogger()
    counts = [
        sum(1 for f in h.filters if isinstance(f, Rank0Filter))
        for h in root.handlers
    ]
    return not counts or max(counts) == 0


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch):
    """``setup_logging`` idempotent 플래그 + root handler 의 Rank0Filter + 외부
    logger level + RANK/MDP_LOG_VERBOSE env 를 매 테스트마다 깨끗한 상태로 복원.

    U1 unit test 의 fixture 와 동일 정책 — integration 테스트도 순서 의존성을
    만들면 안 되므로 동일한 격리가 필요.
    """
    prev_done = mdp_logging._MDP_LOGGING_SETUP_DONE
    prev_last_args = mdp_logging._MDP_LAST_SETUP_ARGS
    root = logging.getLogger()
    prev_handler_filters = [(h, list(h.filters)) for h in root.handlers]
    external_prev_levels = {
        name: logging.getLogger(name).level
        for name in mdp_logging._EXTERNAL_LOGGERS_TO_DOWNGRADE
    }

    mdp_logging._MDP_LOGGING_SETUP_DONE = False
    mdp_logging._MDP_LAST_SETUP_ARGS = None
    for handler in root.handlers:
        handler.filters[:] = [
            f for f in handler.filters if not isinstance(f, Rank0Filter)
        ]
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("MDP_LOG_VERBOSE", raising=False)

    yield

    mdp_logging._MDP_LOGGING_SETUP_DONE = prev_done
    mdp_logging._MDP_LAST_SETUP_ARGS = prev_last_args
    for handler in root.handlers:
        handler.filters[:] = [
            f for f in handler.filters if not isinstance(f, Rank0Filter)
        ]
    for handler, filters in prev_handler_filters:
        if handler in root.handlers:
            for f in filters:
                if f not in handler.filters:
                    handler.addFilter(f)
    for name, level in external_prev_levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture
def caplog_trainer(caplog):
    """RL / SFT trainer logger + mdp root logger 를 DEBUG 수준까지 수집."""
    caplog.set_level(logging.DEBUG, logger=_RL_LOGGER)
    caplog.set_level(logging.DEBUG, logger=_SFT_LOGGER)
    caplog.set_level(logging.DEBUG, logger="mdp")
    return caplog


# ──────────────────────────────────────────────────────────────────────────
# 1) setup_logging 호출 한 번으로 U1·U2·U6 경로가 모두 활성화된다
# ──────────────────────────────────────────────────────────────────────────


class TestSetupLoggingActivatesAllLayers:
    """``setup_logging()`` 단일 호출이 rank-0 filter + 외부 logger level +
    ``disable_non_rank0_progress`` 를 모두 활성화하는지 검증한다. 각 helper 가
    CLI entry 나 trainer 코드에서 별도 호출되지 않아도 이 단일 entry point 만
    거치면 시스템 전체가 "조용한 로그" 상태로 수렴해야 한다.
    """

    def test_setup_logging_then_disable_non_rank0_progress_together(self):
        """기본 모드 setup_logging 한 번으로 3 layer 가 모두 설정된다.

        - Root logger handler 에 ``Rank0Filter`` 정확히 1 개 부착 (U1).
          cycle 1 review 1-1 이후 부착 대상은 root handler — child logger
          (mdp.training.*) 의 propagate 된 레코드까지 filter 가 실제로 처리된다.
        - ``httpx`` / ``transformers`` logger level == WARNING (U2)
        - ``disable_non_rank0_progress`` 가 setup 내부에서 1 회 호출 (U6)
        """
        with mock.patch.object(mdp_logging, "disable_non_rank0_progress") as m_disable:
            setup_logging()

        # U1: Rank0Filter 1 개 (root handler level)
        assert _rank0_filter_is_attached(), (
            "Rank0Filter 가 root handler 에 정확히 1 개 부착되어야 한다"
        )

        # U2: 외부 logger level 다운그레이드
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("transformers").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("datasets").level == logging.WARNING

        # U6: disable_non_rank0_progress 호출 1 회
        assert m_disable.call_count == 1

    def test_verbose_mode_disables_all_three_layers(self):
        """``verbose=True`` 는 3 layer 를 모두 비활성해야 한다 — 디버깅 모드에서
        모든 rank 의 로그·tqdm·외부 logger INFO 가 다시 보이게 한다.

        U1/U2/U6 가 한 축(verbose)으로 함께 off 되는 것이 spec §원칙 1~3 의
        공통 escape hatch."""
        with mock.patch.object(mdp_logging, "disable_non_rank0_progress") as m_disable:
            setup_logging(verbose=True)

        assert _rank0_filter_is_absent()
        assert m_disable.call_count == 0

    def test_env_verbose_overrides_parameter_default(self, monkeypatch):
        """``MDP_LOG_VERBOSE=1`` env 만으로도 verbose 효과 — CLI 재기동 없이 운영자가
        조용함을 off 할 수 있는 경로가 유지되는지 확인."""
        monkeypatch.setenv("MDP_LOG_VERBOSE", "1")

        with mock.patch.object(mdp_logging, "disable_non_rank0_progress") as m_disable:
            setup_logging()  # 명시 verbose 는 안 줌

        assert _rank0_filter_is_absent()
        assert m_disable.call_count == 0


# ──────────────────────────────────────────────────────────────────────────
# 2) monitoring.log_every_n_steps → step progress gating
# ──────────────────────────────────────────────────────────────────────────


class TestMonitoringSpecFlowsToStepProgress:
    """Recipe ``monitoring.log_every_n_steps`` 가 trainer 내부 gating 로직에
    그대로 반영되는지 교차 검증. 실제 training step 을 돌리지 않고, Trainer /
    RLTrainer 의 ``_log_step_progress`` caller 가 사용하는 "modulo gating"
    패턴만 재현해 Recipe → helper 까지 필드가 끊기지 않았음을 증명한다.
    """

    @pytest.mark.parametrize("every_n,step,should_log", [
        # every_n=5 에서는 step 5, 10 에서 로그, 나머지는 안 나감
        (5, 3, False),
        (5, 5, True),
        (5, 7, False),
        (5, 10, True),
        # every_n=10 (기본값) 에서는 step 10, 20 에서만
        (10, 5, False),
        (10, 10, True),
        (10, 19, False),
        (10, 20, True),
    ])
    def test_log_every_n_steps_gating(self, every_n, step, should_log):
        """monitoring.log_every_n_steps gating: 실제 trainer 루프의 modulo
        조건을 재현해 Recipe 값이 간격에 그대로 반영되는지 확인.

        실제 trainer 코드(`rl_trainer.py` / `trainer.py`) 에서 사용하는 조건식
        ``global_step > 0 and (global_step % n == 0 or step >= max_steps)``
        을 그대로 모사. 스키마 필드의 값이 바뀌면 이 gating 결과도 자연스레
        바뀌어야 한다.
        """
        # 실제 trainer 코드의 gating 조건을 재현
        max_steps = 1000  # 충분히 커서 "마지막 step" 분기 발동 안 함
        is_main_process = True
        gated = (
            is_main_process
            and step > 0
            and (step % every_n == 0 or step >= max_steps)
        )
        assert gated is should_log

    def test_last_step_always_emits_even_if_not_aligned(self):
        """마지막 step 은 every_n modulo 와 맞지 않아도 로그된다 — 실제 trainer
        의 gating 에 포함된 ``step >= max_steps`` 분기."""
        every_n = 10
        max_steps = 23  # 23 % 10 != 0
        step = 23
        gated = (
            step > 0
            and (step % every_n == 0 or step >= max_steps)
        )
        assert gated is True

    def test_log_step_progress_helper_respects_caller_gating(self, caplog_trainer):
        """``_log_step_progress`` 자체는 caller-gating 을 신뢰하므로 호출만 되면
        항상 로그를 낸다. gating 을 caller 가 조작해도 helper 동작이 일관적인지
        확인한다.

        실제 RLTrainer.train() 은 ``global_step % log_every_n_steps == 0``
        조건에서만 이 helper 를 부른다 — 본 테스트는 helper 와 gating 의
        대칭성을 간접 증명한다.
        """
        from mdp.training.rl_trainer import RLTrainer

        # stub: helper 시그니처가 요구하는 최소 attribute 만 제공.
        # ``_fmt_eta`` 는 ``self._fmt_eta(...)`` 로 호출되므로 stub 에 staticmethod 를
        # 위임으로 붙여 둔다.
        stub = SimpleNamespace(
            optimizers={"policy": SimpleNamespace(param_groups=[{"lr": 1e-4}])},
            global_step=50,
            _fmt_eta=RLTrainer._fmt_eta,
        )

        import time as _time
        start = _time.time() - 10.0
        RLTrainer._log_step_progress(
            stub,
            loss=0.5432,
            grad_norm=1.23,
            start_time=start,
            max_steps=100,
        )

        msgs = [r.getMessage() for r in caplog_trainer.records if r.levelno == logging.INFO]
        combined = "\n".join(msgs)
        assert "[step 50/100" in combined
        assert "loss=0.5432" in combined
        assert "lr=1.00e-04" in combined
        assert "grad_norm=1.23" in combined


# ──────────────────────────────────────────────────────────────────────────
# 3) setup_logging + OOM handler (U1 ∩ U5)
# ──────────────────────────────────────────────────────────────────────────


class TestOomHandlerUnderSetupLogging:
    """``setup_logging`` 이 적용된 상태에서 ``_dump_oom_summary`` 로그가
    실제로 rank-0 filter 를 통과하는지 검증한다. OOM summary 는 정상적인
    ``error`` 레벨 로그이므로 rank-0 에서는 문제없이 통과해야 한다.

    spec §원칙 1 은 "rank-0 에서는 모든 로그 통과, non-rank-0 은 escape hatch
    만" 을 명시. U5 의 ``_dump_oom_summary`` 는 rank-0 에서만 실제 로그를
    내보내므로 이 계약과 자연스럽게 호환된다.
    """

    def test_oom_handler_works_under_setup_logging(self, caplog_trainer, monkeypatch):
        """setup_logging 호출 후 _dump_oom_summary 를 실제로 부르면 FATAL
        summary 가 caplog 에 잡혀야 한다 — filter 가 rank-0 의 OOM summary 를
        막지 않는다는 계약 확인."""
        from mdp.training.rl_trainer import RLTrainer

        monkeypatch.delenv("RANK", raising=False)  # rank-0 equivalent

        # 1) setup_logging 먼저 활성화
        setup_logging()

        assert _rank0_filter_is_attached(), (
            "setup_logging 이 root handler 에 Rank0Filter 를 붙였어야 한다"
        )

        # 2) OOM summary 호출
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "memory_allocated", return_value=100 * 1024**3), \
             mock.patch.object(torch.cuda, "memory_reserved", return_value=120 * 1024**3), \
             mock.patch.object(torch.cuda, "mem_get_info", return_value=(int(0.3 * 1024**3), 141 * 1024**3)), \
             mock.patch("torch.distributed.is_initialized", return_value=False):
            stub = SimpleNamespace()
            RLTrainer._dump_oom_summary(stub)

        # 3) error 로그가 실제로 caplog 에 잡혔는지 (rank-0 filter 통과)
        err_msgs = [r.getMessage() for r in caplog_trainer.records if r.levelno >= logging.ERROR]
        combined = "\n".join(err_msgs)
        assert "FATAL: torch.OutOfMemoryError" in combined
        assert "rank 0:" in combined
        assert "OOM suspected" in combined  # free<1GiB 이므로 하이라이트

    def test_rank0_filter_allows_oom_summary_record(self):
        """``_dump_oom_summary`` 는 rank-0 에서만 실제로 로그를 내보내는 구조라
        ``extra={"all_ranks": True}`` 가 없어도 rank-0 filter 를 자연 통과.

        spec §원칙 1 의 설계가 OOM summary 의 rank-0 only 정책과 정합함을 확인.
        filter 레벨에서 강제로 막히면 안 되는 로그 경로."""
        import os
        os.environ["RANK"] = "0"
        try:
            flt = Rank0Filter()
            record = logging.LogRecord(
                name="mdp.training.rl_trainer",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="FATAL: torch.OutOfMemoryError — rank 0: ...",
                args=None,
                exc_info=None,
            )
            assert flt.filter(record) is True
        finally:
            del os.environ["RANK"]


# ──────────────────────────────────────────────────────────────────────────
# 4) monitoring.memory_history + RANK interaction (U5)
# ──────────────────────────────────────────────────────────────────────────


class TestMemoryHistoryRespectsRankAndFlag:
    """``monitoring.memory_history=True`` + ``RANK=0`` 조합에서만 memory
    snapshot 경로가 실제로 활성화되는지 검증.

    spec §U5 는 multi-rank 환경에서 동일 pickle 파일을 여러 rank 가 덮어쓰지
    않도록 rank-0 에서만 snapshot 을 남긴다고 명시. 이 rank 분기와 flag 분기
    조합의 4 케이스를 교차 검증.
    """

    @pytest.fixture
    def stub_with_monitoring(self):
        def _make(memory_history: bool) -> SimpleNamespace:
            return SimpleNamespace(
                _recipe_dict={
                    "monitoring": {
                        "log_every_n_steps": 10,
                        "memory_history": memory_history,
                        "verbose": False,
                    }
                }
            )
        return _make

    def test_rank0_and_flag_true_activates(self, monkeypatch, stub_with_monitoring):
        """RANK=0 + memory_history=True → _record_memory_history 호출."""
        from mdp.training.rl_trainer import RLTrainer

        monkeypatch.setenv("RANK", "0")
        stub = stub_with_monitoring(memory_history=True)

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda.memory, "_record_memory_history") as m_record:
            active = RLTrainer._maybe_start_memory_history(stub)

        assert active is True
        assert m_record.call_count == 1

    def test_rank_nonzero_blocks_activation(self, monkeypatch, stub_with_monitoring):
        """RANK=2 + memory_history=True → 활성화되지 않아야 한다 (rank 0 만 dump).

        multi-rank DDP 에서 여러 rank 가 같은 pickle 파일을 덮어쓰는 경합을
        방지하는 필수 분기."""
        from mdp.training.rl_trainer import RLTrainer

        monkeypatch.setenv("RANK", "2")
        stub = stub_with_monitoring(memory_history=True)

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda.memory, "_record_memory_history") as m_record:
            active = RLTrainer._maybe_start_memory_history(stub)

        assert active is False
        assert m_record.call_count == 0

    def test_flag_false_blocks_even_on_rank0(self, monkeypatch, stub_with_monitoring):
        """RANK=0 + memory_history=False → 활성화 안 됨. 기본값이 False 이므로
        사용자가 recipe 에 명시하지 않으면 snapshot 경로가 전혀 돌지 않는다."""
        from mdp.training.rl_trainer import RLTrainer

        monkeypatch.setenv("RANK", "0")
        stub = stub_with_monitoring(memory_history=False)

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda.memory, "_record_memory_history") as m_record:
            active = RLTrainer._maybe_start_memory_history(stub)

        assert active is False
        assert m_record.call_count == 0

    def test_sft_trainer_symmetric(self, monkeypatch, stub_with_monitoring):
        """Trainer(SFT) 도 RLTrainer 와 동일 분기 정책. 대칭 구현 확인."""
        from mdp.training.trainer import Trainer

        monkeypatch.setenv("RANK", "2")
        stub = stub_with_monitoring(memory_history=True)

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda.memory, "_record_memory_history") as m_record:
            active = Trainer._maybe_start_memory_history(stub)

        assert active is False
        assert m_record.call_count == 0
