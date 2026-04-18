"""U1 — Rank0Filter / setup_logging / WARNING_SUPPRESS_PATTERNS 단위 테스트.

spec-system-logging-cleanup §U1 의 Verify 기준(``TestRank0Filter`` 통과)을
충족한다. 후속 Unit(U2 의 external logger downgrade, U4 의 verbose 배너 경로,
U5 의 OOM summary) 는 본 테스트가 박아둔 공유 계약을 그대로 소비한다:

- ``Rank0Filter`` 생성자는 ``RANK`` env 를 캡처하고, 이후 env 변경은 filter
  인스턴스 동작에 영향 없음.
- ``extra={"all_ranks": True}`` escape hatch 의 키 이름 변경 금지.
- ``setup_logging`` idempotency — 동일 인자 재호출은 no-op. root handler 의
  Rank0Filter 개수는 정확히 1.
- ``setup_logging`` args-aware 재설정 — 인자가 바뀌면 상태 전환. verbose=False
  → verbose=True 호출 시 Rank0Filter 가 제거되고, 외부 logger level 이 원복된다.
- ``verbose=True`` 또는 ``MDP_LOG_VERBOSE=1`` 세팅 시 root handler 에 Rank0Filter
  가 부착되지 않는다.

필터의 부착 대상은 cycle 1 review 1-1 이후 **root logger 의 각 handler** 로
이동했다. Python logging 은 child logger(예: ``mdp.training.rl_trainer``)의
레코드를 propagate 체인을 통해 root handler 까지 운반하며, logger 자체의
filters 는 "직접 log 된 레코드" 에만 적용되므로, 부착 대상이 handler 여야
propagate 경로가 실제로 차단된다.

각 테스트는 module-level state (``_MDP_LOGGING_SETUP_DONE``,
``_MDP_LAST_SETUP_ARGS``, root handler 의 Rank0Filter, 외부 logger level) 를
정확히 되돌리기 위해 fixture 로 격리한다.
"""

from __future__ import annotations

import logging
import re
import sys
from unittest import mock

import pytest

from mdp.utils import logging as mdp_logging
from mdp.utils.logging import (
    Rank0Filter,
    WARNING_SUPPRESS_PATTERNS,
    disable_non_rank0_progress,
    setup_logging,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _handlers_with_rank0_filter() -> list[logging.Handler]:
    """Root logger 의 handler 중 Rank0Filter 가 부착된 것들만 반환.

    Handler-level 부착 이후 "filter 가 필요한 handler 에 모두, 각 1 개씩" 이
    계약이다. 다수 handler 존재 환경(pytest caplog 등) 에서도 handler 마다
    정확히 1 개여야 중복 평가가 없다.
    """
    return [
        h for h in logging.getLogger().handlers
        if any(isinstance(f, Rank0Filter) for f in h.filters)
    ]


def _per_handler_rank0_filter_counts() -> list[int]:
    """Root logger 의 각 handler 별 Rank0Filter 부착 수 리스트.

    중복 부착 회귀를 감지할 때 사용 — 어느 handler 든 2 이상이면 계약 위반.
    """
    return [
        sum(1 for f in h.filters if isinstance(f, Rank0Filter))
        for h in logging.getLogger().handlers
    ]


def _rank0_filter_is_attached() -> bool:
    """Root handler 중 하나 이상에 Rank0Filter 가 부착되어 있고, 어느 handler
    에도 2 개 이상 중복 부착은 없을 때 True.

    대부분 테스트에서 "계약 유지" 검증용. 중복 부착 감지 회귀를 함께 막는다.
    """
    counts = _per_handler_rank0_filter_counts()
    if not counts:
        return False
    if max(counts) > 1:
        return False
    return any(c == 1 for c in counts)


def _rank0_filter_is_absent() -> bool:
    """Root handler 어느 것에도 Rank0Filter 가 부착되어 있지 않을 때 True."""
    counts = _per_handler_rank0_filter_counts()
    return not counts or max(counts) == 0


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch):
    """매 테스트마다 ``setup_logging`` idempotent 플래그 + root handler 의
    Rank0Filter + 외부 logger level + ``RANK`` env 를 깨끗하게 복원한다.

    복원하지 않으면 테스트 간 순서 의존성이 생기고 실제 운영 환경(부작용 1회
    허용) 과 다른 상태가 된다. 필터 부착 대상은 cycle 1 review 1-1 이후
    root handler 이므로 snapshot·복원도 root 쪽에서 수행한다.
    """

    # 기존 state snapshot
    prev_done = mdp_logging._MDP_LOGGING_SETUP_DONE
    prev_last_args = mdp_logging._MDP_LAST_SETUP_ARGS
    root = logging.getLogger()
    # handler 별 filters snapshot (id 로 보존하되 teardown 에서는 동일 객체를
    # 다시 꽂아 둔다)
    prev_handler_filters: list[tuple[logging.Handler, list[logging.Filter]]] = [
        (h, list(h.filters)) for h in root.handlers
    ]
    external_prev_levels = {
        name: logging.getLogger(name).level
        for name in mdp_logging._EXTERNAL_LOGGERS_TO_DOWNGRADE
    }

    # 테스트 진입 시 플래그 리셋
    mdp_logging._MDP_LOGGING_SETUP_DONE = False
    mdp_logging._MDP_LAST_SETUP_ARGS = None
    # filter 도 깨끗한 상태에서 시작 — root handler 의 Rank0Filter 를 모두 제거
    for handler in root.handlers:
        handler.filters[:] = [
            f for f in handler.filters if not isinstance(f, Rank0Filter)
        ]

    # RANK 환경변수는 각 테스트가 필요 시 monkeypatch 로 세팅한다.
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("MDP_LOG_VERBOSE", raising=False)

    yield

    # teardown: 원복
    mdp_logging._MDP_LOGGING_SETUP_DONE = prev_done
    mdp_logging._MDP_LAST_SETUP_ARGS = prev_last_args
    # 새로 붙은 Rank0Filter 제거 후 snapshot 복원
    for handler in root.handlers:
        handler.filters[:] = [
            f for f in handler.filters if not isinstance(f, Rank0Filter)
        ]
    for handler, filters in prev_handler_filters:
        if handler in root.handlers:
            for f in filters:
                if f not in handler.filters:
                    handler.addFilter(f)
    # 외부 logger level 원복
    for name, level in external_prev_levels.items():
        logging.getLogger(name).setLevel(level)


def _make_record(
    name: str = "mdp.training.rl_trainer",
    msg: str = "hello",
    *,
    extra: dict | None = None,
) -> logging.LogRecord:
    """LogRecord 수동 생성 헬퍼. Filter 의 ``filter()`` 메서드를 직접 호출하여
    rank 판정만 검증하기 위한 최소 fixture.

    ``extra`` 는 ``logger.makeRecord`` 가 하는 것과 동일하게 record 에 attribute
    로 setattr 한다. 이것이 실제 ``logger.info(..., extra={...})`` 경로의 동작.
    """

    record = logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=None,
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


# ──────────────────────────────────────────────────────────────────────────
# Rank0Filter
# ──────────────────────────────────────────────────────────────────────────


class TestRank0Filter:
    """Distributed RANK 별 필터 동작 3 종."""

    def test_rank0_passes_through(self, monkeypatch):
        """``RANK=0`` (또는 미세팅) 프로세스는 ``all_ranks`` 플래그 없이도 모든
        레코드가 통과해야 한다 — 단일 프로세스 실행에서 기존 로그 동작이
        그대로 유지됨을 보장하는 계약."""

        monkeypatch.setenv("RANK", "0")
        flt = Rank0Filter()

        assert flt.filter(_make_record()) is True
        assert flt.filter(_make_record(extra={"all_ranks": False})) is True

    def test_non_rank0_blocked(self, monkeypatch):
        """``RANK=2`` 같은 non-zero rank 는 일반 레코드를 차단해야 한다.
        이것이 spec §원칙 1 — DDP 4-rank 중복 제거의 핵심 동작."""

        monkeypatch.setenv("RANK", "2")
        flt = Rank0Filter()

        assert flt.filter(_make_record()) is False
        # all_ranks=False 도 명시적으로 통과하면 안 됨
        assert flt.filter(_make_record(extra={"all_ranks": False})) is False

    def test_extra_all_ranks_passes_through(self, monkeypatch):
        """escape hatch: ``extra={"all_ranks": True}`` 가 박힌 레코드는
        non-zero rank 에서도 통과. FSDP shard baseline / OOM per-rank summary
        등에서 활용할 공유 계약."""

        monkeypatch.setenv("RANK", "3")
        flt = Rank0Filter()

        record = _make_record(extra={"all_ranks": True})
        assert flt.filter(record) is True


# ──────────────────────────────────────────────────────────────────────────
# setup_logging
# ──────────────────────────────────────────────────────────────────────────


class TestSetupLogging:
    """setup_logging 의 핵심 계약: idempotency, args-aware 재설정, verbose
    escape, 외부 logger downgrade."""

    def test_idempotent(self):
        """동일 인자로 두 번 호출해도 root handler 의 Rank0Filter 가 **정확히
        1 개** 만 존재해야 한다. 중복 부착은 filter chain 에서 rank 검사가 두
        번 돌아 성능과 로그 동작을 모두 망가뜨린다."""

        setup_logging()
        setup_logging()  # 동일 인자 재호출 — no-op 기대

        assert _rank0_filter_is_attached()

    def test_verbose_skips_setup(self):
        """``verbose=True`` 이면 root handler 에 Rank0Filter 가 부착되지 않아야
        한다. 디버깅 모드에서 모든 rank 의 로그를 보려는 원래 의도 달성."""

        setup_logging(verbose=True)

        assert _rank0_filter_is_absent()

    def test_suppress_external_downgrades_httpx_level(self):
        """``suppress_external=True`` (기본값) 시 httpx logger level 이
        WARNING 으로 올라가야 한다 — INFO 루틴 요청 로그 차단. 이는 spec
        §원칙 2 의 기본 약속이며 U2 가 직접 의존한다."""

        httpx_logger = logging.getLogger("httpx")
        # 세팅 전 기본 NOTSET(0) 또는 외부 영향 — 이후 WARNING(30) 이 되는지만 검증
        setup_logging()

        assert httpx_logger.level == logging.WARNING
        # 동일 사실이 다른 downgrade 대상에도 적용되는지 한 경로 더 확인
        assert logging.getLogger("transformers").level == logging.WARNING

    def test_invokes_disable_non_rank0_progress_on_default_setup(self):
        """기본 모드(``rank0_only=True``, ``verbose=False``) 의 ``setup_logging``
        은 ``disable_non_rank0_progress`` 를 한 번 호출해야 한다.

        U6 의 핵심 계약: 4-rank DDP 환경에서 non-rank-0 프로세스의 HF tqdm
        bar 를 꺼주기 위해 setup 이 별도 호출 없이 이 helper 를 자동 실행한다.
        다른 Unit 의 진입 경로(``_torchrun_entry`` 등) 가 추가 코드를 넣지 않고도
        동작해야 하므로 통합 시점을 테스트에서 고정한다."""

        with mock.patch.object(
            mdp_logging, "disable_non_rank0_progress"
        ) as m:
            setup_logging()

        assert m.call_count == 1

    def test_verbose_skips_disable_progress(self):
        """``verbose=True`` 이면 setup 전체가 no-op 이 되어야 하므로
        ``disable_non_rank0_progress`` 역시 호출되지 않아야 한다.

        verbose 는 디버깅 모드 — 모든 rank 의 tqdm bar 도 그대로 보고 싶어하는
        시나리오이므로 progress bar 를 건드리지 않는 것이 올바른 동작."""

        with mock.patch.object(
            mdp_logging, "disable_non_rank0_progress"
        ) as m:
            setup_logging(verbose=True)

        assert m.call_count == 0


# ──────────────────────────────────────────────────────────────────────────
# Propagation 경로 커버 (cycle 1 review 1-1 회귀 방어)
# ──────────────────────────────────────────────────────────────────────────


class TestRank0FilterOnPropagationPath:
    """Child logger 에서 log 된 레코드가 propagate 경로로 root handler 에 도달할
    때, Rank0Filter 가 실제로 차단되는지 확인.

    cycle 1 review 1-1 근본 원인: Python ``logging`` 의 logger-attached filter
    는 해당 logger 로 **직접 log 된 레코드** 에만 적용된다. 하위 logger 에서
    propagate 된 레코드는 logger.filters 를 건너뛰고 상위 handler 로 흘러간다.
    MDP 전 모듈은 ``logging.getLogger(__name__)`` 패턴을 쓰므로 ``mdp`` logger
    에 filter 를 단 구조 (버그) 로는 rank-0 중복이 실제 운영에서 제거되지 않았다.

    fix 이후 filter 는 root handler 에 부착되므로 propagate 된 레코드도 차단된다.
    본 테스트는 capsys 로 실제 stream 출력을 관찰하여 production-like 동작을
    직접 검증한다 — caplog 는 logger 에 handler 를 직접 달아 propagate 우회가
    가능하므로 신뢰할 수 없다.
    """

    def _prepare_root_stream(self, capsys):
        """basicConfig 대신 직접 StreamHandler 를 root 에 설치해 capsys 로
        관측 가능하게 한다. basicConfig 는 이미 실행됐을 수 있어 멱등하지 않다.
        """
        root = logging.getLogger()
        # 기존 handler 모두 제거하고 sys.stderr 로 흘리는 StreamHandler 1 개 유지.
        old_handlers = list(root.handlers)
        for h in old_handlers:
            root.removeHandler(h)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        root.addHandler(stream_handler)
        root.setLevel(logging.DEBUG)
        return old_handlers, stream_handler

    def _restore_root_handlers(self, old_handlers, stream_handler):
        root = logging.getLogger()
        root.removeHandler(stream_handler)
        for h in old_handlers:
            root.addHandler(h)

    def test_rank0_filter_blocks_propagated_records_from_child_logger(
        self, monkeypatch, capsys
    ):
        """RANK=2 상태에서 setup_logging 호출 후 child logger 에서 log 한 레코드가
        실제 stream 에 남지 않아야 한다.

        이전 버그(logger-attached filter) 에서는 이 테스트가 실패했을 것 —
        "This should be filtered" 이 stderr 에 그대로 출력. handler-level 부착
        으로 전환한 후에는 차단된다.
        """
        monkeypatch.setenv("RANK", "2")
        old, sh = self._prepare_root_stream(capsys)

        try:
            setup_logging()
            child = logging.getLogger("mdp.training.rl_trainer")
            child.info("This should be filtered on non-rank0")
            for handler in logging.getLogger().handlers:
                handler.flush()

            captured = capsys.readouterr()
            assert "This should be filtered on non-rank0" not in captured.err
            assert "This should be filtered on non-rank0" not in captured.out
        finally:
            self._restore_root_handlers(old, sh)

    def test_rank0_filter_passes_extra_all_ranks_through_propagation(
        self, monkeypatch, capsys
    ):
        """escape hatch(``extra={"all_ranks": True}``) 는 non-rank0 에서도
        propagate 경로를 통과해야 한다. U3 FSDP shard baseline 와 U5 OOM per-rank
        summary 가 이 계약에 의존한다."""
        monkeypatch.setenv("RANK", "3")
        old, sh = self._prepare_root_stream(capsys)

        try:
            setup_logging()
            child = logging.getLogger("mdp.training.rl_trainer")
            child.info("All-rank escape hatch record", extra={"all_ranks": True})
            for handler in logging.getLogger().handlers:
                handler.flush()

            captured = capsys.readouterr()
            assert "All-rank escape hatch record" in captured.err
        finally:
            self._restore_root_handlers(old, sh)

    def test_rank0_records_pass_through_propagation(self, monkeypatch, capsys):
        """RANK=0 (또는 미세팅) 에서는 child logger 의 일반 레코드가 모두 통과."""
        monkeypatch.delenv("RANK", raising=False)
        old, sh = self._prepare_root_stream(capsys)

        try:
            setup_logging()
            child = logging.getLogger("mdp.training.rl_trainer")
            child.info("rank-0 normal record")
            for handler in logging.getLogger().handlers:
                handler.flush()

            captured = capsys.readouterr()
            assert "rank-0 normal record" in captured.err
        finally:
            self._restore_root_handlers(old, sh)


# ──────────────────────────────────────────────────────────────────────────
# Args-aware idempotency (cycle 1 review 1-2 회귀 방어)
# ──────────────────────────────────────────────────────────────────────────


class TestSetupLoggingArgsAwareTransition:
    """동일 인자 재호출은 no-op, 인자가 바뀌면 상태 전환.

    cycle 1 review 1-2: CLI 진입부는 env-only 1차 호출 → settings 로드 후 2차
    호출 구조다. 기존 구현은 "첫 호출 후 무조건 no-op" 이어서 recipe
    ``monitoring.verbose=true`` 가 2차 호출에서 무시됐다. fix 이후 인자가
    바뀌면 상태를 재조립하므로 verbose 가 실제 on/off 된다.
    """

    def test_same_args_is_noop(self):
        """동일 인자 두 번째 호출은 filter 를 중복 부착하지 않는다."""
        setup_logging()
        setup_logging()  # 동일 인자
        assert _rank0_filter_is_attached()

    def test_switches_from_non_verbose_to_verbose(self):
        """1차 verbose=False → 2차 verbose=True 호출 시 Rank0Filter 제거 +
        외부 logger level 원복."""
        setup_logging(verbose=False)
        assert _rank0_filter_is_attached()
        assert logging.getLogger("httpx").level == logging.WARNING

        setup_logging(verbose=True)

        assert _rank0_filter_is_absent()
        # NOTSET(0) 으로 복원 — 초기 상태와 동등
        assert logging.getLogger("httpx").level == logging.NOTSET
        assert logging.getLogger("transformers").level == logging.NOTSET

    def test_switches_from_verbose_to_non_verbose(self):
        """1차 verbose=True → 2차 verbose=False 호출 시 Rank0Filter 부착 +
        외부 logger level downgrade 발동."""
        setup_logging(verbose=True)
        assert _rank0_filter_is_absent()

        setup_logging(verbose=False)

        assert _rank0_filter_is_attached()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("transformers").level == logging.WARNING

    def test_env_verbose_overrides_param_across_calls(self, monkeypatch):
        """``MDP_LOG_VERBOSE=1`` 환경변수는 verbose=False 파라미터를 덮어쓴다.
        두 번째 호출도 env 기준이 동일하면 no-op."""
        monkeypatch.setenv("MDP_LOG_VERBOSE", "1")

        setup_logging(verbose=False)  # env 기준 verbose 로 결정
        setup_logging(verbose=False)  # 동일 결정 → no-op

        # verbose 이므로 filter 는 없음
        assert _rank0_filter_is_absent()

    def test_rank0_only_toggle_reconfigures_filter(self):
        """rank0_only=True → rank0_only=False 전환 시 filter 가 제거되어야 한다."""
        setup_logging(rank0_only=True)
        assert _rank0_filter_is_attached()

        setup_logging(rank0_only=False)
        assert _rank0_filter_is_absent()


# ──────────────────────────────────────────────────────────────────────────
# disable_non_rank0_progress (U6)
# ──────────────────────────────────────────────────────────────────────────


class TestDisableNonRank0Progress:
    """HF ``transformers`` tqdm progress bar 의 rank-0 단일화 helper 검증.

    4-rank DDP 환경에서 ``AutoModelForCausalLM.from_pretrained`` 가 찍는
    ``Loading weights: ...`` bar 가 rank 마다 중복 출력되는 문제를, non-rank-0
    프로세스의 bar 만 꺼서 해결한다. 아래 4 케이스는 이 동작의 각 분기를
    커버한다."""

    def test_no_op_on_rank0(self, monkeypatch):
        """``RANK=0`` 또는 미세팅 프로세스에서는 함수가 즉시 반환하고
        ``transformers.utils.logging`` 의 어떤 함수도 호출하지 않아야 한다.

        rank 0 은 최종적으로 bar 를 '보이게' 유지할 프로세스이므로 HF 내부
        상태를 전혀 건드려선 안 된다."""

        monkeypatch.setenv("RANK", "0")

        # transformers.utils.logging.disable_progress_bar 가 호출되는지 감시
        from transformers.utils import logging as hf_logging

        with mock.patch.object(hf_logging, "disable_progress_bar") as m:
            disable_non_rank0_progress()

        assert m.call_count == 0

    def test_calls_disable_on_non_rank0(self, monkeypatch):
        """``RANK=2`` 같은 non-zero rank 에서는 ``transformers.utils.logging``
        의 ``disable_progress_bar`` 가 정확히 1 회 호출되어야 한다.

        이것이 spec §카테고리 A 중 'Loading weights tqdm ×4 중복' 사례의
        직접적 제거 수단."""

        monkeypatch.setenv("RANK", "2")

        from transformers.utils import logging as hf_logging

        with mock.patch.object(hf_logging, "disable_progress_bar") as m:
            disable_non_rank0_progress()

        assert m.call_count == 1

    def test_missing_transformers_graceful(self, monkeypatch):
        """``transformers`` 미설치 환경(경량 실행·일부 CI 이미지 등) 에서도
        예외 없이 silent skip 되어야 한다.

        ``ImportError`` 를 흡수하지 않으면 ``setup_logging`` 초기화 전체가
        무너져 오히려 관측성을 해친다."""

        monkeypatch.setenv("RANK", "1")

        # transformers.utils.logging import 를 ImportError 로 모사.
        # 이미 import 돼 있다면 sys.modules 에서 제거 후 finder 가 실패하도록
        # sys.modules 에 ImportError 유발 sentinel 삽입.
        real_module = sys.modules.get("transformers.utils.logging")
        monkeypatch.setitem(sys.modules, "transformers.utils.logging", None)
        # transformers 패키지 자체도 None 으로 세팅하여 `from transformers...` import
        # 가 ImportError 를 던지게 한다.
        monkeypatch.setitem(sys.modules, "transformers.utils", None)
        monkeypatch.setitem(sys.modules, "transformers", None)

        try:
            # 예외 없이 반환해야 한다
            disable_non_rank0_progress()
        finally:
            # pytest monkeypatch fixture 가 자동 원복하지만, 이후 다른 테스트에
            # 영향을 남기지 않도록 명시적 복원
            if real_module is not None:
                sys.modules["transformers.utils.logging"] = real_module

    def test_old_transformers_no_disable_fn_graceful(self, monkeypatch):
        """``transformers`` 는 설치돼 있으나 ``disable_progress_bar`` API 가
        아직 추가되지 않은 구버전인 경우, ``hasattr`` 분기로 silent skip 되어야
        한다.

        `getattr(hf_logging, 'disable_progress_bar', None) is None` 경로를
        강제로 만들기 위해 속성을 monkeypatch 로 제거한다."""

        monkeypatch.setenv("RANK", "1")

        from transformers.utils import logging as hf_logging

        # 해당 속성만 일시 제거. monkeypatch.delattr 는 teardown 시 원복.
        monkeypatch.delattr(hf_logging, "disable_progress_bar", raising=False)

        # 예외 없이 반환해야 한다 (silent skip)
        disable_non_rank0_progress()


# ──────────────────────────────────────────────────────────────────────────
# WARNING_SUPPRESS_PATTERNS
# ──────────────────────────────────────────────────────────────────────────


class TestWarningSuppressPatterns:
    """화이트리스트 regex 가 정상 컴파일 가능한지 smoke 검증.

    잘못된 regex 가 섞이면 ``warnings.filterwarnings`` 호출 시점에
    ``re.error`` 가 던져져 setup 전체가 실패한다 — 그런 회귀를 test 단에서
    잡아낸다."""

    def test_patterns_are_regex_valid(self):
        assert len(WARNING_SUPPRESS_PATTERNS) > 0
        for pattern in WARNING_SUPPRESS_PATTERNS:
            # compile 실패 시 re.error 가 raise 되어 테스트가 붉게 실패
            re.compile(pattern)
