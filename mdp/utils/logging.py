"""MDP system logging setup helper.

본 모듈은 spec-system-logging-cleanup §원칙 1~3의 기반을 제공한다.
소비자(U2 `setup_logging` 내부 external logger level 세팅, U4 배너/step progress,
U5 OOM memory summary)는 다음 공유 인터페이스를 사용한다:

- ``setup_logging(*, verbose, rank0_only, suppress_external) -> None``: 진입점.
  CLI 진입(torchrun_entry / rl_train / train) 또는 recipe driven runner가 MDP
  실행 시작 직후 1회 호출. Idempotent — 동일 인자로 재호출해도 filter·level·
  warning 필터가 중복되지 않는다. 인자가 바뀐 경로(예: env-only 1차 → settings
  2차)에서는 **상태 전환**으로 다시 조립한다 — verbose on/off 스위칭이 실제로
  발효되어야 recipe.monitoring.verbose 가 의미를 갖는다.
- ``Rank0Filter``: ``logging.Filter`` 서브클래스. distributed 환경 ``RANK``
  환경변수를 기준으로 rank 0 프로세스만 통과시키고, 그 외 rank는 차단한다.
  rank별 정보가 꼭 필요한 경로는 ``logger.info(..., extra={"all_ranks": True})``
  로 escape hatch 사용. **부착 대상은 root logger 의 각 handler** — child
  logger(`mdp.training.rl_trainer` 등) 의 레코드가 propagate 경로로 root 에
  도달했을 때에만 filter 가 실행되기 때문. logger 자체에 부착하면 직접 log
  된 레코드에만 적용되고 propagate 된 레코드는 통과해버리는 무력 상태가 된다.
- ``WARNING_SUPPRESS_PATTERNS``: ``warnings.filterwarnings`` message regex
  리스트. 현재 화이트리스트는 spec §원칙 3의 초기 대상 2종 (HF의
  ``use_cache=True`` gradient-checkpointing 자동 처리 warning, 404
  ``additional_chat_templates`` 404 notice)으로 출발한다. 항목 추가·제거는
  code review로 관리한다 — blanket ``ignore`` 금지.

실행 순서상 제약:
- ``setup_logging`` 은 HuggingFace ``AutoModelForCausalLM.from_pretrained`` 같은
  외부 logger가 첫 메시지를 내보내기 전에 호출되어야 downgrade 효과가 있다.
  CLI 진입부 최상단 호출이 표준 위치.
"""

from __future__ import annotations

import logging
import os
import warnings

__all__ = [
    "Rank0Filter",
    "WARNING_SUPPRESS_PATTERNS",
    "disable_non_rank0_progress",
    "setup_logging",
]


# ---------------------------------------------------------------------------
# Warning suppress 화이트리스트
# ---------------------------------------------------------------------------
#
# `warnings.filterwarnings(message=...)` 에 전달하는 정규식 패턴 목록.
# `re.match` 계열로 처리되므로 (Python ``warnings`` 내부는 ``re.compile`` 후
# ``match``) 시작 anchor ``^`` 를 명시적으로 포함한다.
#
# 추가 정책:
#   1. 대상 warning 이 HF/PyTorch 가 **자동으로 내부 처리** 하거나, 사용자
#      조치가 불가능하며 매 run 마다 반복되는 noise 일 때만 추가.
#   2. 여전히 실제 오동작을 시사할 수 있는 warning 은 suppress 금지 — 새로운
#      warning 이 섞여 나와 발견 가능해야 한다.
#   3. 추가/삭제는 code review + spec 업데이트 동반.
WARNING_SUPPRESS_PATTERNS: list[str] = [
    # HF 가 `model.gradient_checkpointing_enable()` 후 내부적으로
    # `config.use_cache=False` 로 자동 조정하며 출력하는 안내. 사용자 조치
    # 불필요. 4-rank DDP 환경에서 run 마다 4회 출력되어 로그를 부풀림.
    r"^`use_cache=True` is incompatible with gradient checkpointing",
    # LLaMA-3 Base 등 chat template 이 없는 모델에서 HF hub 가
    # `additional_chat_templates` 404 를 INFO 로 내보내는 경우의 Python
    # warning 경로. httpx INFO 는 logger level downgrade 로 처리되지만
    # warnings 경로로도 새어나올 수 있어 명시 suppress.
    r"^.*additional_chat_templates.*404",
]


# ---------------------------------------------------------------------------
# Rank-0 filter
# ---------------------------------------------------------------------------


class Rank0Filter(logging.Filter):
    """Distributed run 에서 rank 0 만 통과시키는 logging filter.

    ``RANK`` 환경변수는 torchrun / torch.distributed.launch 가 워커마다 주입한다.
    단일 프로세스 실행에서는 ``RANK`` 가 없어 기본값 ``0`` 이 되어 모든 레코드
    통과 — non-distributed 실행의 동작은 기존과 동일.

    rank 별 정보가 꼭 필요한 경로 (예: FSDP shard baseline, OOM per-rank
    memory summary) 는 ``logger.info(msg, extra={"all_ranks": True})`` 로
    escape hatch 를 사용한다. 이 경우 ``LogRecord`` 에 ``all_ranks`` attribute
    가 True 로 박히고 filter 가 통과시킨다.

    부착 위치 계약: ``setup_logging`` 은 본 필터를 **root logger 의 각 handler**
    (또는 rank0_only 모드에서 보장되는 StreamHandler) 에 부착한다. Python
    ``logging`` 은 child logger 의 레코드를 propagate 경로로 root handler 에
    전달하는데, logger 자체의 filters 는 "그 logger 로 직접 기록된 레코드" 에만
    적용되기 때문이다. handler-level 부착이어야 `mdp.training.rl_trainer` 같은
    child logger 에서 발생한 레코드도 rank 체크를 받는다.
    """

    def __init__(self) -> None:
        super().__init__()
        # 생성 시점에 rank 를 캡처해 두어 setup 이후 ``os.environ`` 변조에
        # 영향을 받지 않는다 (CI 테스트의 monkeypatch 관점에서도 결정적).
        self._rank = int(os.environ.get("RANK", "0"))

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if self._rank == 0:
            return True
        # non-zero rank: escape hatch 만 통과
        return bool(getattr(record, "all_ranks", False))


# ---------------------------------------------------------------------------
# HF tqdm progress bar rank-0 단일화 (U6)
# ---------------------------------------------------------------------------


def disable_non_rank0_progress() -> None:
    """Non-rank-0 프로세스에서 HuggingFace ``transformers`` tqdm progress bar를
    비활성화한다.

    ``AutoModelForCausalLM.from_pretrained`` 등 HF 로더가 출력하는
    ``Loading weights: 100%|████|`` 형태의 tqdm progress bar는 4-rank DDP
    환경에서 각 rank마다 동일 bar가 중복 출력돼 stdout을 4배로 부풀린다.
    이 함수는 non-rank-0 프로세스에서 `transformers.utils.logging` 의
    ``disable_progress_bar()`` 를 호출해 bar 출력을 끈다. rank 0 은 그대로
    유지되어 최종적으로 bar 1 개만 남는다.

    동작 조건과 방어:

    - ``RANK`` 환경변수가 ``0`` 이거나 미세팅이면 즉시 반환한다 — rank 0 의
      기존 로그 동작을 변경하지 않는다.
    - ``transformers`` 가 설치돼 있지 않은 환경(테스트 mock / 경량 실행 등)
      에서는 ``ImportError`` 를 흡수하고 silent skip.
    - ``disable_progress_bar`` 는 ``transformers`` 특정 버전에서만 제공되는
      API 이므로 ``hasattr`` 로 가드한다. 구버전이면 silent skip.
    - 호출 실패(예: 내부 상태 이슈) 시 debug 레벨로 기록만 하고 예외는
      삼킨다. 로깅 셋업 도중 전체 흐름을 깨뜨리지 않는다.

    Notes
    -----
    본 함수는 ``setup_logging(rank0_only=True, verbose=False)`` 기본 모드에서
    자동 호출된다. 직접 호출하는 경우는 ``setup_logging`` 을 우회하는
    특수한 진입 경로(예: custom launcher)로 제한한다.
    """

    if int(os.environ.get("RANK", "0")) == 0:
        return

    try:
        from transformers.utils import logging as hf_logging  # type: ignore
    except ImportError:
        # transformers 부재 환경 — silent skip (테스트 fixture 등)
        return

    fn = getattr(hf_logging, "disable_progress_bar", None)
    if fn is None:
        # 구버전 transformers — 이 API 가 아직 없음. silent skip.
        return

    try:
        fn()
    except Exception as e:  # pragma: no cover - defensive
        logging.getLogger("mdp.utils.logging").debug(
            "disable_progress_bar failed: %s", e
        )


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

# Idempotent guard + 마지막 적용 인자. 동일 인자 재호출은 no-op 으로 처리하고,
# 인자가 바뀌면 기존 상태를 해제한 뒤 재적용한다. 이 "arg-aware idempotency" 가
# 있어야 CLI 의 "env-only 1차 → settings 로드 후 2차" 호출 구조에서 recipe
# `monitoring.verbose=true` 가 실제로 효과를 갖는다 (cycle 1 review 1-2 해소).
#
# 테스트에서 reset 하려면 ``_MDP_LOGGING_SETUP_DONE=False`` 로 리셋 후 재호출.
_MDP_LOGGING_SETUP_DONE: bool = False
_MDP_LAST_SETUP_ARGS: tuple[bool, bool, bool] | None = None

# 외부 라이브러리 logger 중 INFO → WARNING 으로 downgrade 대상. U2 가 소비.
# HF 생태계 + httpx + urllib3 의 루틴 HTTP 요청 로그를 조용히 만드는 것이
# 1차 목적. 추가 대상이 생기면 spec 에 기재 후 이 리스트에 append.
_EXTERNAL_LOGGERS_TO_DOWNGRADE: tuple[str, ...] = (
    "httpx",
    "urllib3",
    "transformers",
    "datasets",
)


def _remove_rank0_filters_from_root_handlers() -> None:
    """Root logger 의 모든 handler 에서 ``Rank0Filter`` 인스턴스를 제거한다.

    verbose 전환 또는 rank0_only=False 재설정 시 이전 setup 이 붙였던 filter
    를 깨끗이 떼어낸다. 사용자가 별도로 붙인 filter 는 건드리지 않는다.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        handler.filters[:] = [
            f for f in handler.filters if not isinstance(f, Rank0Filter)
        ]


def _restore_external_logger_levels() -> None:
    """외부 logger level 을 NOTSET(0) 으로 되돌려 root 의 level 결정에 맡긴다.

    verbose 전환 시, 기존 WARNING downgrade 를 해제해 INFO 요청 로그 등이 다시
    보이도록 한다. WARNING 설정이 없었던 초기 상태와 동일하지는 않지만 (NOTSET
    vs 미지정), Python logging 의 level 승계 규칙 상 동등하게 동작한다.
    """
    for name in _EXTERNAL_LOGGERS_TO_DOWNGRADE:
        logging.getLogger(name).setLevel(logging.NOTSET)


def setup_logging(
    *,
    verbose: bool = False,
    rank0_only: bool = True,
    suppress_external: bool = True,
) -> None:
    """MDP 진입 시 호출되는 통합 logging setup.

    Parameters
    ----------
    verbose:
        ``True`` 이면 rank-0 filter 부착 · 외부 logger downgrade · warning
        suppress 를 모두 건너뛰고 Python 기본 logging 상태로 남긴다. 디버깅
        전용. ``MDP_LOG_VERBOSE=1`` 환경변수가 세팅되어 있으면 파라미터로 전달한
        값과 무관하게 verbose 로 취급한다 — 운영자가 CLI 인자를 고치지 않고도
        조용함을 off 할 수 있도록 한다.
    rank0_only:
        ``Rank0Filter`` 를 root logger 의 각 handler 에 부착할지 여부.
    suppress_external:
        ``httpx`` / ``urllib3`` / ``transformers`` / ``datasets`` logger 의
        level 을 ``WARNING`` 으로 올릴지 여부.

    Notes
    -----
    Args-aware idempotency: 동일 인자로 재호출 시 즉시 return (no-op). 인자가
    바뀌면 기존 상태(Rank0Filter 부착·외부 logger level) 를 해제한 뒤 새 인자로
    재조립한다. 이 규칙 덕에 CLI 의 "env-only 1차 호출 → settings 2차 호출"
    구조에서 recipe ``monitoring.verbose=true`` 가 2차 호출 시 실제 적용된다.

    계약 (spec §U1):
        - Root logger 의 각 handler 에 부착하는 ``Rank0Filter`` 는 정확히 1 개.
          중복 부착 시 rank 검사가 여러 번 돌아 성능·로그 동작이 어긋난다.
        - ``extra={"all_ranks": True}`` escape hatch 는 U3/U5 가 FSDP shard
          baseline · OOM summary 에서 사용. 이 키 네이밍은 변경 금지.
    """

    global _MDP_LOGGING_SETUP_DONE, _MDP_LAST_SETUP_ARGS

    env_verbose = os.environ.get("MDP_LOG_VERBOSE", "0") == "1"
    effective_verbose = verbose or env_verbose
    args_key = (effective_verbose, rank0_only, suppress_external)

    # 동일 인자 재호출 → 진짜 no-op (spec §U1 idempotency 계약)
    if _MDP_LOGGING_SETUP_DONE and _MDP_LAST_SETUP_ARGS == args_key:
        return

    # 재설정: 이전 호출이 붙였던 Rank0Filter 제거 + 외부 logger level 복원.
    # 처음 호출이면 둘 다 no-op (기존 filter 없음, level 도 이미 NOTSET).
    _remove_rank0_filters_from_root_handlers()
    _restore_external_logger_levels()

    if effective_verbose:
        # verbose 모드: filter·level·warning suppress 없이 완전 투명. 단 flag 는
        # 세팅하여 동일 인자 재호출이 no-op 이 되도록 한다.
        _MDP_LOGGING_SETUP_DONE = True
        _MDP_LAST_SETUP_ARGS = args_key
        return

    # 1) Root logger handler 에 Rank0Filter 부착 (propagate 경로 커버)
    if rank0_only:
        root = logging.getLogger()
        if not root.handlers:
            # basicConfig 가 아직 호출되지 않아 root 에 handler 가 없는 경우.
            # 이 경로는 CLI 진입 순서상 드물지만, setup_logging 을 단독 호출하는
            # 테스트·스크립트 환경에서 filter 가 실제로 작동하도록 stream
            # handler 를 1개 설치한다. 부착 후에는 basicConfig 가 후에 호출돼도
            # 동일 handler 가 유지된다.
            root.addHandler(logging.StreamHandler())

        filter_instance = Rank0Filter()
        for handler in root.handlers:
            # 중복 부착 방지: 방금 모든 Rank0Filter 를 제거했지만 외부 코드가
            # 직접 부착했을 수 있으므로 방어.
            if not any(isinstance(f, Rank0Filter) for f in handler.filters):
                handler.addFilter(filter_instance)

    # 2) 외부 logger level downgrade
    if suppress_external:
        for name in _EXTERNAL_LOGGERS_TO_DOWNGRADE:
            logging.getLogger(name).setLevel(logging.WARNING)

    # 3) Warning suppress 화이트리스트 적용
    for pattern in WARNING_SUPPRESS_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern)

    # 4) Non-rank-0 프로세스의 HF tqdm progress bar 비활성 (U6)
    #    rank0_only 가 켜진 기본 모드에서만 자동 호출한다 — 사용자가 명시적으로
    #    ``rank0_only=False`` 로 호출한 경우엔 rank 별 정보를 보고 싶어하는 것으로
    #    간주하여 progress bar 도 건드리지 않는다.
    if rank0_only:
        disable_non_rank0_progress()

    _MDP_LOGGING_SETUP_DONE = True
    _MDP_LAST_SETUP_ARGS = args_key
