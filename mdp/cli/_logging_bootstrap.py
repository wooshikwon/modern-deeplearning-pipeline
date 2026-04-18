"""CLI-레벨 logging bootstrap helper.

spec-system-logging-cleanup §U2 의 소비 지점을 하나의 함수로 집약한다.
본 모듈은 의도적으로 얇게 유지한다:

- 실제 filter 설치 · 외부 logger level 조정 · warning suppress 화이트리스트는
  `mdp.utils.logging.setup_logging()` 이 전담한다. 본 helper 는 "CLI 진입 시점에
  어떤 파라미터로 setup_logging 을 불러야 하는가" 만 결정한다.
- verbose 결정 소스는 두 축:
    1) 환경변수 ``MDP_LOG_VERBOSE=1`` — CLI 재기동 없이 조용함을 off.
    2) Recipe ``monitoring.verbose`` — run 단위 on/off. spec §U4 가 MonitoringSpec
       에 ``verbose`` 필드를 추가한 이후 동작. 병렬 진행 상황에서도 회귀가
       없도록 ``hasattr`` 기반 우아한 fallback 을 쓴다.
- ``setup_logging`` 이 idempotent 라 본 helper 도 여러 CLI 경로에서 반복 호출
  되어도 상태 중복이 발생하지 않는다 (CLI entry 에서 env-only 1 회 + settings
  로드 후 1 회 호출 구조를 전제).

호출 위치:
- ``mdp/cli/_torchrun_entry.py::main()`` — argparse · settings 로드 전에 env-only,
  settings 로드 후 최종. HF ``from_pretrained`` 첫 호출 이전에 반드시 완료.
- ``mdp/cli/train.py::run_train()`` / ``mdp/cli/rl_train.py::run_rl_train()`` —
  settings 로드 직후, Factory 호출 · apply_liger_patches 전후. Liger patch 는
  `setup_logging` 과 독립이므로 순서는 두 가지 모두 허용되지만, 본 helper 는
  "settings 가 있으면 즉시 호출" 정책으로 통일한다.
"""

from __future__ import annotations

from typing import Any

from mdp.utils.logging import setup_logging

__all__ = ["bootstrap_logging", "resolve_verbose"]


def resolve_verbose(settings: Any | None = None) -> bool:
    """Recipe ``monitoring.verbose`` 와 환경변수 ``MDP_LOG_VERBOSE`` 를 OR 결합.

    spec §원칙 2: "기본값은 조용하게". verbose 는 디버깅·운영자 복원 용도.

    Parameters
    ----------
    settings:
        ``Settings`` 객체 (None 이면 env-only). ``settings.recipe.monitoring.verbose``
        를 ``hasattr`` 로 안전 접근 — U4 가 MonitoringSpec 에 ``verbose`` 필드를
        추가하기 전에도, monitoring 섹션이 None 이어도 안전.

    Notes
    -----
    ``setup_logging`` 내부에도 동일한 env 검사가 있어 기본값 시나리오는 이중
    안전. 다만 recipe 기반 verbose 는 본 helper 만이 알기 때문에, 결과적으로
    "recipe 또는 env 중 하나라도 verbose" 판정을 단일 진입점에서 내려준다.
    """
    import os

    env_verbose = os.environ.get("MDP_LOG_VERBOSE", "0") == "1"

    recipe_verbose = False
    if settings is not None:
        # settings.recipe 가 없거나 monitoring 이 None 인 경우 모두 False.
        monitoring = getattr(getattr(settings, "recipe", None), "monitoring", None)
        # U4 가 MonitoringSpec 에 verbose 필드를 추가하기 전/후 모두 동작.
        recipe_verbose = bool(getattr(monitoring, "verbose", False))

    return env_verbose or recipe_verbose


def bootstrap_logging(settings: Any | None = None) -> None:
    """CLI 진입 시점의 통합 logging setup.

    ``setup_logging`` 은 idempotent — 본 함수도 CLI entry 하나에서 settings 로드
    전/후 두 번 호출해도 중복 filter · level · warning filter 가 적용되지 않는다.
    두 번째 호출은 첫 호출 시점의 verbose 결정에 고정된다 (플래그 guard).

    Parameters
    ----------
    settings:
        None 이면 환경변수만으로 verbose 판단. Settings 가 준비된 이후 호출이
        권장되지만, HF ``from_pretrained`` 첫 요청 이전이라는 순서 제약을
        만족하기 위해 초기 env-only 호출도 함께 사용할 수 있다.
    """
    setup_logging(verbose=resolve_verbose(settings))
