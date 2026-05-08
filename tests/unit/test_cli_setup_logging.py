"""U2 — CLI 진입에서 ``setup_logging`` 호출 · verbose 결정 · idempotency 검증.

spec-system-logging-cleanup §U2 의 Verify 기준:

- CLI entry (``mdp/cli/_torchrun_entry.py::main``, ``mdp/cli/train.py::run_train``,
  ``mdp/cli/rl_train.py::run_rl_train``) 에서 HF ``from_pretrained`` 첫 호출
  이전에 ``setup_logging`` 이 반드시 호출된다.
- ``MDP_LOG_VERBOSE=1`` env 가 있으면 ``verbose=True`` 로 호출된다.
- ``settings.recipe.monitoring.verbose=True`` 가 있으면 ``verbose=True`` 로 호출
  된다. U4 가 MonitoringSpec 에 ``verbose`` 필드를 추가하기 전이라 ``hasattr``
  fallback 이 동작해야 한다 (현재 MonitoringSpec 에는 ``verbose`` 필드 없음).
- 여러 CLI 경로에서 ``bootstrap_logging`` 이 반복 호출되어도 root handler 의
  Rank0Filter 수 · 외부 logger level · warning filter 가 중복 재설정되지
  않는다. 단 cycle 1 review 1-2 fix 이후: 첫 호출 verbose 결정이 바뀌면 (예:
  1차 env-only False → 2차 settings.verbose=True) 상태 전환이 적용된다.

테스트는 외부 의존(HF·AssemblyMaterializer·Settings 파싱)을 피하기 위해 ``setup_logging``
을 ``monkeypatch`` 로 스텁하여 "어떤 인자로 호출되었는가" 만 검증한다.
실제 filter / level 동작은 U1 테스트(``tests/unit/test_logging_setup.py``) 가
이미 커버한다.
"""

from __future__ import annotations

import logging
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from mdp.cli import _logging_bootstrap
from mdp.cli._logging_bootstrap import bootstrap_logging, resolve_verbose
from mdp.utils import logging as mdp_logging
from mdp.utils.logging import Rank0Filter


# ──────────────────────────────────────────────────────────────────────────
# 가짜 Settings: MonitoringSpec 유무 / verbose 필드 유무 시나리오 재현
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeMonitoringWithVerbose:
    """U4 완료 후의 MonitoringSpec 모사 — ``verbose`` 필드 보유."""

    verbose: bool = False
    enabled: bool = False


@dataclass
class _FakeMonitoringNoVerbose:
    """U4 완료 전의 MonitoringSpec 모사 — ``verbose`` 필드 없음.

    ``hasattr(monitoring, "verbose")`` 가 False 인 상황에서도
    ``resolve_verbose`` 가 안전하게 False 를 반환해야 한다 — 이것이
    "hasattr 기반 우아한 fallback" 계약의 핵심.
    """

    enabled: bool = False


@dataclass
class _FakeRecipe:
    monitoring: Any = None


@dataclass
class _FakeSettings:
    recipe: _FakeRecipe = field(default_factory=_FakeRecipe)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _rank0_filter_is_attached() -> bool:
    """Root handler 중 하나 이상에 Rank0Filter 가 부착되어 있고, 어느 handler
    에도 2 개 이상 중복 부착은 없을 때 True.
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
    """U1 테스트와 동일한 원복 정책. module-level 플래그 + root handler 의
    Rank0Filter + 외부 logger level + 환경변수를 깨끗하게 되돌려 각 테스트 간
    격리를 보장한다.
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
def captured_setup_calls(monkeypatch):
    """``setup_logging`` 호출 인자를 수집한다.

    ``_logging_bootstrap`` 모듈이 참조하는 이름 공간의 ``setup_logging`` 을
    스텁으로 교체. 실제 filter / level 재설정은 건너뛰고 "어떤 verbose 값으로
    호출되었는가" 만 기록.
    """
    calls: list[dict[str, Any]] = []

    def _stub(*, verbose: bool = False, **kwargs: Any) -> None:
        calls.append({"verbose": verbose, **kwargs})

    monkeypatch.setattr(_logging_bootstrap, "setup_logging", _stub)
    return calls


# ──────────────────────────────────────────────────────────────────────────
# resolve_verbose
# ──────────────────────────────────────────────────────────────────────────


class TestResolveVerbose:
    """verbose 결정 소스 3 종 (env / recipe / 없음) 의 결합 검증."""

    def test_no_settings_no_env_returns_false(self):
        assert resolve_verbose(None) is False

    def test_env_only_returns_true(self, monkeypatch):
        monkeypatch.setenv("MDP_LOG_VERBOSE", "1")
        assert resolve_verbose(None) is True

    def test_env_any_other_value_returns_false(self, monkeypatch):
        """spec §U1 계약: ``"1"`` 만이 verbose 활성. ``"true"`` · ``"0"`` 은 조용함."""
        monkeypatch.setenv("MDP_LOG_VERBOSE", "true")
        assert resolve_verbose(None) is False
        monkeypatch.setenv("MDP_LOG_VERBOSE", "0")
        assert resolve_verbose(None) is False

    def test_recipe_verbose_true(self):
        """U4 완료 후 시나리오: MonitoringSpec.verbose=True."""
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringWithVerbose(verbose=True))
        )
        assert resolve_verbose(settings) is True

    def test_recipe_verbose_false(self):
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringWithVerbose(verbose=False))
        )
        assert resolve_verbose(settings) is False

    def test_monitoring_without_verbose_field_fallback_false(self):
        """U4 미완료 시나리오: MonitoringSpec 에 ``verbose`` 속성이 아예
        없음. ``hasattr`` fallback 으로 False 를 반환 — 경계 지시 "hasattr
        기반 우아한 fallback" 계약의 정면 검증."""
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringNoVerbose())
        )
        assert resolve_verbose(settings) is False

    def test_monitoring_none_fallback_false(self):
        """settings.recipe.monitoring 이 None — 아주 얇은 recipe 경로."""
        settings = _FakeSettings(recipe=_FakeRecipe(monitoring=None))
        assert resolve_verbose(settings) is False

    def test_env_wins_even_if_recipe_false(self, monkeypatch):
        """env verbose 와 recipe verbose 가 OR 로 합성되는지. 운영자가 CLI
        재기동 없이 env 만으로 verbose 로 전환 가능해야 한다 (spec §원칙 2)."""
        monkeypatch.setenv("MDP_LOG_VERBOSE", "1")
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringWithVerbose(verbose=False))
        )
        assert resolve_verbose(settings) is True


# ──────────────────────────────────────────────────────────────────────────
# bootstrap_logging
# ──────────────────────────────────────────────────────────────────────────


class TestBootstrapLogging:
    """bootstrap_logging 이 resolve_verbose 결과를 setup_logging 에
    올바르게 전달하는지, 그리고 반복 호출에서 idempotency 를 해치지 않는지."""

    def test_bootstrap_no_settings_env_verbose(self, monkeypatch, captured_setup_calls):
        monkeypatch.setenv("MDP_LOG_VERBOSE", "1")
        bootstrap_logging()
        assert captured_setup_calls == [{"verbose": True}]

    def test_bootstrap_no_settings_quiet_default(self, captured_setup_calls):
        bootstrap_logging()
        assert captured_setup_calls == [{"verbose": False}]

    def test_bootstrap_with_settings_recipe_verbose(self, captured_setup_calls):
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringWithVerbose(verbose=True))
        )
        bootstrap_logging(settings)
        assert captured_setup_calls == [{"verbose": True}]

    def test_bootstrap_with_settings_monitoring_no_verbose_attr(
        self, captured_setup_calls
    ):
        """U4 미완료 시점: MonitoringSpec 에 verbose 필드 없음. fallback 으로
        verbose=False 전달."""
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringNoVerbose())
        )
        bootstrap_logging(settings)
        assert captured_setup_calls == [{"verbose": False}]


# ──────────────────────────────────────────────────────────────────────────
# 실제 setup_logging 을 통한 idempotency (스텁 없이)
# ──────────────────────────────────────────────────────────────────────────


class TestBootstrapIdempotency:
    """여러 CLI 경로에서 ``bootstrap_logging`` 이 거듭 호출되어도 mdp logger
    의 Rank0Filter 개수와 외부 logger level 이 중복 재설정되지 않는다.

    실제 ``setup_logging`` 을 그대로 호출하여 U1 · U2 통합 동작을 관찰한다.
    """

    def test_multiple_bootstraps_single_filter(self):
        bootstrap_logging()
        bootstrap_logging()
        settings = _FakeSettings(recipe=_FakeRecipe(monitoring=None))
        bootstrap_logging(settings)

        # 세 호출 모두 verbose=False 결정 → 동일 인자 → no-op. 정확히 1 개.
        assert _rank0_filter_is_attached()

    def test_multiple_bootstraps_external_level_stable(self):
        """외부 logger level 도 WARNING 으로 한 번 올라간 뒤 안정 유지."""
        bootstrap_logging()
        bootstrap_logging()
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("transformers").level == logging.WARNING

    def test_settings_verbose_true_switches_to_verbose_mode(self, monkeypatch):
        """cycle 1 review 1-2 회귀 방어: 첫 호출이 env-only quiet 이더라도
        2차 호출에서 ``settings.recipe.monitoring.verbose=True`` 가 들어오면
        Rank0Filter 가 제거되고 외부 logger level 이 원복되어야 한다.

        이전 구현 (첫 호출 후 무조건 no-op) 은 recipe 기반 verbose 를 무력화
        했지만, args-aware idempotency 로 전환된 후에는 실제 상태 전환이
        일어난다. CLI 의 "env-only 1차 → settings 2차" 호출 구조가 의도대로
        작동하려면 이 전환이 필수."""
        # 1차: env-only, verbose=False
        bootstrap_logging()
        assert _rank0_filter_is_attached()
        assert logging.getLogger("httpx").level == logging.WARNING

        # 2차: recipe.monitoring.verbose=True → verbose 로 전환
        settings = _FakeSettings(
            recipe=_FakeRecipe(monitoring=_FakeMonitoringWithVerbose(verbose=True))
        )
        bootstrap_logging(settings)

        assert _rank0_filter_is_absent()
        # 외부 logger level 도 NOTSET 으로 원복 (verbose 디버깅 모드)
        assert logging.getLogger("httpx").level == logging.NOTSET


# ──────────────────────────────────────────────────────────────────────────
# CLI entry import smoke — bootstrap_logging 참조가 실제로 연결되어 있는가
# ──────────────────────────────────────────────────────────────────────────


class TestCLIEntryBootstrapWiring:
    """CLI 진입 함수가 ``bootstrap_logging`` 을 실제로 호출하는지 import +
    monkeypatch 스파이로 확인. 실제 학습을 돌리지 않기 위해 heavy 의존은
    스텁으로 차단하고, bootstrap_logging 호출 횟수만 관측한다.
    """

    def test_run_train_invokes_bootstrap(self, monkeypatch):
        """``run_train`` 이 settings 로드 전/후 총 2 회 bootstrap_logging 을
        호출하는지 (최소 1 회 이상)."""
        from mdp.cli import train as train_mod

        bootstrap_calls: list[Any] = []

        def _spy(settings: Any | None = None) -> None:
            bootstrap_calls.append(settings)

        monkeypatch.setattr(train_mod, "bootstrap_logging", _spy, raising=False)

        # Settings loading은 존재하지 않는 YAML 경로에서 실패한다. 이 테스트는
        # 그 전에 env-only bootstrap이 호출되는지만 검증한다.
        # apply_liger_patches 는 부작용만 있고 실제 HF 로드 이전이라 통과시켜도
        # 무방하지만, import 시간을 줄이기 위해 no-op 으로 대체.
        import mdp._liger_patch as liger_patch

        monkeypatch.setattr(liger_patch, "apply_liger_patches", lambda: None)

        # run_train 내부는 `from mdp.cli._logging_bootstrap import bootstrap_logging`
        # 로 이름을 가져오므로, 원 모듈의 함수를 교체해야 한다.
        from mdp.cli import _logging_bootstrap as lb_mod

        monkeypatch.setattr(lb_mod, "bootstrap_logging", _spy)

        import typer

        with pytest.raises(typer.Exit):
            train_mod.run_train(
                recipe_path="nonexistent.yaml",
                config_path="nonexistent.yaml",
            )

        # Settings 로드 이전 env-only 1 회는 반드시 호출되어야 한다. Settings
        # 로드 실패 경로라 두 번째 호출까지는 도달하지 않지만, 최소 1 회 호출
        # 사실로 "HF from_pretrained 이전에 bootstrap 된다" 는 계약을 증명.
        assert len(bootstrap_calls) >= 1
        assert bootstrap_calls[0] is None  # 첫 호출은 env-only (settings=None)

    def test_run_rl_train_invokes_bootstrap(self, monkeypatch):
        from mdp.cli import rl_train as rl_train_mod

        bootstrap_calls: list[Any] = []

        def _spy(settings: Any | None = None) -> None:
            bootstrap_calls.append(settings)

        import mdp._liger_patch as liger_patch

        monkeypatch.setattr(liger_patch, "apply_liger_patches", lambda: None)

        from mdp.cli import _logging_bootstrap as lb_mod

        monkeypatch.setattr(lb_mod, "bootstrap_logging", _spy)

        import typer

        with pytest.raises(typer.Exit):
            rl_train_mod.run_rl_train(
                recipe_path="nonexistent.yaml",
                config_path="nonexistent.yaml",
            )

        assert len(bootstrap_calls) >= 1
        assert bootstrap_calls[0] is None

    def test_run_rl_train_consumes_typed_algorithm_spec(self, monkeypatch):
        """정상 RL settings 로드 후 algorithm 표시가 dict API로 회귀하지 않는다."""
        from mdp.cli import rl_train as rl_train_mod

        import mdp._liger_patch as liger_patch

        monkeypatch.setattr(liger_patch, "apply_liger_patches", lambda: None)
        monkeypatch.setattr(rl_train_mod, "_detect_gpu_count", lambda: 0)
        monkeypatch.setattr(
            rl_train_mod,
            "_run_single",
            lambda run_plan, callbacks_observer=None: {
                "total_steps": 1,
                "metrics": {"loss": 0.0},
            },
        )

        repo_root = Path(__file__).resolve().parents[2]
        rl_train_mod.run_rl_train(
            recipe_path=str(repo_root / "tests/fixtures/recipes/gpt2-dpo-preference.yaml"),
            config_path=str(repo_root / "tests/fixtures/configs/local-cpu.yaml"),
        )

    def test_torchrun_worker_bootstrap_order_before_runtime(self, tmp_path, monkeypatch):
        """torchrun worker는 env bootstrap 후 settings bootstrap과 dist init을 거친다."""
        from mdp.cli import _logging_bootstrap
        from mdp.cli import _torchrun_entry
        from tests.e2e.conftest import make_test_settings

        settings = make_test_settings()
        run_plan_path = Path(tmp_path) / "run_plan.json"
        from mdp.runtime.payload import RunPlanPayload
        from mdp.settings.run_plan import RunPlan, RunSources

        run_plan = RunPlan(
            command="train",
            mode="sft",
            settings=settings,
            sources=RunSources(),
            overrides=(),
            callback_configs=(),
            validation_scope="training",
            distributed_intent=False,
        )
        run_plan_path.write_text(
            json.dumps(RunPlanPayload.from_run_plan(run_plan).to_json_dict(), default=str)
        )
        events: list[str] = []

        def _bootstrap(settings_arg: Any | None = None) -> None:
            events.append("bootstrap-settings" if settings_arg is not None else "bootstrap-env")

        def _init_dist(settings_arg: Any) -> None:
            assert settings_arg.recipe.name == settings.recipe.name
            events.append("dist-init")

        def _run_training(run_plan_arg: Any) -> dict:
            assert run_plan_arg.settings.recipe.name == settings.recipe.name
            events.append("run-training")
            return {}

        monkeypatch.setattr(_logging_bootstrap, "bootstrap_logging", _bootstrap)
        monkeypatch.setattr(_torchrun_entry, "_init_distributed_if_torchrun", _init_dist)
        monkeypatch.setattr(_torchrun_entry, "run_training", _run_training)
        monkeypatch.setattr(
            sys,
            "argv",
            ["_torchrun_entry.py", "--run-plan-path", str(run_plan_path)],
        )

        _torchrun_entry.main()

        assert events == [
            "bootstrap-env",
            "bootstrap-settings",
            "dist-init",
            "run-training",
        ]
