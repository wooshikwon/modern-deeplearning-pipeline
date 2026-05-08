"""LaunchPlan contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from mdp.runtime import launcher
from mdp.runtime.launcher import LaunchPlan, build_launch_plan, run_distributed, run_single
from mdp.settings.distributed import has_distributed_intent
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.run_plan_builder import normalize_callback_configs
from mdp.settings.schema import DistributedConfig
from tests.e2e.conftest import make_test_settings


def _settings(distributed: dict | None = None):
    settings = make_test_settings()
    settings.config.compute.distributed = (
        DistributedConfig(**distributed) if distributed is not None else None
    )
    return settings


def _run_plan(
    distributed: dict | None = None,
    callbacks: list[dict] | None = None,
    distributed_intent: bool | None = None,
):
    settings = _settings(distributed)
    return RunPlan(
        command="train",
        mode="sft",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=normalize_callback_configs(callbacks),
        validation_scope="training",
        distributed_intent=(
            has_distributed_intent(settings)
            if distributed_intent is None
            else distributed_intent
        ),
    )


def test_build_launch_plan_keeps_single_process_without_distributed_intent() -> None:
    run_plan = _run_plan(None)

    plan = build_launch_plan(run_plan, nproc=4)

    assert plan == LaunchPlan(
        run_plan=run_plan,
        nproc=4,
        distributed=False,
    )


def test_build_launch_plan_requires_intent_and_multiple_processes() -> None:
    callbacks = [{"_component_": "EarlyStopping", "patience": 1}]
    run_plan = _run_plan({"strategy": "ddp"}, callbacks=callbacks)

    single_gpu_plan = build_launch_plan(run_plan, nproc=1)
    multi_gpu_plan = build_launch_plan(run_plan, nproc=2)

    assert single_gpu_plan.distributed is False
    assert single_gpu_plan.run_plan.callback_configs[0].to_yaml_dict() == callbacks[0]
    assert multi_gpu_plan == LaunchPlan(
        run_plan=run_plan,
        nproc=2,
        distributed=True,
    )


def test_build_launch_plan_uses_run_plan_intent_when_settings_disagree() -> None:
    settings_intent_plan = _run_plan({"strategy": "ddp"}, distributed_intent=False)
    no_settings_intent_plan = _run_plan(None, distributed_intent=True)

    assert build_launch_plan(settings_intent_plan, nproc=2).distributed is False
    assert build_launch_plan(no_settings_intent_plan, nproc=2).distributed is True


def test_build_launch_plan_detects_gpu_count_when_nproc_is_not_provided(
    monkeypatch,
) -> None:
    run_plan = _run_plan({"strategy": "ddp"})
    monkeypatch.setattr(launcher, "detect_gpu_count", lambda: 3)

    plan = build_launch_plan(run_plan)

    assert plan.nproc == 3
    assert plan.distributed is True


def test_run_single_uses_runtime_training_helper(monkeypatch) -> None:
    run_plan = _run_plan(None)
    captured: dict[str, object] = {}

    def fake_run_training(run_plan_arg, *, callbacks_observer=None):
        captured["run_plan"] = run_plan_arg
        captured["callbacks_observer"] = callbacks_observer
        return {"total_steps": 1}

    monkeypatch.setattr(
        "mdp.runtime.training.run_training",
        fake_run_training,
    )
    observer = object()

    result = run_single(run_plan, callbacks_observer=observer)

    assert result == {"total_steps": 1}
    assert captured["run_plan"] is run_plan
    assert captured["callbacks_observer"] is observer


def test_run_distributed_serializes_run_plan_payload(monkeypatch) -> None:
    callbacks = [{"_component_": "ModelCheckpoint", "monitor": "loss"}]
    run_plan = _run_plan({"strategy": "ddp"}, callbacks=callbacks)
    captured: dict[str, object] = {}

    def fake_run(cmd, check):
        settings_path = Path(cmd[cmd.index("--run-plan-path") + 1])
        captured["check"] = check
        captured["payload"] = json.loads(settings_path.read_text())

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    result = run_distributed(run_plan, nproc=2)

    assert result == {}
    assert captured["check"] is True
    assert captured["payload"]["settings"]["recipe"] == run_plan.settings.model_dump(mode="json")["recipe"]
    assert captured["payload"]["settings"]["config"] == run_plan.settings.model_dump(mode="json")["config"]
    assert captured["payload"]["callback_configs"] == callbacks
