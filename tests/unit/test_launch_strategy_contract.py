"""Distributed-intent unit tests."""

from __future__ import annotations

from unittest import mock

from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.planner import AssemblyPlanner
from mdp.settings.distributed import has_distributed_intent, should_launch_distributed
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.schema import DistributedConfig
from mdp.training._common import create_strategy
from tests.e2e.conftest import make_test_settings


def _settings(distributed: dict | None = None):
    settings = make_test_settings()
    settings.config.compute.distributed = distributed
    return settings


def _materializer(settings) -> AssemblyMaterializer:
    run_plan = RunPlan(
        command="train",
        mode="sft",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=(),
        validation_scope="training",
        distributed_intent=has_distributed_intent(settings),
    )
    return AssemblyMaterializer(AssemblyPlanner.from_run_plan(run_plan))


def test_has_distributed_intent_requires_distributed_strategy() -> None:
    assert has_distributed_intent(_settings(None)) is False
    assert has_distributed_intent(_settings({"strategy": "none"})) is False
    assert has_distributed_intent(_settings({"strategy": "ddp"})) is True
    assert has_distributed_intent(_settings({"strategy": {"_component_": "DDPStrategy"}})) is True


def test_should_launch_distributed_ignores_gpu_count_without_intent() -> None:
    assert should_launch_distributed(_settings(None), detected_gpu_count=2) is False
    assert should_launch_distributed(_settings({"strategy": "none"}), detected_gpu_count=2) is False
    assert should_launch_distributed(_settings({"strategy": "ddp"}), detected_gpu_count=1) is False
    assert should_launch_distributed(_settings({"strategy": "ddp"}), detected_gpu_count=2) is True


def test_materializer_create_dataloaders_uses_same_distributed_intent() -> None:
    settings = _settings({"strategy": "none"})

    with mock.patch("mdp.data.dataloader.create_dataloaders", return_value={}) as create_dataloaders:
        _materializer(settings).materialize_dataloaders()

    assert create_dataloaders.call_args.kwargs["distributed"] is False

    settings = _settings({"strategy": "ddp"})
    with mock.patch("mdp.data.dataloader.create_dataloaders", return_value={}) as create_dataloaders:
        _materializer(settings).materialize_dataloaders()

    assert create_dataloaders.call_args.kwargs["distributed"] is True


def test_create_strategy_merges_top_level_kwargs_into_component_strategy() -> None:
    settings = _settings(None)
    settings.config.compute.distributed = DistributedConfig(
        strategy={"_component_": "DDPStrategy"},
        backend="gloo",
    )

    strategy = create_strategy(settings, _materializer(settings).resolver)

    assert strategy.backend == "gloo"
