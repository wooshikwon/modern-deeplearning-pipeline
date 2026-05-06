"""Launch and DataLoader distributed-intent contract tests."""

from __future__ import annotations

from unittest import mock

from mdp.factory.factory import Factory
from mdp.settings.schema import RLSpec
from mdp.settings.distributed import has_distributed_intent, should_launch_distributed
from tests.e2e.conftest import make_test_settings


def _settings(distributed: dict | None = None):
    settings = make_test_settings()
    settings.config.compute.distributed = distributed
    return settings


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


def test_train_cli_uses_single_path_when_multi_gpu_has_no_distributed_intent(monkeypatch) -> None:
    import mdp.cli.train as train_mod

    settings = _settings(None)
    captured: dict[str, bool] = {}

    monkeypatch.setattr(train_mod, "_detect_gpu_count", lambda: 2)
    def fake_single(*args, **kwargs):
        captured["single"] = True
        return {}

    def fake_distributed(*args, **kwargs):
        captured["distributed"] = True
        return {}

    monkeypatch.setattr(train_mod, "_run_single", fake_single)
    monkeypatch.setattr(train_mod, "_run_distributed", fake_distributed)

    with mock.patch("mdp._liger_patch.apply_liger_patches"), \
         mock.patch("mdp.cli._logging_bootstrap.bootstrap_logging"), \
         mock.patch("mdp.settings.factory.SettingsFactory") as mock_factory:
        mock_factory.return_value.for_training.return_value = settings
        train_mod.run_train("recipe.yaml", "config.yaml")

    assert captured == {"single": True}


def test_train_cli_uses_distributed_path_when_intent_and_multi_gpu(monkeypatch) -> None:
    import mdp.cli.train as train_mod

    settings = _settings({"strategy": "ddp"})
    captured: dict[str, bool] = {}

    monkeypatch.setattr(train_mod, "_detect_gpu_count", lambda: 2)
    def fake_single(*args, **kwargs):
        captured["single"] = True
        return {}

    def fake_distributed(*args, **kwargs):
        captured["distributed"] = True
        return {}

    monkeypatch.setattr(train_mod, "_run_single", fake_single)
    monkeypatch.setattr(train_mod, "_run_distributed", fake_distributed)

    with mock.patch("mdp._liger_patch.apply_liger_patches"), \
         mock.patch("mdp.cli._logging_bootstrap.bootstrap_logging"), \
         mock.patch("mdp.settings.factory.SettingsFactory") as mock_factory:
        mock_factory.return_value.for_training.return_value = settings
        train_mod.run_train("recipe.yaml", "config.yaml")

    assert captured == {"distributed": True}


def test_rl_train_cli_uses_same_distributed_launch_contract(monkeypatch) -> None:
    import mdp.cli.rl_train as rl_train_mod

    settings = _settings({"strategy": "none"})
    settings.recipe.rl = RLSpec(
        algorithm={"_component_": "PPO"},
        models={"policy": {"_component_": "tests.e2e.models.TinyVisionModel", "optimizer": {"_component_": "AdamW"}}},
    )
    captured: dict[str, bool] = {}

    monkeypatch.setattr(rl_train_mod, "_detect_gpu_count", lambda: 2)
    def fake_single(*args, **kwargs):
        captured["single"] = True
        return {}

    def fake_distributed(*args, **kwargs):
        captured["distributed"] = True
        return {}

    monkeypatch.setattr(rl_train_mod, "_run_single", fake_single)
    monkeypatch.setattr(rl_train_mod, "_run_distributed", fake_distributed)

    with mock.patch("mdp._liger_patch.apply_liger_patches"), \
         mock.patch("mdp.cli._logging_bootstrap.bootstrap_logging"), \
         mock.patch("mdp.settings.factory.SettingsFactory") as mock_factory:
        mock_factory.return_value.for_training.return_value = settings
        rl_train_mod.run_rl_train("recipe.yaml", "config.yaml")

    assert captured == {"single": True}


def test_factory_create_dataloaders_uses_same_distributed_intent() -> None:
    settings = _settings({"strategy": "none"})

    with mock.patch("mdp.data.dataloader.create_dataloaders", return_value={}) as create_dataloaders:
        Factory(settings).create_dataloaders()

    assert create_dataloaders.call_args.kwargs["distributed"] is False

    settings = _settings({"strategy": "ddp"})
    with mock.patch("mdp.data.dataloader.create_dataloaders", return_value={}) as create_dataloaders:
        Factory(settings).create_dataloaders()

    assert create_dataloaders.call_args.kwargs["distributed"] is True
