"""--callbacks YAML 파일 로드 유닛 테스트."""

from __future__ import annotations

import pytest
import yaml

from mdp.training._common import load_callbacks_from_file


class TestLoadCallbacksFromFile:
    def test_valid_file(self, tmp_path):
        """정상적인 콜백 YAML 파일을 로드한다."""
        cb_file = tmp_path / "callbacks.yaml"
        data = [
            {"_component_": "EarlyStopping", "patience": 3, "monitor": "val_loss"},
            {"_component_": "ModelCheckpoint", "save_top_k": 2},
        ]
        cb_file.write_text(yaml.dump(data))

        result = load_callbacks_from_file(str(cb_file))
        assert len(result) == 2
        assert result[0]["_component_"] == "EarlyStopping"
        assert result[0]["patience"] == 3
        assert result[1]["_component_"] == "ModelCheckpoint"

    def test_empty_file(self, tmp_path):
        """빈 YAML 파일은 빈 리스트를 반환한다."""
        cb_file = tmp_path / "empty.yaml"
        cb_file.write_text("")

        result = load_callbacks_from_file(str(cb_file))
        assert result == []

    def test_not_a_list_raises(self, tmp_path):
        """YAML이 dict이면 ValueError."""
        cb_file = tmp_path / "bad.yaml"
        cb_file.write_text(yaml.dump({"_component_": "EarlyStopping"}))

        with pytest.raises(ValueError, match="리스트여야"):
            load_callbacks_from_file(str(cb_file))

    def test_missing_component_key_raises(self, tmp_path):
        """항목에 _component_ 키가 없으면 ValueError."""
        cb_file = tmp_path / "bad.yaml"
        cb_file.write_text(yaml.dump([{"patience": 3}]))

        with pytest.raises(ValueError, match="_component_"):
            load_callbacks_from_file(str(cb_file))

    def test_file_not_found_raises(self):
        """존재하지 않는 파일은 FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_callbacks_from_file("/nonexistent/callbacks.yaml")

    def test_single_item(self, tmp_path):
        """단일 콜백도 정상 로드."""
        cb_file = tmp_path / "single.yaml"
        cb_file.write_text(yaml.dump([{"_component_": "ProgressBar"}]))

        result = load_callbacks_from_file(str(cb_file))
        assert len(result) == 1
        assert result[0]["_component_"] == "ProgressBar"


class TestCallbacksInjectToTrainer:
    """--callbacks CLI 경로가 Trainer에 직접 주입되는지 검증 (U3 패턴)."""

    def test_cli_train_passes_cb_configs_to_run_training(self, tmp_path, monkeypatch):
        """run_train()이 load한 cb_configs를 _run_single에 전달한다."""
        import mdp.cli.train as train_mod

        cb_file = tmp_path / "cbs.yaml"
        cb_file.write_text(yaml.dump([
            {"_component_": "ModelCheckpoint", "monitor": "val_loss", "save_top_k": 1},
        ]))

        captured = {}

        def fake_run_single(settings, cb_configs=None):
            captured["cb_configs"] = cb_configs
            return {}

        monkeypatch.setattr(train_mod, "_run_single", fake_run_single)
        monkeypatch.setattr(train_mod, "_detect_gpu_count", lambda: 0)

        from tests.e2e.conftest import make_test_settings
        import unittest.mock as mock

        with mock.patch("mdp.settings.factory.SettingsFactory") as MockFactory:
            MockFactory.return_value.for_training.return_value = make_test_settings()
            train_mod.run_train(
                recipe_path="dummy.yaml",
                config_path="dummy_cfg.yaml",
                callbacks_file=str(cb_file),
            )

        assert captured["cb_configs"] is not None
        assert len(captured["cb_configs"]) == 1
        assert captured["cb_configs"][0]["_component_"] == "ModelCheckpoint"

    def test_no_callbacks_file_passes_none_to_run_training(self, monkeypatch):
        """callbacks_file=None이면 cb_configs가 falsy로 전달된다."""
        import mdp.cli.train as train_mod

        captured = {}

        def fake_run_single(settings, cb_configs=None):
            captured["cb_configs"] = cb_configs
            return {}

        monkeypatch.setattr(train_mod, "_run_single", fake_run_single)
        monkeypatch.setattr(train_mod, "_detect_gpu_count", lambda: 0)

        from tests.e2e.conftest import make_test_settings
        import unittest.mock as mock

        with mock.patch("mdp.settings.factory.SettingsFactory") as MockFactory:
            MockFactory.return_value.for_training.return_value = make_test_settings()
            train_mod.run_train(
                recipe_path="dummy.yaml",
                config_path="dummy_cfg.yaml",
                callbacks_file=None,
            )

        # cb_configs=None이거나 [] (falsy) 이면 됨
        assert not captured.get("cb_configs")

    def test_run_training_resolves_and_injects_callbacks(self):
        """run_training()이 cb_configs를 resolve하여 Trainer(callbacks=...)에 주입한다."""
        from unittest import mock
        from tests.e2e.conftest import make_test_settings
        import mdp.cli._torchrun_entry as entry_mod

        settings = make_test_settings()
        cb_configs = [{"_component_": "ModelCheckpoint", "monitor": "val_loss", "save_top_k": 1}]

        captured_trainer_kwargs = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                captured_trainer_kwargs.update(kwargs)
            def train(self):
                return {}

        def fake_run_rl(settings, cb_configs=None):
            return {}

        with mock.patch.object(entry_mod, "_resolve_cb_configs", return_value=["resolved_cb"]), \
             mock.patch.object(entry_mod, "_print_callbacks_log"), \
             mock.patch("mdp.factory.factory.Factory") as MockFactory, \
             mock.patch("mdp.training.trainer.Trainer", FakeTrainer):
            MockFactory.return_value.create_model.return_value = mock.MagicMock()
            MockFactory.return_value.create_dataloaders.return_value = {"train": mock.MagicMock()}

            entry_mod.run_training(settings, cb_configs=cb_configs)

        assert "callbacks" in captured_trainer_kwargs
        assert captured_trainer_kwargs["callbacks"] == ["resolved_cb"]


class TestCallbacksInference:
    """inference 경로에서 콜백 파일 로드 검증."""

    def test_load_and_resolve_callbacks(self, tmp_path):
        """콜백 파일을 로드하고 ComponentResolver로 resolve한다."""
        cb_file = tmp_path / "analysis.yaml"
        # aliases.yaml에 등록된 ModelCheckpoint 사용 (EarlyStopping은 U2에서 alias 제거됨)
        cb_file.write_text(yaml.dump([
            {"_component_": "ModelCheckpoint", "monitor": "val_loss", "save_top_k": 1},
        ]))

        from mdp.settings.resolver import ComponentResolver
        from mdp.training._common import create_callbacks, load_callbacks_from_file

        configs = load_callbacks_from_file(str(cb_file))
        callbacks = create_callbacks(configs, ComponentResolver())

        assert len(callbacks) == 1
        from mdp.training.callbacks.checkpoint import ModelCheckpoint
        assert isinstance(callbacks[0], ModelCheckpoint)
        assert callbacks[0].monitor == "val_loss"
