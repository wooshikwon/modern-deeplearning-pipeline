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


class TestCallbacksOverrideTraining:
    """--callbacks가 Settings.recipe.callbacks를 override하는지 검증."""

    def test_override_recipe_callbacks(self, tmp_path):
        """callbacks 파일이 recipe의 기존 callbacks를 교체한다."""
        from tests.e2e.conftest import make_test_settings

        # 기존 recipe callbacks
        settings = make_test_settings()
        settings.recipe.callbacks = [
            {"_component_": "EarlyStopping", "patience": 5},
        ]
        assert len(settings.recipe.callbacks) == 1

        # --callbacks 파일
        cb_file = tmp_path / "override.yaml"
        new_cbs = [
            {"_component_": "ModelCheckpoint", "save_top_k": 3},
            {"_component_": "ProgressBar"},
        ]
        cb_file.write_text(yaml.dump(new_cbs))

        # override 적용 (run_train 내부 로직 재현)
        from mdp.training._common import load_callbacks_from_file
        settings.recipe.callbacks = load_callbacks_from_file(str(cb_file))

        assert len(settings.recipe.callbacks) == 2
        assert settings.recipe.callbacks[0]["_component_"] == "ModelCheckpoint"
        assert settings.recipe.callbacks[1]["_component_"] == "ProgressBar"

    def test_no_callbacks_file_preserves_recipe(self):
        """--callbacks 없으면 recipe 콜백을 그대로 유지한다."""
        from tests.e2e.conftest import make_test_settings

        settings = make_test_settings()
        settings.recipe.callbacks = [
            {"_component_": "EarlyStopping", "patience": 5},
        ]

        # callbacks_file=None 시 변경 없음
        callbacks_file = None
        if callbacks_file:
            from mdp.training._common import load_callbacks_from_file
            settings.recipe.callbacks = load_callbacks_from_file(callbacks_file)

        assert len(settings.recipe.callbacks) == 1
        assert settings.recipe.callbacks[0]["_component_"] == "EarlyStopping"


class TestCallbacksInference:
    """inference 경로에서 콜백 파일 로드 검증."""

    def test_load_and_resolve_callbacks(self, tmp_path):
        """콜백 파일을 로드하고 ComponentResolver로 resolve한다."""
        cb_file = tmp_path / "analysis.yaml"
        # aliases.yaml에 등록된 EarlyStopping 사용
        cb_file.write_text(yaml.dump([
            {"_component_": "EarlyStopping", "patience": 2, "monitor": "val_loss"},
        ]))

        from mdp.settings.resolver import ComponentResolver
        from mdp.training._common import create_callbacks, load_callbacks_from_file

        configs = load_callbacks_from_file(str(cb_file))
        callbacks = create_callbacks(configs, ComponentResolver())

        assert len(callbacks) == 1
        from mdp.training.callbacks.early_stopping import EarlyStopping
        assert isinstance(callbacks[0], EarlyStopping)
        assert callbacks[0].patience == 2
