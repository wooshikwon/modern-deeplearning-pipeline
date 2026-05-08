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
        assert result[0].component == "EarlyStopping"
        assert result[0].kwargs["patience"] == 3
        assert result[1].component == "ModelCheckpoint"

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
        assert result[0].component == "ProgressBar"


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
