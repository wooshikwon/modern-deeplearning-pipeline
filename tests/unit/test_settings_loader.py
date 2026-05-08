"""SettingsLoader artifact config snapshot and YAML tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def _write_recipe(artifact_dir: Path) -> None:
    recipe = {
        "name": "snapshot-test",
        "task": "image_classification",
        "model": {
            "_component_": "tests.e2e.models.TinyVisionModel",
            "num_classes": 2,
            "hidden_dim": 16,
        },
        "data": {
            "dataset": {
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": "/tmp/fake",
                "split": "train",
            },
            "collator": {
                "_component_": "mdp.data.collators.ClassificationCollator",
                "tokenizer": "gpt2",
            },
        },
        "training": {"epochs": 1},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3},
        "metadata": {"author": "test", "description": "snapshot test"},
    }
    (artifact_dir / "recipe.yaml").write_text(yaml.dump(recipe))


def test_load_artifact_settings_uses_config_snapshot_by_default(
    tmp_path: Path,
) -> None:
    from mdp.settings.loader import SettingsLoader

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_recipe(artifact_dir)
    (artifact_dir / "config.yaml").write_text(
        yaml.dump({
            "serving": {
                "max_batch_size": 4,
                "device_map": "auto",
                "max_memory": {"0": "24GiB"},
            },
        })
    )

    settings = SettingsLoader().load_artifact_settings(str(artifact_dir))

    assert settings.config.serving is not None
    assert settings.config.serving.max_batch_size == 4
    assert settings.config.serving.device_map == "auto"
    assert settings.config.serving.max_memory == {"0": "24GiB"}


def test_load_artifact_settings_can_ignore_config_snapshot(tmp_path: Path) -> None:
    from mdp.settings.loader import SettingsLoader

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_recipe(artifact_dir)
    (artifact_dir / "config.yaml").write_text(
        yaml.dump({"serving": {"device_map": "auto"}})
    )

    settings = SettingsLoader().load_artifact_settings(
        str(artifact_dir),
        use_config_snapshot=False,
    )

    assert settings.config.serving is None


def test_load_artifact_settings_manifest_config_file_and_override_precedence(
    tmp_path: Path,
) -> None:
    from mdp.settings.loader import SettingsLoader

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_recipe(artifact_dir)
    (artifact_dir / "runtime.yaml").write_text(
        yaml.dump({"serving": {"device_map": "balanced", "max_batch_size": 2}})
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"config_file": "runtime.yaml"})
    )

    settings = SettingsLoader().load_artifact_settings(
        str(artifact_dir),
        overrides=["config.serving.device_map=sequential"],
    )

    assert settings.config.serving is not None
    assert settings.config.serving.max_batch_size == 2
    assert settings.config.serving.device_map == "sequential"


def test_load_yaml_rejects_duplicate_keys_with_file_and_yaml_path(
    tmp_path: Path,
) -> None:
    from mdp.settings.loader import SettingsLoader

    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "training:\n"
        "  epochs: 1\n"
        "training:\n"
        "  max_steps: 10\n"
    )

    try:
        SettingsLoader.load_yaml(str(path))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("duplicate YAML key should fail")

    assert str(path) in message
    assert "YAML path $.training" in message
    assert "duplicate key 'training'" in message


def test_load_yaml_reports_nested_duplicate_key_path(tmp_path: Path) -> None:
    from mdp.settings.loader import SettingsLoader

    path = tmp_path / "nested-duplicate.yaml"
    path.write_text(
        "training:\n"
        "  epochs: 1\n"
        "  epochs: 2\n"
    )

    try:
        SettingsLoader.load_yaml(str(path))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("nested duplicate YAML key should fail")

    assert str(path) in message
    assert "YAML path $.training.epochs" in message
    assert "duplicate key 'epochs'" in message


def test_load_yaml_rejects_empty_mapping_root_with_file_and_yaml_path(
    tmp_path: Path,
) -> None:
    from mdp.settings.loader import SettingsLoader

    path = tmp_path / "empty.yaml"
    path.write_text("")

    try:
        SettingsLoader.load_yaml(str(path))
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("empty YAML should fail")

    assert str(path) in message
    assert "YAML path $" in message
    assert "empty" in message


def test_load_yaml_rejects_list_and_scalar_roots_with_file_and_yaml_path(
    tmp_path: Path,
) -> None:
    from mdp.settings.loader import SettingsLoader

    list_path = tmp_path / "list.yaml"
    scalar_path = tmp_path / "scalar.yaml"
    list_path.write_text("- item\n")
    scalar_path.write_text("plain-scalar\n")

    for path, expected_type in ((list_path, "list"), (scalar_path, "str")):
        try:
            SettingsLoader.load_yaml(str(path))
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"{path} should fail")

        assert str(path) in message
        assert "YAML path $" in message
        assert "root must be a mapping" in message
        assert f"actual: {expected_type}" in message
