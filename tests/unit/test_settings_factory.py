"""SettingsFactory artifact config snapshot tests."""

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


def test_from_artifact_uses_config_snapshot_by_default(tmp_path: Path) -> None:
    from mdp.settings.factory import SettingsFactory

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

    settings = SettingsFactory().from_artifact(str(artifact_dir))

    assert settings.config.serving is not None
    assert settings.config.serving.max_batch_size == 4
    assert settings.config.serving.device_map == "auto"
    assert settings.config.serving.max_memory == {"0": "24GiB"}


def test_from_artifact_can_ignore_config_snapshot(tmp_path: Path) -> None:
    from mdp.settings.factory import SettingsFactory

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_recipe(artifact_dir)
    (artifact_dir / "config.yaml").write_text(
        yaml.dump({"serving": {"device_map": "auto"}})
    )

    settings = SettingsFactory().from_artifact(
        str(artifact_dir),
        use_config_snapshot=False,
    )

    assert settings.config.serving is None


def test_from_artifact_manifest_config_file_and_override_precedence(
    tmp_path: Path,
) -> None:
    from mdp.settings.factory import SettingsFactory

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_recipe(artifact_dir)
    (artifact_dir / "runtime.yaml").write_text(
        yaml.dump({"serving": {"device_map": "balanced", "max_batch_size": 2}})
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"config_file": "runtime.yaml"})
    )

    settings = SettingsFactory().from_artifact(
        str(artifact_dir),
        overrides=["config.serving.device_map=sequential"],
    )

    assert settings.config.serving is not None
    assert settings.config.serving.max_batch_size == 2
    assert settings.config.serving.device_map == "sequential"
