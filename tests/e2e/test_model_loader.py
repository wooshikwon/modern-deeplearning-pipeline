"""공용 모델 가중치 로딩 테스트."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from mdp.serving.model_loader import load_checkpoint_weights
from tests.e2e.models import TinyVisionModel


def test_load_safetensors(tmp_path: Path) -> None:
    """model.safetensors에서 가중치를 로드."""
    from safetensors.torch import save_file

    model_orig = TinyVisionModel(num_classes=2, hidden_dim=16)
    save_file(model_orig.state_dict(), tmp_path / "model.safetensors")

    model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
    load_checkpoint_weights(model_new, tmp_path)

    for (n1, p1), (n2, p2) in zip(
        model_orig.named_parameters(), model_new.named_parameters()
    ):
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch"


def test_load_model_pt_fallback(tmp_path: Path) -> None:
    """model.safetensors 없을 때 model.pt로 fallback."""
    model_orig = TinyVisionModel(num_classes=2, hidden_dim=16)
    torch.save(model_orig.state_dict(), tmp_path / "model.pt")

    model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
    load_checkpoint_weights(model_new, tmp_path)

    for (_, p1), (_, p2) in zip(
        model_orig.named_parameters(), model_new.named_parameters()
    ):
        assert torch.equal(p1, p2)


def test_load_no_weights_warns(tmp_path: Path, caplog) -> None:
    """가중치 파일 없으면 warning."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)

    with caplog.at_level(logging.WARNING):
        load_checkpoint_weights(model, tmp_path)

    assert "가중치 파일이 없습니다" in caplog.text


def test_artifact_load_plan_selects_manifest_policy_role(tmp_path: Path) -> None:
    """manifest checkpoint에서 RL serving 기본 role인 policy를 선택한다."""
    from mdp.serving.model_loader import resolve_artifact_load_plan

    policy_dir = tmp_path / "policy"
    reward_dir = tmp_path / "reward"
    policy_dir.mkdir()
    reward_dir.mkdir()
    (policy_dir / "model.safetensors").touch()
    (reward_dir / "model.safetensors").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "models": {
                "policy": {
                    "role": "policy",
                    "format": "safetensors",
                    "path": "policy/model.safetensors",
                    "trainable": True,
                },
                "reward": {
                    "role": "reward",
                    "format": "safetensors",
                    "path": "reward/model.safetensors",
                    "trainable": False,
                },
            }
        })
    )

    plan = resolve_artifact_load_plan(tmp_path)

    assert plan.artifact_kind == "training_checkpoint"
    assert plan.role == "policy"
    assert plan.weight_format == "safetensors"
    assert plan.weights_dir == policy_dir


def test_reconstruct_model_from_manifest_checkpoint(tmp_path: Path) -> None:
    """manifest checkpoint는 record path 기준으로 가중치를 로드한다."""
    from safetensors.torch import save_file

    from mdp.serving.model_loader import reconstruct_model

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "models": {
                "model": {
                    "role": "policy",
                    "format": "safetensors",
                    "path": "model.safetensors",
                    "trainable": True,
                }
            }
        })
    )
    recipe = {
        "name": "manifest-checkpoint-test",
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
            "collator": {"_component_": "mdp.data.collators.ClassificationCollator"},
        },
        "training": {"epochs": 1},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3},
        "metadata": {"author": "test", "description": "manifest checkpoint test"},
    }
    (tmp_path / "recipe.yaml").write_text(yaml.dump(recipe))

    loaded, settings = reconstruct_model(tmp_path)

    assert settings.recipe.name == "manifest-checkpoint-test"
    for (_, expected), (_, actual) in zip(
        model.named_parameters(),
        loaded.named_parameters(),
    ):
        assert torch.equal(expected, actual)


def test_reconstruct_model_dispatches_safetensors_with_device_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """device_map reconstruction uses the public artifact loading path."""
    from safetensors.torch import save_file

    from mdp.serving import model_loader

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    _write_tiny_recipe(tmp_path, name="device-map-safetensors-test")

    calls: list[tuple[Any, str, str, dict[str, str] | None]] = []

    def fake_dispatch(
        model: Any,
        checkpoint: str,
        device_map: str,
        max_memory: dict[str, str] | None = None,
    ) -> Any:
        calls.append((model, checkpoint, device_map, max_memory))
        model.hf_device_map = {"": "cpu"}
        return model

    monkeypatch.setattr(model_loader, "_dispatch_model", fake_dispatch)

    loaded, _settings = model_loader.reconstruct_model(
        tmp_path,
        device_map="auto",
        max_memory={"cpu": "4GiB"},
    )

    assert calls
    assert calls[0][1] == str(tmp_path / "model.safetensors")
    assert calls[0][2] == "auto"
    assert calls[0][3] == {"cpu": "4GiB"}
    assert loaded.hf_device_map == {"": "cpu"}


def test_reconstruct_model_dispatches_model_pt_fallback_with_device_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """device_map reconstruction falls back to model.pt through public behavior."""
    from mdp.serving import model_loader

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    _write_tiny_recipe(tmp_path, name="device-map-pt-test")

    checkpoints: list[str] = []

    def fake_dispatch(
        model: Any,
        checkpoint: str,
        device_map: str,
        max_memory: dict[str, str] | None = None,
    ) -> Any:
        checkpoints.append(checkpoint)
        return model

    monkeypatch.setattr(model_loader, "_dispatch_model", fake_dispatch)

    model_loader.reconstruct_model(tmp_path, device_map="auto")

    assert checkpoints == [str(tmp_path / "model.pt")]


def test_reconstruct_model_device_map_requires_weights(tmp_path: Path) -> None:
    """device_map reconstruction fails clearly when no dispatchable weights exist."""
    from mdp.serving.model_loader import reconstruct_model

    _write_tiny_recipe(tmp_path, name="device-map-missing-weights-test")

    with pytest.raises(ValueError, match="model.safetensors/model.pt"):
        reconstruct_model(tmp_path, device_map="auto")


def test_serving_config_device_map_fields() -> None:
    """ServingConfig에 device_map, max_memory 필드가 존재한다."""
    from mdp.settings.schema import ServingConfig

    config = ServingConfig(device_map="auto", max_memory={"0": "24GiB"})
    assert config.device_map == "auto"
    assert config.max_memory == {"0": "24GiB"}

    default = ServingConfig()
    assert default.device_map is None
    assert default.max_memory is None


def _write_tiny_recipe(tmp_path: Path, *, name: str) -> None:
    recipe = {
        "name": name,
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
            "collator": {"_component_": "mdp.data.collators.ClassificationCollator"},
        },
        "training": {"epochs": 1},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3},
        "metadata": {"author": "test", "description": "device map test"},
    }
    (tmp_path / "recipe.yaml").write_text(yaml.dump(recipe))
