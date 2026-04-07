"""체크포인트에 recipe.yaml이 저장/복원되는지 검증."""

from __future__ import annotations

import yaml
import torch

from mdp.training.callbacks.checkpoint import ModelCheckpoint
from tests.e2e.models import TinyVisionModel


def test_checkpoint_saves_recipe(tmp_path) -> None:
    """save_checkpoint이 recipe.yaml을 저장하는지."""
    ckpt = ModelCheckpoint(dirpath=tmp_path)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    recipe_dict = {"name": "test-exp", "task": "image_classification", "model": {"_component_": "test"}}

    ckpt.save_checkpoint(model, optimizer, None, 0, 100, recipe_dict=recipe_dict)

    recipe_path = tmp_path / "checkpoint-100" / "recipe.yaml"
    assert recipe_path.exists()
    saved = yaml.safe_load(recipe_path.read_text())
    assert saved["name"] == "test-exp"
    assert saved["task"] == "image_classification"


def test_checkpoint_recipe_written_once(tmp_path) -> None:
    """recipe.yaml은 최초 1회만 저장, 이후 체크포인트에서는 덮어쓰지 않는다."""
    ckpt = ModelCheckpoint(dirpath=tmp_path)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    original = {"name": "original", "task": "image_classification"}
    modified = {"name": "modified", "task": "text_generation"}

    # 첫 번째 저장
    ckpt.save_checkpoint(model, optimizer, None, 0, 100, recipe_dict=original)
    # 같은 step에 다시 저장 시도 (recipe가 달라도 덮어쓰지 않음)
    ckpt.save_checkpoint(model, optimizer, None, 1, 100, recipe_dict=modified)

    saved = yaml.safe_load((tmp_path / "checkpoint-100" / "recipe.yaml").read_text())
    assert saved["name"] == "original"


def test_checkpoint_without_recipe(tmp_path) -> None:
    """recipe_dict=None이면 recipe.yaml을 생성하지 않는다."""
    ckpt = ModelCheckpoint(dirpath=tmp_path)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt.save_checkpoint(model, optimizer, None, 0, 200)

    recipe_path = tmp_path / "checkpoint-200" / "recipe.yaml"
    assert not recipe_path.exists()
