"""SettingsFactory override 통합 테스트."""

import tempfile
from pathlib import Path

import pytest
import yaml

from mdp.settings.factory import SettingsFactory


@pytest.fixture
def recipe_and_config(tmp_path):
    """최소 Recipe + Config YAML을 생성한다."""
    recipe = {
        "name": "test-override",
        "task": "text_generation",
        "model": {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://gpt2",
        },
        "data": {
            "dataset": {"_component_": "HuggingFaceDataset", "source": "wikitext"},
            "collator": {"_component_": "CausalLMCollator", "tokenizer": "gpt2"},
        },
        "training": {"epochs": 10, "precision": "fp32"},
        "metadata": {"author": "test", "description": "override test"},
    }
    config = {
        "compute": {"gpus": 0},
    }

    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    recipe_path.write_text(yaml.dump(recipe))
    config_path.write_text(yaml.dump(config))
    return str(recipe_path), str(config_path)


def test_override_recipe_field(recipe_and_config):
    """training.epochs=0.5 오버라이드가 적용된다."""
    recipe_path, config_path = recipe_and_config
    settings = SettingsFactory().for_training(
        recipe_path, config_path,
        overrides=["training.epochs=0.5"],
    )
    assert settings.recipe.training.epochs == 0.5


def test_override_config_field(recipe_and_config):
    """config. 접두사로 Config 필드를 오버라이드한다."""
    recipe_path, config_path = recipe_and_config
    settings = SettingsFactory().for_training(
        recipe_path, config_path,
        overrides=["config.storage.checkpoint_dir=./my-ckpts"],
    )
    assert settings.config.storage.checkpoint_dir == "./my-ckpts"


def test_override_multiple(recipe_and_config):
    """여러 오버라이드를 동시에 적용한다."""
    recipe_path, config_path = recipe_and_config
    settings = SettingsFactory().for_training(
        recipe_path, config_path,
        overrides=["training.epochs=3", "training.precision=bf16"],
    )
    assert settings.recipe.training.epochs == 3
    assert settings.recipe.training.precision == "bf16"


def test_no_overrides(recipe_and_config):
    """오버라이드 없이 기본 동작 유지."""
    recipe_path, config_path = recipe_and_config
    settings = SettingsFactory().for_training(recipe_path, config_path)
    assert settings.recipe.training.epochs == 10
