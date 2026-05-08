"""SettingsFactory override 통합 테스트."""

import tempfile
from pathlib import Path

import pytest
import yaml

from mdp.settings.factory import SettingsFactory
from mdp.settings.planner import SettingsPlanner


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


def test_env_var_substitution_with_auto_cast(tmp_path, monkeypatch):
    """${VAR:default} 치환은 SettingsFactory에서 타입 변환까지 적용된다."""
    monkeypatch.setenv("MDP_TEST_EPOCHS", "4")
    monkeypatch.setenv("MDP_TEST_GPUS", "0")
    recipe = {
        "name": "test-env",
        "task": "text_generation",
        "model": {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://gpt2",
        },
        "data": {
            "dataset": {"_component_": "HuggingFaceDataset", "source": "wikitext"},
            "collator": {"_component_": "CausalLMCollator", "tokenizer": "gpt2"},
        },
        "training": {"epochs": "${MDP_TEST_EPOCHS:1}", "precision": "fp32"},
        "metadata": {"author": "test", "description": "env override test"},
    }
    config = {"compute": {"gpus": "${MDP_TEST_GPUS:1}"}}

    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    recipe_path.write_text(yaml.dump(recipe))
    config_path.write_text(yaml.dump(config))

    settings = SettingsFactory().for_training(str(recipe_path), str(config_path))

    assert settings.recipe.training.epochs == 4
    assert settings.config.compute.gpus == 0


def test_settings_planner_preserves_callback_configs_without_resolving(
    recipe_and_config,
    tmp_path,
):
    """planner는 callbacks YAML을 raw config로 보존하고 instance resolve는 하지 않는다."""
    recipe_path, config_path = recipe_and_config
    callbacks_path = tmp_path / "callbacks.yaml"
    callbacks_path.write_text(
        yaml.dump([
            {"_component_": "ModelCheckpoint", "monitor": "val_loss", "save_top_k": 1}
        ])
    )

    plan = SettingsPlanner().load_training(
        recipe_path,
        config_path,
        callbacks_file=str(callbacks_path),
    )

    assert plan.callback_configs == (
        {"_component_": "ModelCheckpoint", "monitor": "val_loss", "save_top_k": 1},
    )
    assert plan.command == "train"
    assert plan.mode == "sft"


def test_settings_plan_distributed_intent_ignores_gpu_count_without_strategy(
    recipe_and_config,
):
    """distributed_intent는 compute.gpus가 아니라 compute.distributed를 기준으로 한다."""
    recipe_path, config_path = recipe_and_config

    multi_gpu_plan = SettingsPlanner().load_inference(
        recipe_path,
        config_path,
        overrides=["config.compute.gpus=4"],
    )
    distributed_plan = SettingsPlanner().load_training(
        recipe_path,
        config_path,
        overrides=["config.compute.gpus=4", "config.compute.distributed.strategy=ddp"],
    )

    assert multi_gpu_plan.settings.config.compute.gpus == 4
    assert multi_gpu_plan.distributed_intent is False
    assert distributed_plan.distributed_intent is True


def test_no_overrides(recipe_and_config):
    """오버라이드 없이 기본 동작 유지."""
    recipe_path, config_path = recipe_and_config
    settings = SettingsFactory().for_training(recipe_path, config_path)
    assert settings.recipe.training.epochs == 10
