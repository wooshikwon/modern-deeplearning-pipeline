"""Schema hardening contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mdp.settings.components import ComponentSpec, ModelComponentSpec, RoleModelSpec
from mdp.settings.schema import Config, Recipe


def _minimal_recipe() -> dict:
    return {
        "name": "schema-contract",
        "task": "text_generation",
        "model": {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://gpt2",
            "trust_remote_code": True,
        },
        "data": {
            "dataset": {
                "_component_": "HuggingFaceDataset",
                "source": "wikitext",
                "custom_dataset_kwarg": "kept-open",
            },
            "collator": {
                "_component_": "CausalLMCollator",
                "tokenizer": "gpt2",
                "custom_collator_kwarg": 1,
            },
            "dataloader": {"batch_size": 2},
        },
        "training": {"epochs": 1},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3, "custom_kwarg": True},
        "metadata": {"author": "test", "description": "schema contract"},
    }


def test_closed_zone_rejects_unknown_nested_recipe_fields() -> None:
    recipe = _minimal_recipe()
    recipe["training"]["val_check_units"] = "step"

    with pytest.raises(ValidationError) as exc_info:
        Recipe(**recipe)

    assert "training.val_check_units" in str(exc_info.value)


def test_component_kwargs_remain_open() -> None:
    recipe = Recipe(**_minimal_recipe())

    assert isinstance(recipe.model, ModelComponentSpec)
    assert recipe.model.kwargs["trust_remote_code"] is True
    assert isinstance(recipe.data.dataset, ComponentSpec)
    assert recipe.data.dataset.kwargs["custom_dataset_kwarg"] == "kept-open"
    assert recipe.data.collator.kwargs["custom_collator_kwarg"] == 1
    assert recipe.optimizer is not None
    assert recipe.optimizer.kwargs["custom_kwarg"] is True


def test_component_serializer_preserves_yaml_snapshot_shape() -> None:
    recipe = Recipe(**_minimal_recipe())
    dumped = recipe.model_dump(mode="json")

    assert dumped["model"] == _minimal_recipe()["model"]
    assert dumped["data"]["dataset"] == _minimal_recipe()["data"]["dataset"]
    assert dumped["optimizer"] == _minimal_recipe()["optimizer"]


def test_model_component_allows_pretrained_only_route() -> None:
    recipe = _minimal_recipe()
    recipe["model"] = {"pretrained": "hf://gpt2", "torch_dtype": "bfloat16"}

    parsed = Recipe(**recipe)

    assert parsed.model.component is None
    assert parsed.model.pretrained == "hf://gpt2"
    assert parsed.model.kwargs["torch_dtype"] == "bfloat16"
    assert parsed.model_dump(mode="json")["model"] == recipe["model"]


def test_non_model_component_rejects_missing_component() -> None:
    recipe = _minimal_recipe()
    recipe["data"]["dataset"] = {"source": "wikitext"}

    with pytest.raises(ValidationError) as exc_info:
        Recipe(**recipe)

    assert "data.dataset" in str(exc_info.value)
    assert "_component_" in str(exc_info.value)


def test_adapter_shorthand_string_rejected_in_final_recipe() -> None:
    recipe = _minimal_recipe()
    recipe["adapter"] = "qlora"

    with pytest.raises(ValidationError) as exc_info:
        Recipe(**recipe)

    assert "adapter" in str(exc_info.value)


def test_role_model_spec_owns_rl_nested_components() -> None:
    recipe = _minimal_recipe()
    recipe["task"] = "text_generation"
    recipe["loss"] = None
    recipe["rl"] = {
        "algorithm": {"_component_": "DPO", "beta": 0.1},
        "models": {
            "policy": {
                "_component_": "transformers.AutoModelForCausalLM",
                "pretrained": "hf://gpt2",
                "torch_dtype": "bfloat16",
                "adapter": {"_component_": "LoRA", "r": 8},
                "optimizer": {"_component_": "AdamW", "lr": 1e-4},
            },
            "reference": {
                "pretrained": "hf://gpt2",
                "freeze": True,
            },
        },
    }

    parsed = Recipe(**recipe)
    policy = parsed.rl.models["policy"] if parsed.rl is not None else None
    reference = parsed.rl.models["reference"] if parsed.rl is not None else None

    assert isinstance(policy, RoleModelSpec)
    assert policy.model.component == "transformers.AutoModelForCausalLM"
    assert policy.model.kwargs["torch_dtype"] == "bfloat16"
    assert policy.adapter is not None
    assert policy.adapter.component == "LoRA"
    assert policy.optimizer is not None
    assert policy.optimizer.kwargs["lr"] == 1e-4
    assert reference is not None
    assert reference.model.pretrained == "hf://gpt2"
    assert reference.freeze is True
    dumped_rl = parsed.model_dump(mode="json", exclude_none=True)["rl"]
    assert dumped_rl == recipe["rl"]


def test_role_model_explicit_model_rejects_stray_role_keys() -> None:
    recipe = _minimal_recipe()
    recipe["loss"] = None
    recipe["rl"] = {
        "algorithm": {"_component_": "DPO", "beta": 0.1},
        "models": {
            "policy": {
                "model": {"_component_": "tests.e2e.models.TinyLanguageModel"},
                "_component_": "typo.ShouldNotBeHere",
                "optimizer": {"_component_": "AdamW", "lr": 1e-4},
            },
        },
    }

    with pytest.raises(ValidationError) as exc_info:
        Recipe(**recipe)

    message = str(exc_info.value)
    assert "rl.models.policy" in message
    assert "_component_" in message


def test_role_model_explicit_model_preserves_model_namespace_on_dump() -> None:
    recipe = _minimal_recipe()
    recipe["loss"] = None
    recipe["rl"] = {
        "algorithm": {"_component_": "DPO", "beta": 0.1},
        "models": {
            "policy": {
                "model": {
                    "_component_": "tests.e2e.models.TinyLanguageModel",
                    "optimizer": "model-constructor-value",
                },
                "optimizer": {"_component_": "AdamW", "lr": 1e-4},
            },
        },
    }

    parsed = Recipe(**recipe)
    dumped_policy = parsed.model_dump(mode="json", exclude_none=True)["rl"]["models"]["policy"]

    assert dumped_policy == recipe["rl"]["models"]["policy"]
    reparsed = Recipe(**parsed.model_dump(mode="json", exclude_none=True))
    policy = reparsed.rl.models["policy"] if reparsed.rl is not None else None
    assert policy is not None
    assert policy.model.kwargs["optimizer"] == "model-constructor-value"
    assert policy.optimizer is not None
    assert policy.optimizer.component == "AdamW"


def test_closed_zone_rejects_unknown_config_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Config(compute={"gpus": 1, "gpu_count": 1})

    assert "compute.gpu_count" in str(exc_info.value)


def test_distributed_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Config(compute={"gpus": 2, "distributed": {"stratgey": "none"}})

    assert "compute.distributed.stratgey" in str(exc_info.value)


def test_distributed_config_normalizes_strategy_component_block() -> None:
    config = Config(
        compute={
            "distributed": {
                "strategy": {"_component_": "DDPStrategy", "backend": "gloo"},
                "moe": {"enabled": True, "ep_size": 2},
            }
        }
    )

    assert isinstance(config.compute.distributed.strategy, ComponentSpec)
    assert config.compute.distributed.strategy.component == "DDPStrategy"
    assert config.compute.distributed.strategy.kwargs["backend"] == "gloo"
    assert config.compute.distributed.moe == {"enabled": True, "ep_size": 2}


def test_distributed_config_rejects_duplicate_strategy_kwargs() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Config(
            compute={
                "distributed": {
                    "strategy": {"_component_": "DDPStrategy", "backend": "gloo"},
                    "backend": "nccl",
                }
            }
        )

    assert "duplicated" in str(exc_info.value)
    assert "backend" in str(exc_info.value)


def test_settings_loader_schema_errors_include_file_and_yaml_path(
    tmp_path: Path,
) -> None:
    from mdp.settings.loader import SettingsLoader

    recipe = _minimal_recipe()
    recipe["training"]["val_check_units"] = "step"
    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    recipe_path.write_text(yaml.dump(recipe))
    config_path.write_text(yaml.dump({"compute": {"gpus": 0}}))

    with pytest.raises(ValueError) as exc_info:
        SettingsLoader().load_training_settings(recipe_path, config_path)

    message = str(exc_info.value)
    assert str(recipe_path) in message
    assert "YAML path $.training.val_check_units" in message
