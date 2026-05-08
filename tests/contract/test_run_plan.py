"""RunPlan contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from mdp.settings.components import ComponentSpec
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.run_plan_builder import RunPlanBuilder


def _write_training_recipe(path: Path) -> None:
    recipe = {
        "name": "contract-settings",
        "task": "text_generation",
        "model": {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://gpt2",
        },
        "data": {
            "dataset": {
                "_component_": "HuggingFaceDataset",
                "source": "wikitext",
            },
            "collator": {
                "_component_": "CausalLMCollator",
                "tokenizer": "gpt2",
            },
        },
        "training": {"epochs": "${MDP_CONTRACT_EPOCHS:1}", "precision": "fp32"},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3},
        "metadata": {"author": "contract", "description": "settings plan"},
    }
    path.write_text(yaml.safe_dump(recipe))


def _write_config(path: Path) -> None:
    config = {
        "compute": {"gpus": "${MDP_CONTRACT_GPUS:0}"},
        "storage": {"checkpoint_dir": "./checkpoints"},
    }
    path.write_text(yaml.safe_dump(config))


def test_training_returns_run_plan_with_runtime_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MDP_CONTRACT_EPOCHS", "4")
    monkeypatch.setenv("MDP_CONTRACT_GPUS", "2")
    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    callbacks_path = tmp_path / "callbacks.yaml"
    _write_training_recipe(recipe_path)
    _write_config(config_path)
    callbacks_path.write_text(
        yaml.safe_dump([
            {"_component_": "ModelCheckpoint", "monitor": "loss", "save_top_k": 1}
        ])
    )

    plan = RunPlanBuilder().training(
        recipe_path,
        config_path,
        overrides=[
            "training.precision=bf16",
            "config.storage.checkpoint_dir=./contract-ckpts",
            "config.compute.distributed.strategy=ddp",
        ],
        callbacks_file=callbacks_path,
    )

    assert isinstance(plan, RunPlan)
    assert plan.command == "train"
    assert plan.mode == "sft"
    assert plan.validation_scope == "training"
    assert plan.sources == RunSources(
        recipe_path=recipe_path,
        config_path=config_path,
    )
    assert plan.overrides == (
        "training.precision=bf16",
        "config.storage.checkpoint_dir=./contract-ckpts",
        "config.compute.distributed.strategy=ddp",
    )
    assert plan.callback_configs == (
        ComponentSpec(
            component="ModelCheckpoint",
            kwargs={"monitor": "loss", "save_top_k": 1},
            path="callbacks[0]",
        ),
    )
    assert plan.settings.recipe.training.epochs == 4
    assert plan.settings.recipe.training.precision == "bf16"
    assert plan.settings.config.compute.gpus == 2
    assert plan.settings.config.storage.checkpoint_dir == "./contract-ckpts"
    assert plan.distributed_intent is True


def test_artifact_records_config_snapshot_source_and_scope(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    _write_training_recipe(artifact_dir / "recipe.yaml")
    (artifact_dir / "runtime.yaml").write_text(
        yaml.safe_dump({"serving": {"device_map": "balanced", "max_batch_size": 2}})
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"config_file": "runtime.yaml"})
    )

    plan = RunPlanBuilder().artifact(
        artifact_dir,
        overrides=["config.serving.device_map=sequential"],
        command="serve",
    )

    assert plan.command == "serve"
    assert plan.mode == "serving"
    assert plan.validation_scope == "artifact"
    assert plan.sources == RunSources(
        recipe_path=artifact_dir / "recipe.yaml",
        config_path=artifact_dir / "runtime.yaml",
        artifact_dir=artifact_dir,
    )
    assert plan.settings.config.serving is not None
    assert plan.settings.config.serving.max_batch_size == 2
    assert plan.settings.config.serving.device_map == "sequential"
