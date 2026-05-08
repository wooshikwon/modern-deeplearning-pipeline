"""AssemblyPlan contract tests."""

from __future__ import annotations

import pytest
from torch import nn
from torch.utils.data import DataLoader

from mdp.assembly.planner import AssemblyPlanner
from mdp.assembly.specs import ComponentSpec
from mdp.settings.components import ComponentSpec as SettingsComponentSpec
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    Recipe,
    RLSpec,
    Settings,
    TrainingSpec,
)


def _data_spec() -> DataSpec:
    return DataSpec(
        dataset={
            "_component_": "mdp.data.datasets.HuggingFaceDataset",
            "source": "/tmp/fake",
            "split": "train",
        },
        val_dataset={
            "_component_": "mdp.data.datasets.HuggingFaceDataset",
            "source": "/tmp/fake-val",
            "split": "validation",
        },
        collator={
            "_component_": "mdp.data.collators.CausalLMCollator",
            "tokenizer": "gpt2",
        },
        dataloader={"batch_size": 4, "num_workers": 0},
    )


def _run_plan(settings: Settings, *, mode: str = "sft") -> RunPlan:
    return RunPlan(
        command="rl-train" if mode == "rl" else "train",
        mode=mode,
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=(
            SettingsComponentSpec(
                component="ModelCheckpoint",
                kwargs={"monitor": "loss"},
                path="callbacks[0]",
            ),
        ),
        validation_scope="training",
        distributed_intent=bool(settings.config.compute.distributed),
    )


def _assert_no_component_instances(value: object) -> None:
    assert not isinstance(value, (nn.Module, DataLoader))
    if isinstance(value, ComponentSpec):
        _assert_no_component_instances(value.kwargs)
    elif isinstance(value, dict):
        for child in value.values():
            _assert_no_component_instances(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_no_component_instances(child)


def test_sft_recipe_builds_policy_only_component_graph() -> None:
    settings = Settings(
        recipe=Recipe(
            name="contract-sft",
            task="image_classification",
            model={
                "_component_": "tests.e2e.models.TinyVisionModel",
                "num_classes": 2,
                "hidden_dim": 16,
            },
            head={
                "_component_": "ClassificationHead",
                "slot": "head.cls",
                "num_classes": 5,
                "hidden_dim": 16,
            },
            adapter={"_component_": "LoRA", "target": ["attn.q", "attn.v"]},
            data=_data_spec(),
            training=TrainingSpec(epochs=1),
            optimizer={"_component_": "AdamW", "lr": 1e-3},
            scheduler={"_component_": "StepLR", "step_size": 1},
            metadata=MetadataSpec(author="contract", description="assembly sft"),
        ),
        config=Config(),
    )

    plan = AssemblyPlanner.from_run_plan(_run_plan(settings))

    assert plan.kind == "sft_training"
    assert plan.trainer.kind == "sft"
    assert len(plan.models) == 1
    policy = plan.models[0]
    assert policy.role == "policy"
    assert policy.trainable is True
    assert policy.model.to_dict() == settings.recipe.model.to_yaml_dict()
    assert policy.model.resolved_component == "tests.e2e.models.TinyVisionModel"
    assert policy.model.model_route == "component"
    assert policy.head is not None
    assert policy.head.kwargs["slot"] == "head.cls"
    assert policy.head.resolved_component == "mdp.models.heads.classification.ClassificationHead"
    assert "_target_attr" not in policy.head.kwargs
    assert policy.adapter is not None
    assert policy.adapter.kwargs["target"] == ["attn.q", "attn.v"]
    assert policy.adapter.resolved_component == "mdp.models.adapters.lora.apply_lora"
    assert "target_modules" not in policy.adapter.kwargs
    assert policy.optimizer is not None
    assert policy.scheduler is not None
    assert policy.loss is None
    assert plan.data.dataset.to_dict() == settings.recipe.data.dataset.to_yaml_dict()
    assert plan.data.val_dataset is not None
    assert plan.data.val_dataset.to_dict() == settings.recipe.data.val_dataset.to_yaml_dict()
    assert plan.data.dataloader_config["batch_size"] == 4
    assert plan.data.distributed_intent is False
    assert plan.strategy is None
    assert len(plan.callbacks) == 1
    assert plan.callbacks[0].config.component == "ModelCheckpoint"
    assert plan.callbacks[0].config.resolved_component == (
        "mdp.training.callbacks.checkpoint.ModelCheckpoint"
    )
    _assert_no_component_instances(plan.models)
    _assert_no_component_instances(plan.data)
    _assert_no_component_instances(plan.trainer)
    _assert_no_component_instances(plan.callbacks)


def test_distributed_sft_plan_captures_strategy_boundary() -> None:
    settings = Settings(
        recipe=Recipe(
            name="contract-distributed",
            task="text_generation",
            model={"pretrained": "hf://some-org/some-model"},
            data=_data_spec(),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="contract", description="distributed"),
        ),
        config=Config(compute={"distributed": {"strategy": "ddp"}}),
    )

    plan = AssemblyPlanner.from_run_plan(_run_plan(settings))

    assert plan.data.distributed_intent is True
    assert plan.strategy is not None
    assert plan.strategy.strategy == "ddp"
    assert plan.strategy.distributed_intent is True
    assert "setup_models" in plan.strategy.capability_boundary
    assert plan.models[0].model.component is None
    assert plan.models[0].model.model_route == "pretrained"
    assert plan.models[0].model.to_dict() == {"pretrained": "hf://some-org/some-model"}


def test_rl_recipe_builds_role_model_nodes_with_trainable_flags() -> None:
    settings = Settings(
        recipe=Recipe(
            name="contract-rl",
            task="preference_optimization",
            model={"_component_": "unused-top-level-model"},
            data=_data_spec(),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="contract", description="assembly rl"),
            rl=RLSpec(
                algorithm={"_component_": "DPO", "beta": 0.2},
                models={
                    "policy": {
                        "_component_": "tests.e2e.models.TinyLanguageModel",
                        "optimizer": {"_component_": "AdamW", "lr": 1e-4},
                        "scheduler": {"_component_": "StepLR", "step_size": 1},
                    },
                    "reference": {
                        "_component_": "tests.e2e.models.TinyLanguageModel",
                        "freeze": True,
                    },
                    "reward": {
                        "_component_": "tests.e2e.models.TinyLanguageModel",
                        "trainable": False,
                    },
                },
            ),
        ),
        config=Config(),
    )

    plan = AssemblyPlanner.from_run_plan(_run_plan(settings, mode="rl"))

    assert plan.kind == "rl_training"
    assert plan.trainer.kind == "rl"
    assert plan.trainer.algorithm is not None
    assert plan.trainer.algorithm.to_dict() == {"_component_": "DPO", "beta": 0.2}
    assert plan.trainer.algorithm.resolved_component == "mdp.training.losses.rl.DPOLoss"
    models = {node.role: node for node in plan.models}
    assert set(models) == {"policy", "reference", "reward"}
    assert models["policy"].trainable is True
    assert models["policy"].optimizer is not None
    assert models["policy"].scheduler is not None
    assert models["reference"].trainable is False
    assert models["reference"].optimizer is None
    assert models["reference"].model.to_dict() == {
        "_component_": "tests.e2e.models.TinyLanguageModel"
    }
    assert models["reward"].trainable is False
    _assert_no_component_instances(plan.models)
    _assert_no_component_instances(plan.trainer)


def test_qlora_route_inputs_remain_owned_by_model_node() -> None:
    settings = Settings(
        recipe=Recipe(
            name="contract-qlora",
            task="text_generation",
            model={
                "_component_": "transformers.AutoModelForCausalLM",
                "pretrained": "hf://Qwen/Qwen2.5-7B",
                "torch_dtype": "bfloat16",
            },
            adapter={
                "_component_": "QLoRA",
                "target": "all-linear",
                "save": ["head.lm"],
            },
            data=_data_spec(),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="contract", description="qlora"),
        ),
        config=Config(),
    )

    plan = AssemblyPlanner.from_run_plan(_run_plan(settings))
    model = plan.models[0]

    assert model.model.to_dict() == settings.recipe.model.to_yaml_dict()
    assert model.adapter is not None
    assert model.adapter.to_dict() == settings.recipe.adapter.to_yaml_dict()
    assert model.adapter.kwargs["target"] == "all-linear"
    assert "target_modules" not in model.adapter.kwargs
    assert "modules_to_save" not in model.adapter.kwargs


def test_plan_normalization_reports_semantic_raw_conflict_with_dot_path() -> None:
    settings = Settings(
        recipe=Recipe(
            name="contract-conflict",
            task="text_generation",
            model={"pretrained": "hf://some-org/some-model"},
            adapter=SettingsComponentSpec(
                component="LoRA",
                kwargs={"target": ["attn.q"], "target_modules": ["q_proj"]},
            ),
            data=_data_spec(),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="contract", description="conflict"),
        ),
        config=Config(),
    )

    with pytest.raises(ValueError, match=r"recipe\.adapter.*target.*target_modules"):
        AssemblyPlanner.from_run_plan(_run_plan(settings))
