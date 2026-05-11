"""AssemblyMaterializer contract tests."""

from __future__ import annotations

from pathlib import Path
from unittest import mock
import warnings

import pytest
import torch
from torch import Tensor, nn

from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.planner import AssemblyPlanner
from mdp.models.base import BaseModel
from mdp.settings.components import ComponentSpec, ModelComponentSpec
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


class FamilyAwareHeadModel(BaseModel):
    """Tiny model with a detectable HF-style family for head semantic tests."""

    _block_classes = None

    def __init__(self) -> None:
        super().__init__()
        self.config = mock.MagicMock()
        self.config.model_type = "bert"
        self.classifier = nn.Linear(4, 2)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"logits": self.classifier(batch["input"])}


def _data_spec(tmp_path: Path) -> DataSpec:
    return DataSpec(
        dataset={
            "_component_": "mdp.data.datasets.HuggingFaceDataset",
            "source": str(tmp_path / "train.jsonl"),
            "split": "train",
        },
        val_dataset={
            "_component_": "mdp.data.datasets.HuggingFaceDataset",
            "source": str(tmp_path / "val.jsonl"),
            "split": "validation",
        },
        collator={"_component_": "mdp.data.collators.CausalLMCollator", "tokenizer": "gpt2"},
        sampler={"_component_": "mdp.data.samplers.LengthGroupedBatchSampler"},
        dataloader={"batch_size": 2, "num_workers": 0},
    )


def _sft_settings(tmp_path: Path) -> Settings:
    return Settings(
        recipe=Recipe(
            name="materializer-sft",
            task="image_classification",
            model={
                "_component_": "tests.e2e.models.TinyVisionModel",
                "num_classes": 2,
                "hidden_dim": 16,
            },
            head={
                "_component_": "ClassificationHead",
                "_target_attr": "head",
                "num_classes": 5,
                "hidden_dim": 16,
                "dropout": 0.0,
            },
            data=_data_spec(tmp_path),
            training=TrainingSpec(epochs=1),
            optimizer={"_component_": "AdamW", "lr": 1e-3},
            metadata=MetadataSpec(author="contract", description="materializer sft"),
        ),
        config=Config(),
    )


def _run_plan(
    settings: Settings,
    *,
    mode: str = "sft",
    callback_configs: tuple[ComponentSpec, ...] = (),
) -> RunPlan:
    return RunPlan(
        command="rl-train" if mode == "rl" else "train",
        mode=mode,
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=callback_configs,
        validation_scope="training",
        distributed_intent=bool(settings.config.compute.distributed),
    )


def _assembly_plan(settings: Settings, *, mode: str = "sft"):
    return AssemblyPlanner.from_run_plan(_run_plan(settings, mode=mode))


def test_policy_model_materialization_shape(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    plan = _assembly_plan(settings)

    model = AssemblyMaterializer(plan).materialize_policy_model()

    assert type(model).__name__ == "TinyVisionModel"
    assert model.num_classes == 2
    assert model.hidden_dim == 16
    assert model.head.classifier.out_features == 5
    batch = {
        "pixel_values": torch.randn(2, 3, 8, 8),
        "labels": torch.tensor([0, 1]),
    }
    assert torch.isfinite(model(batch)["loss"])


def test_materializer_reuses_cache_for_policy_model(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    materializer = AssemblyMaterializer(_assembly_plan(settings))
    model = materializer.materialize_policy_model()

    assert materializer.materialize_policy_model() is model


def test_dataloader_materialization_delegates_plan_data_node(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    settings.config.compute.distributed = {"strategy": "ddp"}
    plan = _assembly_plan(settings)
    loaders = {"train": object(), "val": object()}

    with mock.patch(
        "mdp.data.dataloader.create_dataloaders",
        return_value=loaders,
    ) as create_dataloaders:
        result = AssemblyMaterializer(plan).materialize_dataloaders()

    assert result == loaders
    data_node = plan.data
    assert create_dataloaders.call_args.kwargs == {
        "dataset_config": data_node.dataset,
        "collator_config": data_node.collator,
        "dataloader_config": settings.recipe.data.dataloader.model_dump(),
        "val_dataset_config": data_node.val_dataset,
        "sampler_config": data_node.sampler,
        "distributed": True,
    }


def test_callback_materialization_preserves_typed_component_spec(
    tmp_path: Path,
) -> None:
    settings = _sft_settings(tmp_path)
    callback = ComponentSpec(
        component="mdp.training.callbacks.checkpoint.ModelCheckpoint",
        kwargs={
            "dirpath": str(tmp_path / "ckpt"),
            "every_n_steps": 1,
            "save_top_k": 1,
        },
        path="callbacks[0]",
    )
    plan = AssemblyPlanner.from_run_plan(
        _run_plan(settings, callback_configs=(callback,))
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        callbacks = AssemblyMaterializer(plan).materialize_callbacks()

    assert [type(cb).__name__ for cb in callbacks] == ["ModelCheckpoint"]
    assert not any(
        issubclass(item.category, DeprecationWarning)
        and "raw dict support is deprecated" in str(item.message)
        for item in caught
    )


def test_rl_model_materialization_returns_role_dict(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    settings.recipe.rl = RLSpec(
        algorithm={"_component_": "DPO"},
        models={
            "policy": {
                "_component_": "tests.e2e.models.TinyLanguageModel",
                "optimizer": {"_component_": "AdamW", "lr": 1e-4},
            },
            "reference": {
                "_component_": "tests.e2e.models.TinyLanguageModel",
                "freeze": True,
            },
        },
    )
    plan = _assembly_plan(settings, mode="rl")

    materialized = AssemblyMaterializer(plan).materialize_models()

    assert set(materialized) == {"policy", "reference"}
    assert all(type(model).__name__ == "TinyLanguageModel" for model in materialized.values())
    assert materialized["policy"] is not materialized["reference"]


def test_sft_training_bundle_materializes_public_shape(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    plan = _assembly_plan(settings)
    loaders = {"train": [object()], "val": [object()]}

    with mock.patch(
        "mdp.data.dataloader.create_dataloaders",
        return_value=loaders,
    ):
        bundle = AssemblyMaterializer(plan).materialize_sft_training_bundle()

    assert bundle.settings is settings
    assert type(bundle.model).__name__ == "TinyVisionModel"
    assert bundle.train_loader == loaders["train"]
    assert bundle.val_loader == loaders["val"]
    assert bundle.optimizer is not None
    assert bundle.loss_fn is None


def test_pretrained_only_model_node_uses_pretrained_resolver_boundary(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    settings.recipe.model = ModelComponentSpec.from_yaml_dict(
        {"pretrained": "hf://some-org/some-model"}
    )
    plan = _assembly_plan(settings)
    fake_model = mock.MagicMock()
    fake_model.from_pretrained = mock.MagicMock()

    with mock.patch(
        "mdp.models.pretrained.PretrainedResolver.load",
        return_value=fake_model,
    ) as load:
        result = AssemblyMaterializer(plan).materialize_policy_model()

    assert result is fake_model
    load.assert_called_once_with("hf://some-org/some-model")


def test_qlora_route_keeps_pretrained_boundary_and_resolves_semantic_config(
    tmp_path: Path,
) -> None:
    settings = _sft_settings(tmp_path)
    settings.recipe.model = ModelComponentSpec.from_yaml_dict(
        {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://meta-llama/Meta-Llama-3-8B",
            "torch_dtype": "bfloat16",
        }
    )
    settings.recipe.adapter = ComponentSpec(
        component="QLoRA",
        kwargs={"target": ["attn.q"], "save": ["head.lm"]},
        path="recipe.adapter",
    )
    plan = _assembly_plan(settings)
    fake_model = mock.MagicMock()
    fake_config = mock.MagicMock()
    fake_config.model_type = "llama"

    with mock.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=fake_config,
    ), mock.patch(
        "mdp.models.adapters.qlora.apply_qlora",
        return_value=fake_model,
    ) as apply_qlora:
        result = AssemblyMaterializer(plan).materialize_policy_model()

    assert result is fake_model
    assert apply_qlora.call_args.kwargs["model_name_or_path"] == (
        "meta-llama/Meta-Llama-3-8B"
    )
    assert apply_qlora.call_args.kwargs["class_path"] == (
        "transformers.AutoModelForCausalLM"
    )
    assert apply_qlora.call_args.kwargs["target_modules"] == ["q_proj"]
    assert apply_qlora.call_args.kwargs["modules_to_save"] == ["lm_head"]
    assert apply_qlora.call_args.kwargs["torch_dtype"] is torch.bfloat16


@pytest.mark.parametrize(
    ("adapter_update", "message"),
    [
        (
            {"target": ["attn.q"], "target_modules": ["q_proj"]},
            "target.*target_modules",
        ),
        (
            {"save": ["head.lm"], "modules_to_save": ["lm_head"]},
            "save.*modules_to_save",
        ),
    ],
)
def test_materializer_rejects_mixed_semantic_and_raw_adapter_keys(
    tmp_path: Path,
    adapter_update: dict,
    message: str,
) -> None:
    settings = _sft_settings(tmp_path)
    settings.recipe.model = ModelComponentSpec.from_yaml_dict(
        {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://meta-llama/Meta-Llama-3-8B",
        }
    )
    settings.recipe.adapter = ComponentSpec(
        component="QLoRA",
        kwargs=adapter_update,
        path="recipe.adapter",
    )
    fake_config = mock.MagicMock()
    fake_config.model_type = "llama"

    with mock.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=fake_config,
    ), pytest.raises(ValueError, match=message):
        AssemblyMaterializer(_assembly_plan(settings)).materialize_policy_model()


def test_materializer_rejects_mixed_semantic_and_raw_head_keys(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    settings.recipe.model = ModelComponentSpec.from_yaml_dict(
        {"_component_": "tests.contract.test_materializer.FamilyAwareHeadModel"}
    )
    settings.recipe.head = ComponentSpec(
        component="ClassificationHead",
        kwargs={
            "slot": "head.cls",
            "_target_attr": "head",
            "num_classes": 5,
            "hidden_dim": 16,
        },
        path="recipe.head",
    )

    with pytest.raises(ValueError, match="slot.*_target_attr"):
        AssemblyMaterializer(_assembly_plan(settings)).materialize_policy_model()


def test_materializer_semantic_resolution_does_not_mutate_settings(
    tmp_path: Path,
) -> None:
    settings = _sft_settings(tmp_path)
    settings.recipe.model = ModelComponentSpec.from_yaml_dict(
        {
            "_component_": "transformers.AutoModelForCausalLM",
            "pretrained": "hf://meta-llama/Meta-Llama-3-8B",
            "torch_dtype": "bfloat16",
        }
    )
    settings.recipe.adapter = ComponentSpec(
        component="QLoRA",
        kwargs={"target": ["attn.q"], "save": ["head.lm"]},
        path="recipe.adapter",
    )
    original_adapter = settings.recipe.adapter.to_yaml_dict()
    plan = _assembly_plan(settings)
    fake_model = mock.MagicMock()
    fake_config = mock.MagicMock()
    fake_config.model_type = "llama"

    with mock.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=fake_config,
    ), mock.patch(
        "mdp.models.adapters.qlora.apply_qlora",
        return_value=fake_model,
    ):
        AssemblyMaterializer(plan).materialize_policy_model()

    assert settings.recipe.adapter.to_yaml_dict() == original_adapter
    policy = plan.models[0]
    assert policy.adapter is not None
    assert policy.adapter.to_dict() == original_adapter


def test_materializer_requires_matching_plan_kind(tmp_path: Path) -> None:
    settings = _sft_settings(tmp_path)
    plan = _assembly_plan(settings)

    with pytest.raises(ValueError, match="rl_training"):
        AssemblyMaterializer(plan).materialize_rl_training_bundle()
