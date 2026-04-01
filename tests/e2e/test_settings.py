"""E2E tests for Settings factory, resolver, and validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdp.settings.factory import SettingsFactory
from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import (
    AdapterSpec,
    ComputeConfig,
    Config,
    DataSpec,
    MetadataSpec,
    ModelSpec,
    MonitoringSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from mdp.settings.validation.business_validator import BusinessValidator
from mdp.settings.validation.compat_validator import CompatValidator

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
RECIPES = FIXTURES / "recipes"
CONFIGS = FIXTURES / "configs"


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

_YAML_PAIRS = [
    ("vit-lora-cifar10.yaml", "local-single-gpu.yaml"),
    ("gpt2-finetune-text.yaml", "remote-4gpu-ddp.yaml"),
    ("yolo-detection-custom.yaml", "local-single-gpu-detection.yaml"),
    ("clip-finetune-custom.yaml", "cloud-gcp-8gpu.yaml"),
    ("qwen25-qlora-instruct.yaml", "multi-node-2x4gpu-deepspeed.yaml"),
]


@pytest.mark.parametrize("recipe_file,config_file", _YAML_PAIRS)
def test_yaml_parsing(recipe_file: str, config_file: str) -> None:
    """SettingsFactory.for_training succeeds for each recipe-config pair."""
    factory = SettingsFactory()
    settings = factory.for_training(
        str(RECIPES / recipe_file),
        str(CONFIGS / config_file),
    )
    assert isinstance(settings, Settings)
    assert settings.recipe.name
    assert settings.recipe.task


# ---------------------------------------------------------------------------
# ComponentResolver
# ---------------------------------------------------------------------------


def test_component_resolver_alias() -> None:
    """Resolver maps 'ClassificationHead' alias to the full class path."""
    resolver = ComponentResolver()
    # resolve_partial returns (class, kwargs) without instantiating
    resolved_path = resolver._resolve_alias("ClassificationHead")
    assert resolved_path == "mdp.models.heads.classification.ClassificationHead"


# ---------------------------------------------------------------------------
# BusinessValidator — head-task mismatch
# ---------------------------------------------------------------------------


def _make_minimal_settings(
    task: str = "image_classification",
    head: dict | None = None,
    adapter: AdapterSpec | None = None,
    distributed: dict | None = None,
) -> Settings:
    """Build a minimal Settings object for validator tests."""
    recipe = Recipe.model_construct(
        name="test",
        task=task,
        model=ModelSpec(class_path="test.Model"),
        head=head,
        adapter=adapter,
        data=DataSpec.model_construct(
            source="test-dataset",
            fields={},
            format="auto",
            split="train",
        ),
        training=TrainingSpec(epochs=1),
        optimizer={"_component_": "torch.optim.SGD", "lr": 0.01},
        scheduler=None,
        loss=None,
        evaluation=None,
        generation=None,
        monitoring=MonitoringSpec(),
        callbacks=[],
        metadata=MetadataSpec(author="test", description="test"),
    )
    config = Config.model_construct(
        environment={"name": "local"},
        compute=ComputeConfig(distributed=distributed),
        mlflow=None,
        storage=None,
        serving=None,
        job=None,
    )
    return Settings.model_construct(recipe=recipe, config=config)


def test_business_validator_head_task_mismatch() -> None:
    """image_classification + CausalLMHead should produce an error."""
    settings = _make_minimal_settings(
        task="image_classification",
        head={"_component_": "mdp.models.heads.causal_lm.CausalLMHead"},
    )
    result = BusinessValidator().validate(settings)
    assert result.has_errors
    assert any("CausalLMHead" in e for e in result.errors)


# ---------------------------------------------------------------------------
# CompatValidator — FSDP + QLoRA
# ---------------------------------------------------------------------------


def test_compat_validator_fsdp_qlora() -> None:
    """FSDP strategy with QLoRA adapter should produce an incompatibility error."""
    settings = _make_minimal_settings(
        task="text_generation",
        head={"_component_": "mdp.models.heads.causal_lm.CausalLMHead"},
        adapter=AdapterSpec(method="qlora", r=16, alpha=32),
        distributed={"strategy": "fsdp"},
    )
    result = CompatValidator().validate(settings)
    assert result.has_errors
    assert any("FSDP" in e and "QLoRA" in e for e in result.errors)
