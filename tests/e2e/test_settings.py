"""E2E tests for Settings factory, resolver, and validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from mdp.settings.factory import SettingsFactory
from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import (
    ComputeConfig,
    Config,
    DataSpec,
    MetadataSpec,
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
    ("clip-finetune-custom.yaml", "local-single-gpu.yaml"),
    ("gpt2-finetune-text.yaml", "cloud-gcp-8gpu.yaml"),
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
    adapter: dict | None = None,
    distributed: dict | None = None,
) -> Settings:
    """Build a minimal Settings object for validator tests."""
    recipe = Recipe.model_construct(
        name="test",
        task=task,
        model={"_component_": "test.Model"},
        head=head,
        adapter=adapter,
        data=DataSpec.model_construct(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "test-dataset", "split": "train"},
            collator={"_component_": "mdp.data.collators.CausalLMCollator", "tokenizer": "gpt2"},
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
    assert len(result.errors) > 0
    assert any("CausalLMHead" in e for e in result.errors)


# ---------------------------------------------------------------------------
# CompatValidator — FSDP + QLoRA
# ---------------------------------------------------------------------------


def test_compat_validator_fsdp_qlora() -> None:
    """FSDP strategy with QLoRA adapter should produce an incompatibility error."""
    settings = _make_minimal_settings(
        task="text_generation",
        head={"_component_": "mdp.models.heads.causal_lm.CausalLMHead"},
        adapter={"_component_": "QLoRA", "r": 16, "alpha": 32, "quantization": {"bits": 4}},
        distributed={"strategy": "fsdp"},
    )
    result = CompatValidator().validate(settings)
    assert len(result.errors) > 0
    assert any("FSDP" in e and "QLoRA" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Seq2SeqLMHead alias → CausalLMHead
# ---------------------------------------------------------------------------


def test_seq2seq_alias_resolves_to_causal_lm() -> None:
    """Seq2SeqLMHead alias가 CausalLMHead로 해석되는지 확인."""
    from mdp.models.heads.causal_lm import CausalLMHead

    resolver = ComponentResolver()
    head = resolver.resolve(
        {"_component_": "Seq2SeqLMHead", "hidden_dim": 64, "vocab_size": 100}
    )
    assert isinstance(head, CausalLMHead)


# ---------------------------------------------------------------------------
# CompatValidator — FSDP string variants
# ---------------------------------------------------------------------------


def test_compat_fsdp_variant_detected() -> None:
    """FSDP 변형 문자열(fsdp_full_shard)도 QLoRA 비호환으로 감지."""
    settings = _make_minimal_settings(
        task="text_generation",
        head={"_component_": "mdp.models.heads.causal_lm.CausalLMHead"},
        adapter={"_component_": "QLoRA", "r": 16, "alpha": 32, "quantization": {"bits": 4}},
        distributed={"strategy": "fsdp_full_shard"},
    )
    result = CompatValidator().validate(settings)
    assert len(result.errors) > 0
    assert any("FSDP" in e and "QLoRA" in e for e in result.errors)


def test_compat_strategy_dict_no_crash() -> None:
    """strategy가 dict일 때 검증이 크래시하지 않는다."""
    settings = _make_minimal_settings(
        task="text_generation",
        head={"_component_": "mdp.models.heads.causal_lm.CausalLMHead"},
        adapter={"_component_": "QLoRA", "r": 16, "alpha": 32, "quantization": {"bits": 4}},
        distributed={"strategy": {"_component_": "FSDPStrategy"}},
    )
    # dict strategy should not raise — validator checks isinstance(str)
    result = CompatValidator().validate(settings)
    assert isinstance(result.errors, list)
    assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# BusinessValidator — validate_partial
# ---------------------------------------------------------------------------


def test_validate_partial_subset() -> None:
    """validate_partial이 지정된 검증만 실행한다."""
    settings = _make_minimal_settings(
        task="image_classification",
        head={"_component_": "mdp.models.heads.causal_lm.CausalLMHead"},
        adapter={"_component_": "LoRA", "r": 8, "alpha": 16},
    )
    # adapter-only check should produce no errors (lora config is valid)
    adapter_result = BusinessValidator.validate_partial(settings, checks=["adapter"])
    assert len(adapter_result.errors) == 0

    # head_task check should produce errors (CausalLMHead ≠ image_classification)
    head_result = BusinessValidator.validate_partial(settings, checks=["head_task"])
    assert len(head_result.errors) > 0
    assert any("CausalLMHead" in e for e in head_result.errors)


