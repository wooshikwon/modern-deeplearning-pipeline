"""Factory._resolve_semantic / _resolve_semantic_from_config 단위 테스트.

spec §3.2 verify 상세를 전부 커버한다:
- _resolve_semantic: Llama/BERT/ViT-timm mock 기반 semantic → raw 번역
- _resolve_semantic_from_config: QLoRA 경로, AutoConfig mock
- semantic 키 없는 config는 변경 없이 반환 (기존 raw 경로 회귀)
- target과 target_modules 동시 존재는 Factory가 ValueError로 차단

설계 결정 명시:
  "target과 target_modules 동시 존재는 Factory의 _resolve_semantic이 차단한다.
   apply_lora 내부에서는 이 검증을 하지 않는다 — apply_lora는 semantic routing을
   전혀 모르는 하위 소비자이기 때문이다."
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from mdp.factory.factory import Factory
from mdp.factory.planner import AssemblyPlanner
from mdp.settings.plan import SettingsPlan
from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    Recipe,
    Settings,
    TrainingSpec,
)


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────


def _mock_hf_model(model_type: str) -> nn.Module:
    """HF model.config.model_type를 흉내내는 Mock 모델."""
    model = MagicMock(spec=nn.Module)
    model.config = MagicMock()
    model.config.model_type = model_type
    del model.default_cfg
    return model


def _mock_timm_model(architecture: str) -> nn.Module:
    """timm model.default_cfg["architecture"]를 흉내내는 Mock 모델."""
    model = MagicMock(spec=nn.Module)
    del model.config
    model.default_cfg = {"architecture": architecture}
    return model


def _make_factory() -> Factory:
    """최소 Settings으로 Factory 인스턴스를 생성한다."""
    settings = MagicMock(spec=Settings)
    return Factory(settings)


def _settings_plan_for_recipe(recipe: Recipe) -> SettingsPlan:
    return SettingsPlan(
        command="train",
        mode="sft",
        settings=Settings(recipe=recipe, config=Config()),
        recipe_path=None,
        config_path=None,
        artifact_dir=None,
        overrides=(),
        callback_configs=(),
        validation_scope="training",
        distributed_intent=False,
    )


# ──────────────────────────────────────────────────────────────────────
# _resolve_semantic 테스트 (일반 경로)
# ──────────────────────────────────────────────────────────────────────


class TestResolveSemantic:
    """Factory._resolve_semantic의 semantic → raw 번역을 검증한다."""

    def test_llama_adapter_target(self) -> None:
        """Llama mock + adapter target → target_modules로 번역."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        adapter_config: dict[str, Any] = {
            "_component_": "LoRA",
            "r": 16,
            "target": ["attn.q", "attn.v"],
        }

        head_out, adapter_out = factory._resolve_semantic(model, None, adapter_config)

        assert head_out is None
        assert adapter_out is not None
        assert adapter_out["target_modules"] == ["q_proj", "v_proj"]
        assert "target" not in adapter_out
        # 다른 키는 보존
        assert adapter_out["_component_"] == "LoRA"
        assert adapter_out["r"] == 16

    def test_bert_adapter_target(self) -> None:
        """BERT mock + 동일 semantic → [query, value]로 번역."""
        factory = _make_factory()
        model = _mock_hf_model("bert")
        adapter_config: dict[str, Any] = {"target": ["attn.q", "attn.v"]}

        _, adapter_out = factory._resolve_semantic(model, None, adapter_config)

        assert adapter_out is not None
        assert adapter_out["target_modules"] == ["query", "value"]

    def test_vit_timm_head_slot(self) -> None:
        """ViT-timm mock + head slot → _target_attr로 번역."""
        factory = _make_factory()
        model = _mock_timm_model("vit_base_patch16_224")
        head_config: dict[str, Any] = {
            "_component_": "ClassificationHead",
            "slot": "head.cls",
            "num_classes": 10,
        }

        head_out, adapter_out = factory._resolve_semantic(model, head_config, None)

        assert adapter_out is None
        assert head_out is not None
        assert head_out["_target_attr"] == "head"
        assert "slot" not in head_out
        # 다른 키는 보존
        assert head_out["_component_"] == "ClassificationHead"
        assert head_out["num_classes"] == 10

    def test_llama_save_modules(self) -> None:
        """Llama mock + save → modules_to_save로 번역."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        adapter_config: dict[str, Any] = {
            "save": ["head.lm", "embed.token"],
        }

        _, adapter_out = factory._resolve_semantic(model, None, adapter_config)

        assert adapter_out is not None
        assert adapter_out["modules_to_save"] == ["lm_head", "embed_tokens"]
        assert "save" not in adapter_out

    def test_no_semantic_keys_passthrough(self) -> None:
        """semantic 키 없는 config는 변경 없이 반환 (기존 raw 경로 회귀)."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        head_config: dict[str, Any] = {
            "_component_": "ClassificationHead",
            "_target_attr": "lm_head",
        }
        adapter_config: dict[str, Any] = {
            "_component_": "LoRA",
            "target_modules": ["q_proj", "v_proj"],
        }

        head_out, adapter_out = factory._resolve_semantic(
            model, head_config, adapter_config,
        )

        # 원본과 동일 (dict copy가 아닌 원본 반환)
        assert head_out is head_config
        assert adapter_out is adapter_config

    def test_none_configs_passthrough(self) -> None:
        """None config는 None으로 반환."""
        factory = _make_factory()
        model = _mock_hf_model("llama")

        head_out, adapter_out = factory._resolve_semantic(model, None, None)

        assert head_out is None
        assert adapter_out is None

    def test_both_head_slot_and_adapter_target(self) -> None:
        """head slot과 adapter target 동시 resolve."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        head_config: dict[str, Any] = {"slot": "head.lm"}
        adapter_config: dict[str, Any] = {"target": ["attn.q", "attn.v"]}

        head_out, adapter_out = factory._resolve_semantic(
            model, head_config, adapter_config,
        )

        assert head_out is not None
        assert head_out["_target_attr"] == "lm_head"
        assert adapter_out is not None
        assert adapter_out["target_modules"] == ["q_proj", "v_proj"]

    def test_target_and_target_modules_conflict_raises(self) -> None:
        """target과 target_modules 동시 지정 → ValueError."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        adapter_config: dict[str, Any] = {
            "target": ["attn.q"],
            "target_modules": ["q_proj"],
        }

        with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
            factory._resolve_semantic(model, None, adapter_config)

    def test_save_and_modules_to_save_conflict_raises(self) -> None:
        """save와 modules_to_save 동시 지정 → ValueError."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        adapter_config: dict[str, Any] = {
            "save": ["head.lm"],
            "modules_to_save": ["lm_head"],
        }

        with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
            factory._resolve_semantic(model, None, adapter_config)

    def test_slot_and_target_attr_conflict_raises(self) -> None:
        """slot과 _target_attr 동시 지정 → ValueError."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        head_config: dict[str, Any] = {
            "slot": "head.lm",
            "_target_attr": "lm_head",
        }

        with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
            factory._resolve_semantic(model, head_config, None)

    def test_original_config_not_mutated(self) -> None:
        """원본 config dict가 변경되지 않아야 한다."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        adapter_config: dict[str, Any] = {
            "target": ["attn.q"],
            "r": 16,
        }
        original = dict(adapter_config)

        factory._resolve_semantic(model, None, adapter_config)

        assert adapter_config == original

    def test_adapter_target_with_wildcard(self) -> None:
        """wildcard target (attn.*) 확장."""
        factory = _make_factory()
        model = _mock_hf_model("llama")
        adapter_config: dict[str, Any] = {"target": ["attn.*"]}

        _, adapter_out = factory._resolve_semantic(model, None, adapter_config)

        assert adapter_out is not None
        assert set(adapter_out["target_modules"]) == {
            "q_proj", "k_proj", "v_proj", "o_proj",
        }


class TestAssemblyPlanSemanticBoundary:
    """AssemblyPlanner keeps semantic config unresolved for materialization."""

    def test_sft_head_and_adapter_semantic_keys_are_preserved(self) -> None:
        recipe = Recipe(
            name="semantic-plan",
            task="image_classification",
            model={
                "_component_": "transformers.AutoModel",
                "pretrained": "hf://google/vit-base-patch16-224",
            },
            head={"_component_": "ClassificationHead", "slot": "head.cls"},
            adapter={"_component_": "LoRA", "target": ["attn.q", "attn.v"]},
            data=DataSpec(
                dataset={"_component_": "ImageClassificationDataset", "source": "cifar10"},
                collator={"_component_": "VisionCollator"},
            ),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="test", description="semantic plan"),
        )

        plan = AssemblyPlanner.from_settings_plan(_settings_plan_for_recipe(recipe))
        model_node = plan.models[0]

        assert model_node.head is not None
        assert model_node.adapter is not None
        assert model_node.head.config["slot"] == "head.cls"
        assert "_target_attr" not in model_node.head.config
        assert model_node.adapter.config["target"] == ["attn.q", "attn.v"]
        assert "target_modules" not in model_node.adapter.config

    def test_qlora_semantic_keys_are_not_family_resolved(self) -> None:
        recipe = Recipe(
            name="qlora-plan",
            task="text_generation",
            model={
                "_component_": "mdp.models.language.CausalLM",
                "pretrained": "hf://Qwen/Qwen2.5-7B",
            },
            adapter={
                "_component_": "QLoRA",
                "target": "all-linear",
                "save": ["head.lm"],
            },
            data=DataSpec(
                dataset={"_component_": "HuggingFaceDataset", "source": "wikitext"},
                collator={"_component_": "CausalLMCollator", "tokenizer": "gpt2"},
            ),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="test", description="qlora plan"),
        )

        plan = AssemblyPlanner.from_settings_plan(_settings_plan_for_recipe(recipe))
        adapter = plan.models[0].adapter

        assert adapter is not None
        assert adapter.config["target"] == "all-linear"
        assert adapter.config["save"] == ["head.lm"]
        assert "target_modules" not in adapter.config
        assert "modules_to_save" not in adapter.config


# ──────────────────────────────────────────────────────────────────────
# _resolve_semantic_from_config 테스트 (QLoRA 경로)
# ──────────────────────────────────────────────────────────────────────


class TestResolveSemanticFromConfig:
    """Factory._resolve_semantic_from_config (QLoRA용) 을 검증한다."""

    def test_qlora_llama_target(self) -> None:
        """pretrained hf:// URI + target → family 추정 → target_modules 번역."""
        factory = _make_factory()
        model_config: dict[str, Any] = {
            "pretrained": "hf://meta-llama/Meta-Llama-3-8B",
            "_component_": "transformers.AutoModelForCausalLM",
        }
        adapter_config: dict[str, Any] = {
            "_component_": "QLoRA",
            "target": ["attn.q"],
        }

        # AutoConfig.from_pretrained를 mock하여 네트워크 없이 테스트
        mock_config = MagicMock()
        mock_config.model_type = "llama"

        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=mock_config,
        ):
            result = factory._resolve_semantic_from_config(
                model_config, adapter_config,
            )

        assert result["target_modules"] == ["q_proj"]
        assert "target" not in result
        assert result["_component_"] == "QLoRA"

    def test_qlora_no_semantic_keys_passthrough(self) -> None:
        """semantic 키 없는 config → 기존 경로 그대로."""
        factory = _make_factory()
        model_config: dict[str, Any] = {"pretrained": "hf://some/model"}
        adapter_config: dict[str, Any] = {
            "_component_": "QLoRA",
            "target_modules": ["q_proj", "v_proj"],
        }

        result = factory._resolve_semantic_from_config(
            model_config, adapter_config,
        )

        assert result["target_modules"] == ["q_proj", "v_proj"]
        assert result["_component_"] == "QLoRA"

    def test_qlora_target_and_target_modules_conflict_raises(self) -> None:
        """QLoRA 경로에서도 target + target_modules 동시 지정 → ValueError."""
        factory = _make_factory()
        model_config: dict[str, Any] = {"pretrained": "hf://meta-llama/Llama-3-8B"}
        adapter_config: dict[str, Any] = {
            "target": ["attn.q"],
            "target_modules": ["q_proj"],
        }

        mock_config = MagicMock()
        mock_config.model_type = "llama"

        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=mock_config,
        ):
            with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
                factory._resolve_semantic_from_config(
                    model_config, adapter_config,
                )

    def test_qlora_save_semantic(self) -> None:
        """QLoRA 경로에서 save → modules_to_save 번역."""
        factory = _make_factory()
        model_config: dict[str, Any] = {"pretrained": "hf://meta-llama/Llama-3-8B"}
        adapter_config: dict[str, Any] = {
            "save": ["head.lm"],
        }

        mock_config = MagicMock()
        mock_config.model_type = "llama"

        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=mock_config,
        ):
            result = factory._resolve_semantic_from_config(
                model_config, adapter_config,
            )

        assert result["modules_to_save"] == ["lm_head"]
        assert "save" not in result

    def test_qlora_non_hf_uri_raises(self) -> None:
        """hf:// 프로토콜이 아닌 URI → ValueError."""
        factory = _make_factory()
        model_config: dict[str, Any] = {"pretrained": "/local/path/model"}
        adapter_config: dict[str, Any] = {"target": ["attn.q"]}

        with pytest.raises(ValueError, match="hf:// 프로토콜만 지원"):
            factory._resolve_semantic_from_config(
                model_config, adapter_config,
            )

    def test_qlora_unknown_model_type_raises(self) -> None:
        """AutoConfig의 model_type이 _FAMILY_ROUTING에 없으면 ValueError."""
        factory = _make_factory()
        model_config: dict[str, Any] = {"pretrained": "hf://unknown/model"}
        adapter_config: dict[str, Any] = {"target": ["attn.q"]}

        mock_config = MagicMock()
        mock_config.model_type = "totally_unknown_arch"

        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=mock_config,
        ):
            with pytest.raises(ValueError, match="_FAMILY_ROUTING에 없습니다"):
                factory._resolve_semantic_from_config(
                    model_config, adapter_config,
                )

    def test_original_config_not_mutated(self) -> None:
        """원본 adapter_config가 변경되지 않아야 한다."""
        factory = _make_factory()
        model_config: dict[str, Any] = {"pretrained": "hf://meta-llama/Llama-3-8B"}
        adapter_config: dict[str, Any] = {"target": ["attn.q"], "r": 16}
        original = dict(adapter_config)

        mock_config = MagicMock()
        mock_config.model_type = "llama"

        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=mock_config,
        ):
            factory._resolve_semantic_from_config(model_config, adapter_config)

        assert adapter_config == original
