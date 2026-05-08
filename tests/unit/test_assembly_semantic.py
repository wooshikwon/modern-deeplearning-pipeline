"""Typed semantic resolution boundary tests for materializer materialization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.specs import ComponentSpec


def _mock_hf_model(model_type: str) -> nn.Module:
    model = MagicMock(spec=nn.Module)
    model.config = MagicMock()
    model.config.model_type = model_type
    del model.default_cfg
    return model


def _materializer() -> AssemblyMaterializer:
    return AssemblyMaterializer()


class TestResolveSemantic:
    def test_llama_adapter_target(self) -> None:
        materializer = _materializer()
        model = _mock_hf_model("llama")
        adapter_config = ComponentSpec.from_config(
            {"_component_": "LoRA", "r": 16, "target": ["attn.q", "attn.v"]}
        )

        head_out, adapter_out = materializer._resolve_semantic(
            model, None, adapter_config
        )

        assert head_out is None
        assert adapter_out is not None
        assert adapter_out.kwargs["target_modules"] == ["q_proj", "v_proj"]
        assert "target" not in adapter_out.kwargs
        assert adapter_out.component == "LoRA"
        assert adapter_out.kwargs["r"] == 16

    def test_no_semantic_keys_passthrough(self) -> None:
        materializer = _materializer()
        model = _mock_hf_model("llama")
        head_config = ComponentSpec.from_config(
            {"_component_": "ClassificationHead", "_target_attr": "lm_head"}
        )
        adapter_config = ComponentSpec.from_config(
            {"_component_": "LoRA", "target_modules": ["q_proj", "v_proj"]}
        )

        head_out, adapter_out = materializer._resolve_semantic(
            model, head_config, adapter_config
        )

        assert head_out is head_config
        assert adapter_out is adapter_config

    def test_target_and_target_modules_conflict_raises(self) -> None:
        materializer = _materializer()
        model = _mock_hf_model("llama")
        adapter_config = ComponentSpec(
            component="LoRA",
            kwargs={"target": ["attn.q"], "target_modules": ["q_proj"]},
            path="recipe.adapter",
        )

        with pytest.raises(ValueError, match="동시에 지정할 수 없습니다"):
            materializer._resolve_semantic(model, None, adapter_config)

    def test_original_config_not_mutated(self) -> None:
        materializer = _materializer()
        model = _mock_hf_model("llama")
        adapter_config = ComponentSpec.from_config(
            {"_component_": "LoRA", "target": ["attn.q"], "r": 16}
        )
        original = dict(adapter_config.kwargs)

        materializer._resolve_semantic(model, None, adapter_config)

        assert adapter_config.kwargs == original


class TestResolveSemanticFromConfig:
    def test_qlora_llama_target(self) -> None:
        materializer = _materializer()
        model_config = ComponentSpec.from_config(
            {
                "pretrained": "hf://meta-llama/Meta-Llama-3-8B",
                "_component_": "transformers.AutoModelForCausalLM",
            }
        )
        adapter_config = ComponentSpec.from_config(
            {"_component_": "QLoRA", "target": ["attn.q"]}
        )
        mock_config = MagicMock()
        mock_config.model_type = "llama"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            result = materializer._resolve_semantic_from_config(
                model_config, adapter_config
            )

        assert result.kwargs["target_modules"] == ["q_proj"]
        assert "target" not in result.kwargs
        assert result.component == "QLoRA"

    def test_qlora_no_semantic_keys_passthrough(self) -> None:
        materializer = _materializer()
        model_config = ComponentSpec.from_config({"pretrained": "hf://some/model"})
        adapter_config = ComponentSpec.from_config(
            {"_component_": "QLoRA", "target_modules": ["q_proj", "v_proj"]}
        )

        result = materializer._resolve_semantic_from_config(
            model_config, adapter_config
        )

        assert result.kwargs["target_modules"] == ["q_proj", "v_proj"]
        assert result.component == "QLoRA"

    def test_original_config_not_mutated(self) -> None:
        materializer = _materializer()
        model_config = ComponentSpec.from_config(
            {"pretrained": "hf://meta-llama/Llama-3-8B"}
        )
        adapter_config = ComponentSpec.from_config(
            {"_component_": "QLoRA", "target": ["attn.q"], "r": 16}
        )
        original = dict(adapter_config.kwargs)
        mock_config = MagicMock()
        mock_config.model_type = "llama"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config):
            materializer._resolve_semantic_from_config(model_config, adapter_config)

        assert adapter_config.kwargs == original
