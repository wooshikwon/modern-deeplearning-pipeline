"""mdp generate CLI 유닛 테스트."""

import pytest

from mdp.cli.generate import _resolve_tokenizer_name


class _FakeSpec:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestResolveTokenizerName:
    def test_from_collator(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator={"_component_": "CausalLMCollator", "tokenizer": "gpt2"},
                    dataset={"_component_": "HuggingFaceDataset"},
                ),
                model={"_component_": "AutoModelForCausalLM"},
            ),
        )
        assert _resolve_tokenizer_name(settings) == "gpt2"

    def test_from_dataset(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator={"_component_": "CausalLMCollator"},
                    dataset={"_component_": "HuggingFaceDataset", "tokenizer": "bert-base"},
                ),
                model={"_component_": "AutoModelForCausalLM"},
            ),
        )
        assert _resolve_tokenizer_name(settings) == "bert-base"

    def test_from_pretrained(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator={"_component_": "CausalLMCollator"},
                    dataset={"_component_": "HuggingFaceDataset"},
                ),
                model={"_component_": "AutoModelForCausalLM", "pretrained": "hf://meta-llama/Meta-Llama-3-8B"},
            ),
        )
        assert _resolve_tokenizer_name(settings) == "meta-llama/Meta-Llama-3-8B"

    def test_no_tokenizer_raises(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator={"_component_": "CausalLMCollator"},
                    dataset={"_component_": "HuggingFaceDataset"},
                ),
                model={"_component_": "CustomModel"},
            ),
        )
        with pytest.raises(ValueError, match="토크나이저를 결정할 수 없습니다"):
            _resolve_tokenizer_name(settings)
