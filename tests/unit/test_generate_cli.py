"""mdp generate CLI 유닛 테스트."""

import pytest

from mdp.cli.generate import _resolve_tokenizer_name
from mdp.settings.components import ComponentSpec, ModelComponentSpec


class _FakeSpec:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestResolveTokenizerName:
    def test_artifact_settings_from_typed_collator(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator=ComponentSpec(
                        component="CausalLMCollator",
                        kwargs={"tokenizer": "gpt2"},
                    ),
                    dataset=ComponentSpec(component="HuggingFaceDataset"),
                ),
                model=ModelComponentSpec(component="AutoModelForCausalLM"),
            ),
        )
        assert _resolve_tokenizer_name(settings) == "gpt2"

    def test_artifact_settings_from_typed_dataset(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator=ComponentSpec(component="CausalLMCollator"),
                    dataset=ComponentSpec(
                        component="HuggingFaceDataset",
                        kwargs={"tokenizer": "bert-base"},
                    ),
                ),
                model=ModelComponentSpec(component="AutoModelForCausalLM"),
            ),
        )
        assert _resolve_tokenizer_name(settings) == "bert-base"

    def test_artifact_settings_from_typed_model_pretrained(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator=ComponentSpec(component="CausalLMCollator"),
                    dataset=ComponentSpec(component="HuggingFaceDataset"),
                ),
                model=ModelComponentSpec(
                    component="AutoModelForCausalLM",
                    pretrained="hf://meta-llama/Meta-Llama-3-8B",
                ),
            ),
        )
        assert _resolve_tokenizer_name(settings) == "meta-llama/Meta-Llama-3-8B"

    def test_no_tokenizer_raises(self):
        settings = _FakeSpec(
            recipe=_FakeSpec(
                data=_FakeSpec(
                    collator=ComponentSpec(component="CausalLMCollator"),
                    dataset=ComponentSpec(component="HuggingFaceDataset"),
                ),
                model=ModelComponentSpec(component="CustomModel"),
            ),
        )
        with pytest.raises(ValueError, match="토크나이저를 결정할 수 없습니다"):
            _resolve_tokenizer_name(settings)
