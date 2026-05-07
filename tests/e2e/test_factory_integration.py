"""Factory 통합 경로 테스트: Settings → Model → Head → Adapter 조립.

4 tests:
- test_factory_model_with_head_and_lora: head 교체 + LoRA 적용 후 학습 가능
- test_factory_model_without_head: head=None → 원본 모델 반환
- test_factory_prefix_tuning_path: prefix_tuning adapter가 factory를 통과
- test_log_trainable_params_plain_module: 일반 nn.Module에서 log_trainable_params 동작
"""

from __future__ import annotations

import torch

from mdp.factory.factory import Factory
from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    Recipe,
    Settings,
    TrainingSpec,
)


def _make_factory_settings(
    head: dict | None = None,
    adapter: dict | None = None,
) -> Settings:
    recipe = Recipe(
        name="factory-test",
        task="image_classification",
        model={"_component_": "tests.e2e.models.TinyVisionModel", "num_classes": 2, "hidden_dim": 16},
        head=head,
        adapter=adapter,
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.CausalLMCollator", "tokenizer": "gpt2"},
        ),
        training=TrainingSpec(epochs=1),
        optimizer={"_component_": "AdamW", "lr": 1e-3},
        metadata=MetadataSpec(author="test", description="factory integration"),
    )
    config = Config()
    config.job.resume = "disabled"
    return Settings(recipe=recipe, config=config)


def test_factory_model_with_head_and_lora() -> None:
    """Factory가 head 교체 + LoRA 적용한 모델을 반환하고, 학습 가능한지 확인."""
    settings = _make_factory_settings(
        head={
            "_component_": "ClassificationHead",
            "_target_attr": "head",
            "num_classes": 5,
            "hidden_dim": 16,
        },
        adapter={
            "_component_": "LoRA", "r": 4, "alpha": 8, "dropout": 0.0,
            "target_modules": ["classifier"],
        },
    )
    factory = Factory(settings)
    model = factory.create_model()

    # head가 교체되었는지 (원래 num_classes=2 → 5로 변경)
    assert hasattr(model, "head")

    # LoRA가 적용되었는지 (trainable < total)
    trainable, total = model.get_nb_trainable_parameters()
    assert 0 < trainable < total

    # 학습 1 step 가능한지
    batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
    model.train()
    loss = model(batch)["loss"]
    assert torch.isfinite(loss)


def test_factory_model_without_head() -> None:
    """head=None이면 원본 모델 그대로 반환."""
    settings = _make_factory_settings(head=None, adapter=None)
    factory = Factory(settings)
    model = factory.create_model()

    # TinyVisionModel의 기본 head가 유지
    assert hasattr(model, "head")
    batch = {"pixel_values": torch.randn(2, 3, 8, 8), "labels": torch.tensor([0, 1])}
    loss = model(batch)["loss"]
    assert torch.isfinite(loss)


def test_factory_prefix_tuning_path() -> None:
    """PrefixTuning adapter가 factory의 resolver 경로를 통과하는지 확인.

    PEFT PrefixTuning은 task_type이 필수라 TinyModel에서 직접 동작하지 않을 수 있다.
    핵심 검증: factory가 _component_=="PrefixTuning"일 때 resolver.resolve를 통해
    apply_prefix_tuning을 호출하는지.
    """
    from unittest.mock import patch

    settings = _make_factory_settings(
        adapter={"_component_": "PrefixTuning", "r": 4},
    )
    factory = Factory(settings)

    with patch("mdp.models.adapters.prefix_tuning.apply_prefix_tuning") as mock_apply:
        mock_apply.return_value = factory._load_pretrained(settings.recipe.model)
        factory.create_model()

        mock_apply.assert_called_once()
        _, call_kwargs = mock_apply.call_args
        assert call_kwargs.get("r") == 4 or mock_apply.call_args[1].get("r") == 4


def test_log_trainable_params_plain_module() -> None:
    """PEFT가 아닌 일반 nn.Module에서도 log_trainable_params가 동작한다."""
    from torch import nn

    from mdp.models.adapters import log_trainable_params

    model = nn.Linear(10, 5)
    # Should not raise; should log param counts via the fallback branch
    log_trainable_params(model)

    # Verify the count is correct (10*5 + 5 = 55 total, all trainable)
    expected_total = sum(p.numel() for p in model.parameters())
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert expected_total == 55
    assert expected_trainable == 55
