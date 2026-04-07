"""DPO RL 학습 통합 테스트.

3 tests:
- test_dpo_loss_computation: DPO loss가 올바른 값을 반환하는지
- test_rl_trainer_dpo: RLTrainer가 DPO로 학습 완료되는지
- test_rl_recipe_validation: RL Recipe 검증 (models.policy 필수 등)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    ModelSpec,
    RLModelSpec,
    RLSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from mdp.training.losses.rl import DPOLoss, compute_log_probs


def test_dpo_loss_computation() -> None:
    """DPOLoss가 올바른 loss를 반환하는지."""
    batch_size, seq_len, vocab = 2, 8, 32

    trainable_out = {
        "policy": {
            "chosen_logits": torch.randn(batch_size, seq_len, vocab),
            "rejected_logits": torch.randn(batch_size, seq_len, vocab),
        }
    }
    frozen_out = {
        "reference": {
            "chosen_logits": torch.randn(batch_size, seq_len, vocab),
            "rejected_logits": torch.randn(batch_size, seq_len, vocab),
        }
    }
    batch = {
        "chosen_labels": torch.randint(0, vocab, (batch_size, seq_len)),
        "rejected_labels": torch.randint(0, vocab, (batch_size, seq_len)),
    }

    dpo = DPOLoss(beta=0.1)
    losses = dpo.compute_loss(trainable_out, frozen_out, batch)

    assert "policy" in losses
    assert torch.isfinite(losses["policy"])
    assert losses["policy"].shape == ()  # scalar


def test_compute_log_probs() -> None:
    """compute_log_probs가 올바른 shape과 masking을 하는지."""
    logits = torch.randn(2, 10, 32)
    labels = torch.randint(0, 32, (2, 10))
    labels[0, 5:] = -100  # 마스킹

    lp = compute_log_probs(logits, labels)
    assert lp.shape == (2, 9)  # shifted
    # 마스킹된 위치는 0
    assert (lp[0, 4:] == 0).all()


def test_rl_trainer_dpo() -> None:
    """RLTrainer가 TinyModel로 DPO 3 step 학습을 완료하는지."""
    from tests.e2e.datasets import ListDataLoader
    from mdp.training.rl_trainer import RLTrainer

    # TinyModel: 간단한 LM head
    class TinyLM(nn.Module):
        def __init__(self, vocab=32, hidden=16):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids, attention_mask=None):
            h = self.embed(input_ids)
            logits = self.head(h)
            return type("Out", (), {"logits": logits})()

    # preference 배치 생성
    def make_pref_batches(n, batch_size, seq_len=8, vocab=32):
        batches = []
        for _ in range(n):
            batches.append({
                "chosen_input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
                "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "chosen_labels": torch.randint(0, vocab, (batch_size, seq_len)),
                "rejected_input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
                "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "rejected_labels": torch.randint(0, vocab, (batch_size, seq_len)),
            })
        return batches

    # Settings
    recipe = Recipe(
        name="dpo-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": RLModelSpec(
                    class_path="tests.e2e.test_rl_dpo.TinyLM",
                    optimizer={"_component_": "AdamW", "lr": 1e-3},
                ),
                "reference": RLModelSpec(
                    class_path="tests.e2e.test_rl_dpo.TinyLM",
                ),
            },
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=3),
        metadata=MetadataSpec(author="test", description="dpo test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"

    # 모델 생성 (Factory 우회, 직접 생성)
    models = {"policy": TinyLM(), "reference": TinyLM()}

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(make_pref_batches(5, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    result = trainer.train()

    assert result["total_steps"] == 3
    assert result["algorithm"] == "DPOLoss"
    assert "loss" in result["metrics"]
    assert result["metrics"]["loss"] > 0


def test_rl_recipe_validation() -> None:
    """RL Recipe에서 rl.models.policy가 없으면 에러."""
    with pytest.raises(ValueError, match="rl.models.policy가 필수"):
        Recipe(
            name="bad-rl",
            task="text_generation",
            rl=RLSpec(
                algorithm={"_component_": "DPO"},
                models={
                    "reference": RLModelSpec(class_path="x"),
                },
            ),
            data=DataSpec(
                dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
                collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
            ),
            training=TrainingSpec(max_steps=1),
            metadata=MetadataSpec(author="test", description="test"),
        )

    # RLSpec requires models, so Pydantic will reject it
    with pytest.raises(ValidationError):
        RLSpec(algorithm={"_component_": "DPO"})


def test_dpo_validation_produces_preference_accuracy() -> None:
    """DPO 학습 중 validation이 preference accuracy를 반환한다."""
    from tests.e2e.datasets import ListDataLoader
    from mdp.training.rl_trainer import RLTrainer

    class TinyLM(nn.Module):
        def __init__(self, vocab=32, hidden=16):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, input_ids, attention_mask=None):
            h = self.embed(input_ids)
            logits = self.head(h)
            return type("Out", (), {"logits": logits})()

    def make_pref_batches(n, batch_size, seq_len=8, vocab=32):
        batches = []
        for _ in range(n):
            batches.append({
                "chosen_input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
                "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "chosen_labels": torch.randint(0, vocab, (batch_size, seq_len)),
                "rejected_input_ids": torch.randint(0, vocab, (batch_size, seq_len)),
                "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "rejected_labels": torch.randint(0, vocab, (batch_size, seq_len)),
            })
        return batches

    recipe = Recipe(
        name="dpo-val-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": RLModelSpec(
                    class_path="tests.e2e.test_rl_dpo.TinyLM",
                    optimizer={"_component_": "AdamW", "lr": 1e-3},
                ),
                "reference": RLModelSpec(
                    class_path="tests.e2e.test_rl_dpo.TinyLM",
                ),
            },
        ),
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.PreferenceCollator", "tokenizer": "gpt2", "max_length": 2048},
        ),
        training=TrainingSpec(max_steps=4, val_check_interval=2, val_check_unit="step"),
        metadata=MetadataSpec(author="test", description="dpo val test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"

    models = {"policy": TinyLM(), "reference": TinyLM()}

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(make_pref_batches(6, 4)),
        val_loader=ListDataLoader(make_pref_batches(3, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    result = trainer.train()

    assert result["total_steps"] == 4
    # Validation should have run at step 2 and 4, populating last_metrics
    assert "val_preference_accuracy" in trainer.last_metrics, (
        f"Expected val_preference_accuracy in last_metrics, got: {trainer.last_metrics}"
    )
    acc = trainer.last_metrics["val_preference_accuracy"]
    assert 0.0 <= acc <= 1.0, f"Preference accuracy should be in [0,1], got {acc}"
