"""GRPO/PPO generation 루프 테스트.

TinyLM에 간단한 generate()를 구현하여 generation 경로가 동작하는지 검증.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mdp.settings.schema import (
    Config,
    DataSpec,
    GenerationSpec,
    MetadataSpec,
    RLModelSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from tests.e2e.datasets import ListDataLoader


class TinyGenLM(nn.Module):
    """generate()를 지원하는 작은 테스트 LM."""

    def __init__(self, vocab=32, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.config = type("Config", (), {"pad_token_id": 0})()
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return type("Out", (), {"logits": logits})()

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4, **kwargs):
        """간단한 greedy 생성."""
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


def _make_prompt_batches(n, batch_size, seq_len=4, vocab=32):
    """prompt만 있는 배치 (응답은 policy가 생성)."""
    batches = []
    for _ in range(n):
        batches.append({
            "input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        })
    return batches


def test_grpo_generation_loop() -> None:
    """GRPOLoss가 generation 경로로 3 step 학습을 완료하는지."""
    from mdp.training.rl_trainer import RLTrainer

    recipe = Recipe(
        name="grpo-test",
        task="text_generation",
        algorithm={"_component_": "GRPO", "clip_range": 0.2, "kl_coeff": 0.01},
        models={
            "policy": RLModelSpec(
                class_path="tests.e2e.test_rl_generation.TinyGenLM",
                optimizer={"_component_": "AdamW", "lr": 1e-3},
            ),
            "reference": RLModelSpec(
                class_path="tests.e2e.test_rl_generation.TinyGenLM",
            ),
        },
        data=DataSpec(source="/tmp/fake"),
        training=TrainingSpec(max_steps=3),
        generation=GenerationSpec(max_new_tokens=4),
        metadata=MetadataSpec(author="test", description="grpo test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"

    models = {"policy": TinyGenLM(), "reference": TinyGenLM()}

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(_make_prompt_batches(5, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    result = trainer.train()

    assert result["total_steps"] == 3
    assert result["algorithm"] == "GRPOLoss"
    assert result["metrics"]["loss"] != 0


def test_ppo_generation_with_value_model() -> None:
    """PPOLoss가 policy + value model로 학습을 완료하는지."""
    from mdp.training.rl_trainer import RLTrainer

    recipe = Recipe(
        name="ppo-test",
        task="text_generation",
        algorithm={"_component_": "PPO", "clip_range": 0.2, "value_coeff": 0.5, "ppo_epochs": 2},
        models={
            "policy": RLModelSpec(
                class_path="tests.e2e.test_rl_generation.TinyGenLM",
                optimizer={"_component_": "AdamW", "lr": 1e-3},
            ),
            "value": RLModelSpec(
                class_path="tests.e2e.test_rl_generation.TinyGenLM",
                optimizer={"_component_": "AdamW", "lr": 1e-3},
                freeze=False,
            ),
            "reference": RLModelSpec(
                class_path="tests.e2e.test_rl_generation.TinyGenLM",
            ),
        },
        data=DataSpec(source="/tmp/fake"),
        training=TrainingSpec(max_steps=2),
        generation=GenerationSpec(max_new_tokens=4),
        metadata=MetadataSpec(author="test", description="ppo test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"

    models = {"policy": TinyGenLM(), "value": TinyGenLM(), "reference": TinyGenLM()}

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(_make_prompt_batches(5, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    result = trainer.train()

    assert result["total_steps"] == 2
    assert result["algorithm"] == "PPOLoss"
