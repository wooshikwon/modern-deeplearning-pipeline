"""커스텀 RL 알고리즘 검증 — weighted-NTP 패턴이 mdp rl-train으로 동작하는지.

사용자가 MDP 코드를 수정하지 않고, _component_ 패턴으로 커스텀 알고리즘을 주입하여
frozen critic + policy weighted CE 학습이 가능한지 검증한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    RLModelSpec,
    RLSpec,
    Recipe,
    Settings,
    TrainingSpec,
)
from mdp.training.losses.rl import compute_log_probs, masked_mean


# ── 사용자 커스텀 코드 (MDP 밖) ──


class SimpleWeightedCELoss:
    """weighted-NTP를 단순화한 커스텀 알고리즘.

    frozen critic의 logits를 "value"로 해석하고,
    간단한 advantage → weight 변환 후 weighted CE를 계산한다.
    GAE/AWR의 완전 구현이 아닌, 구조적 호환성 검증용.
    """

    def __init__(self, weight_scale: float = 1.0):
        self.weight_scale = weight_scale

    def compute_loss(self, trainable_out, frozen_out, batch):
        # policy logits
        policy_logits = trainable_out["policy"]["logits"]
        # critic "value" — logits의 첫 번째 차원을 value로 사용 (단순화)
        critic_logits = frozen_out["critic"]["logits"]
        critic_values = critic_logits[:, :, 0]  # [batch, seq]

        # 간단한 advantage: value의 시간차
        advantages = critic_values[:, 1:] - critic_values[:, :-1]
        # weight = softmax(advantage * scale)
        weights = F.softmax(advantages * self.weight_scale, dim=-1)

        # weighted CE
        shift_logits = policy_logits[:, :-1, :]
        shift_labels = batch["labels"][:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)

        mask = shift_labels != -100
        loss = (ce * weights * mask).sum() / mask.sum().clamp(min=1)

        return {"policy": loss}


class TinyLM(nn.Module):
    """테스트용 작은 LM."""

    def __init__(self, vocab=32, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return type("Out", (), {"logits": logits})()


# ── 테스트 ──


def test_custom_weighted_ntp_via_rl_train() -> None:
    """커스텀 WeightedCE 알고리즘이 mdp rl-train 구조로 동작하는지."""
    from tests.e2e.datasets import ListDataLoader
    from mdp.training.rl_trainer import RLTrainer

    # causal 배치 (preference가 아님)
    def make_causal_batches(n, batch_size, seq_len=8, vocab=32):
        batches = []
        for _ in range(n):
            input_ids = torch.randint(0, vocab, (batch_size, seq_len))
            labels = input_ids.clone()
            labels[:, :2] = -100  # 처음 2 토큰은 prompt masking
            batches.append({
                "input_ids": input_ids,
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                "labels": labels,
            })
        return batches

    recipe = Recipe(
        name="weighted-ntp-test",
        task="text_generation",
        rl=RLSpec(
            algorithm={
                "_component_": "tests.e2e.test_rl_custom_algorithm.SimpleWeightedCELoss",
                "weight_scale": 2.0,
            },
            models={
                "policy": RLModelSpec(
                    class_path="tests.e2e.test_rl_custom_algorithm.TinyLM",
                    optimizer={"_component_": "AdamW", "lr": 1e-3},
                ),
                "critic": RLModelSpec(
                    class_path="tests.e2e.test_rl_custom_algorithm.TinyLM",
                    # optimizer 없음 → frozen
                ),
            },
        ),
        data=DataSpec(source="/tmp/fake", label_strategy="causal"),
        training=TrainingSpec(max_steps=5),
        metadata=MetadataSpec(author="test", description="custom weighted-ntp test"),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"

    models = {"policy": TinyLM(), "critic": TinyLM()}

    trainer = RLTrainer(
        settings=settings,
        models=models,
        train_loader=ListDataLoader(make_causal_batches(10, 4)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    result = trainer.train()

    assert result["total_steps"] == 5
    assert result["algorithm"] == "SimpleWeightedCELoss"
    assert result["metrics"]["loss"] > 0
    assert torch.isfinite(torch.tensor(result["metrics"]["loss"]))
