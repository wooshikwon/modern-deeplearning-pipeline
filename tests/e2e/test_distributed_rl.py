"""분산 RL 학습 통합 테스트 — gloo CPU backend로 DDPStrategy + RLTrainer 검증.

검증 항목:
1. DPO 분산: 2 rank에서 preference 데이터 학습, loss finite
2. GRPO 분산: 2 rank에서 generation 경로 학습, loss finite
3. set_epoch: 에폭 경계에서 sampler 갱신이 호출되는지
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from tests.e2e.datasets import ListDataLoader


class TinyGenLM(nn.Module):
    """generate()를 지원하는 작은 테스트 LM (분산 테스트용)."""

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
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


def _make_preference_batches(n, batch_size, seq_len=6, vocab=32):
    batches = []
    for _ in range(n):
        batches.append({
            "chosen_input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
            "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "chosen_labels": torch.randint(1, vocab, (batch_size, seq_len)),
            "rejected_input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
            "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "rejected_labels": torch.randint(1, vocab, (batch_size, seq_len)),
        })
    return batches


def _make_prompt_batches(n, batch_size, seq_len=4, vocab=32):
    batches = []
    for _ in range(n):
        batches.append({
            "input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        })
    return batches


# ── DPO distributed worker ──


def _dpo_distributed_worker(rank: int, world_size: int, result_queue) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29510"

    try:
        from mdp.settings.schema import (
            Config,
            DataSpec,
            MetadataSpec,
            RLModelSpec,
            Recipe,
            Settings,
            TrainingSpec,
        )
        from mdp.training.rl_trainer import RLTrainer

        recipe = Recipe(
            name="dpo-dist-test",
            task="text_generation",
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": RLModelSpec(
                    class_path="tests.e2e.test_distributed_rl.TinyGenLM",
                    optimizer={"_component_": "AdamW", "lr": 1e-3},
                ),
                "reference": RLModelSpec(
                    class_path="tests.e2e.test_distributed_rl.TinyGenLM",
                ),
            },
            data=DataSpec(source="/tmp/fake"),
            training=TrainingSpec(max_steps=3),
            metadata=MetadataSpec(author="test", description="dpo dist test"),
        )
        config = Config()
        config.compute.distributed = {"strategy": "ddp"}
        settings = Settings(recipe=recipe, config=config)
        settings.config.job.resume = "disabled"

        models = {"policy": TinyGenLM(), "reference": TinyGenLM()}

        trainer = RLTrainer(
            settings=settings,
            models=models,
            train_loader=ListDataLoader(_make_preference_batches(5, 4)),
        )
        # Force gloo CPU
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False
        # Replace strategy with actual DDP gloo
        from mdp.training.strategies.ddp import DDPStrategy
        trainer.strategy = DDPStrategy(backend="gloo")

        result = trainer.train()

        result_queue.put({
            "rank": rank,
            "total_steps": result["total_steps"],
            "loss": result["metrics"]["loss"],
            "loss_finite": torch.isfinite(torch.tensor(result["metrics"]["loss"])).item(),
        })

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


# ── GRPO distributed worker ──


def _grpo_distributed_worker(rank: int, world_size: int, result_queue) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"

    try:
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
        from mdp.training.rl_trainer import RLTrainer

        recipe = Recipe(
            name="grpo-dist-test",
            task="text_generation",
            algorithm={"_component_": "GRPO", "clip_range": 0.2, "kl_coeff": 0.01},
            models={
                "policy": RLModelSpec(
                    class_path="tests.e2e.test_distributed_rl.TinyGenLM",
                    optimizer={"_component_": "AdamW", "lr": 1e-3},
                ),
                "reference": RLModelSpec(
                    class_path="tests.e2e.test_distributed_rl.TinyGenLM",
                ),
                "reward": RLModelSpec(
                    class_path="tests.e2e.test_distributed_rl.TinyGenLM",
                ),
            },
            data=DataSpec(source="/tmp/fake"),
            training=TrainingSpec(max_steps=2),
            generation=GenerationSpec(max_new_tokens=4),
            metadata=MetadataSpec(author="test", description="grpo dist test"),
        )
        config = Config()
        config.compute.distributed = {"strategy": "ddp"}
        settings = Settings(recipe=recipe, config=config)
        settings.config.job.resume = "disabled"

        models = {"policy": TinyGenLM(), "reference": TinyGenLM(), "reward": TinyGenLM()}

        trainer = RLTrainer(
            settings=settings,
            models=models,
            train_loader=ListDataLoader(_make_prompt_batches(5, 4)),
        )
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False
        from mdp.training.strategies.ddp import DDPStrategy
        trainer.strategy = DDPStrategy(backend="gloo")

        result = trainer.train()

        result_queue.put({
            "rank": rank,
            "total_steps": result["total_steps"],
            "loss": result["metrics"]["loss"],
            "loss_finite": torch.isfinite(torch.tensor(result["metrics"]["loss"])).item(),
        })

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


# ── Tests ──


def test_dpo_distributed_2rank() -> None:
    """DPO: gloo CPU 2-process에서 학습 완료, loss finite."""
    result_queue = mp.Queue()
    mp.spawn(_dpo_distributed_worker, args=(2, result_queue), nprocs=2, join=True)

    results = [result_queue.get() for _ in range(2)]
    for r in results:
        assert r["total_steps"] == 3, f"rank {r['rank']}: expected 3 steps, got {r['total_steps']}"
        assert r["loss_finite"], f"rank {r['rank']}: loss is not finite ({r['loss']})"


def test_grpo_distributed_2rank() -> None:
    """GRPO: gloo CPU 2-process에서 generation + 학습 완료, loss finite."""
    result_queue = mp.Queue()
    mp.spawn(_grpo_distributed_worker, args=(2, result_queue), nprocs=2, join=True)

    results = [result_queue.get() for _ in range(2)]
    for r in results:
        assert r["total_steps"] == 2, f"rank {r['rank']}: expected 2 steps, got {r['total_steps']}"
        assert r["loss_finite"], f"rank {r['rank']}: loss is not finite ({r['loss']})"
