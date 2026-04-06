"""분산 RL 학습 통합 테스트 — gloo CPU backend로 DDPStrategy + RLTrainer 검증.

subprocess + torchrun 방식으로 실행하여 pytest의 fork/spawn 컨텍스트 충돌을 회피한다.

검증 항목:
1. DPO 분산: 2 rank에서 preference 데이터 학습, loss finite
2. GRPO 분산: 2 rank에서 generation 경로 학습, loss finite
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_distributed_worker(worker_name: str, nproc: int = 2) -> list[dict]:
    """torchrun으로 워커를 실행하고 결과를 반환한다."""
    with tempfile.TemporaryDirectory() as result_dir:
        port = _find_free_port()
        project_root = str(Path(__file__).resolve().parents[2])
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable, "-m", "torch.distributed.run",
                "--standalone",
                "--nproc_per_node", str(nproc),
                "--master_port", str(port),
                __file__,
                "--worker", worker_name,
                "--result-dir", result_dir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_root,
            env=env,
        )
        assert result.returncode == 0, (
            f"{worker_name} worker failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        results = []
        for i in range(nproc):
            result_path = Path(result_dir) / f"result_{i}.json"
            assert result_path.exists(), f"rank-{i} result file not found"
            results.append(json.loads(result_path.read_text()))
        return results


def test_dpo_distributed_2rank() -> None:
    """DPO: gloo CPU 2-process에서 학습 완료, loss finite."""
    results = _run_distributed_worker("dpo")
    for r in results:
        assert r["total_steps"] == 3, f"rank {r['rank']}: expected 3 steps, got {r['total_steps']}"
        assert r["loss_finite"], f"rank {r['rank']}: loss is not finite ({r['loss']})"


def test_grpo_distributed_2rank() -> None:
    """GRPO: gloo CPU 2-process에서 generation + 학습 완료, loss finite."""
    results = _run_distributed_worker("grpo")
    for r in results:
        assert r["total_steps"] == 2, f"rank {r['rank']}: expected 2 steps, got {r['total_steps']}"
        assert r["loss_finite"], f"rank {r['rank']}: loss is not finite ({r['loss']})"


# ── Worker implementations (torchrun이 실행) ──


def _write_result(result_dir: str, rank: int, data: dict) -> None:
    Path(result_dir).joinpath(f"result_{rank}.json").write_text(json.dumps(data))


def _dpo_worker_main(result_dir: str) -> None:
    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn

    rank = int(os.environ["RANK"])

    try:
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
        from mdp.training.rl_trainer import RLTrainer

        from tests.e2e.datasets import ListDataLoader

        class TinyGenLM(nn.Module):
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

            def generate(self, input_ids, attention_mask=None, max_new_tokens=4, **kw):
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

        recipe = Recipe(
            name="dpo-dist-test",
            task="text_generation",
            rl=RLSpec(
                algorithm={"_component_": "DPO", "beta": 0.1},
                models={
                    "policy": RLModelSpec(
                        class_path="__main__.TinyGenLM",
                        optimizer={"_component_": "AdamW", "lr": 1e-3},
                    ),
                    "reference": RLModelSpec(
                        class_path="__main__.TinyGenLM",
                    ),
                },
            ),
            data=DataSpec(source="/tmp/fake", label_strategy="preference"),
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
        trainer.device = torch.device("cpu")
        trainer.amp_enabled = False
        from mdp.training.strategies.ddp import DDPStrategy
        trainer.strategy = DDPStrategy(backend="gloo")

        result = trainer.train()

        _write_result(result_dir, rank, {
            "rank": rank,
            "total_steps": result["total_steps"],
            "loss": result["metrics"]["loss"],
            "loss_finite": torch.isfinite(torch.tensor(result["metrics"]["loss"])).item(),
        })

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _grpo_worker_main(result_dir: str) -> None:
    import os

    import torch
    import torch.distributed as dist
    import torch.nn as nn

    rank = int(os.environ["RANK"])

    try:
        from mdp.settings.schema import (
            Config,
            DataSpec,
            MetadataSpec,
            RLGenerationSpec,
            RLModelSpec,
            RLSpec,
            Recipe,
            Settings,
            TrainingSpec,
        )
        from mdp.training.rl_trainer import RLTrainer

        from tests.e2e.datasets import ListDataLoader

        class TinyGenLM(nn.Module):
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

            def generate(self, input_ids, attention_mask=None, max_new_tokens=4, **kw):
                ids = input_ids
                for _ in range(max_new_tokens):
                    logits = self.forward(ids).logits
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    ids = torch.cat([ids, next_token], dim=1)
                return ids

        def _make_prompt_batches(n, batch_size, seq_len=4, vocab=32):
            batches = []
            for _ in range(n):
                batches.append({
                    "input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
                    "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                })
            return batches

        recipe = Recipe(
            name="grpo-dist-test",
            task="text_generation",
            rl=RLSpec(
                algorithm={"_component_": "GRPO", "clip_range": 0.2, "kl_coeff": 0.01},
                models={
                    "policy": RLModelSpec(
                        class_path="__main__.TinyGenLM",
                        optimizer={"_component_": "AdamW", "lr": 1e-3},
                    ),
                    "reference": RLModelSpec(
                        class_path="__main__.TinyGenLM",
                    ),
                    "reward": RLModelSpec(
                        class_path="__main__.TinyGenLM",
                    ),
                },
                generation=RLGenerationSpec(max_new_tokens=4),
            ),
            data=DataSpec(source="/tmp/fake", label_strategy="preference"),
            training=TrainingSpec(max_steps=2),
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

        _write_result(result_dir, rank, {
            "rank": rank,
            "total_steps": result["total_steps"],
            "loss": result["metrics"]["loss"],
            "loss_finite": torch.isfinite(torch.tensor(result["metrics"]["loss"])).item(),
        })

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=["dpo", "grpo"])
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    if args.worker == "dpo":
        _dpo_worker_main(args.result_dir)
    elif args.worker == "grpo":
        _grpo_worker_main(args.result_dir)
