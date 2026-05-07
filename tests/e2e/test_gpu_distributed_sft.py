"""GPU distributed SFT e2e — multi-rank DDP causal LM training step over NCCL.

Validates: AutoModelForCausalLM + DDPStrategy + NCCL all_reduce gradient sync
across 2 GPU ranks, 2 training steps, checkpoint saved.

Skipped automatically when fewer than 2 CUDA GPUs are available.

mdp.cli.train.run_train internally spawns torchrun (mdp/cli/train.py
_run_distributed) when settings.compute.gpus >= 2, so no extra subprocess
plumbing is needed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _make_yaml(tmp_path: Path, model_path: Path, data_path: Path, gpus: int) -> tuple[str, str]:
    recipe = {
        "name": "e2e-gpu-distributed-sft",
        "task": "text_generation",
        "model": {
            "_component_": "AutoModelForCausalLM",
            "pretrained": str(model_path),
        },
        "data": {
            "dataset": {
                "_component_": "HuggingFaceDataset",
                "source": str(data_path),
                "split": "train",
                "fields": {"text": "text"},
                "tokenizer": str(model_path),
                "max_length": 32,
            },
            "collator": {
                "_component_": "CausalLMCollator",
                "tokenizer": str(model_path),
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
                "drop_last": True,
            },
        },
        "training": {
            "epochs": 1,
            "max_steps": 2,
            "precision": "fp32",
            "gradient_clip_max_norm": 1.0,
        },
        "optimizer": {"_component_": "AdamW", "lr": 1e-4},
        "metadata": {"author": "test", "description": "GPU distributed SFT e2e"},
    }
    config = {
        "environment": {"name": "test"},
        "compute": {
            "target": "local",
            "gpus": gpus,
            "distributed": {"strategy": "ddp"},
        },
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "gpu-ddp-sft-e2e",
        },
        "storage": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "output_dir": str(tmp_path / "out"),
        },
        "job": {"resume": "auto", "max_retries": 0},
    }
    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    yaml.safe_dump(recipe, recipe_path.open("w"))
    yaml.safe_dump(config, config_path.open("w"))
    return str(recipe_path), str(config_path)


@pytest.mark.distributed
@pytest.mark.fixtures
def test_ddp_causal_lm_2gpu(tmp_path, smollm2, wikitext_tiny):
    """End-to-end: HF causal LM SFT under DDP (2 ranks, NCCL), 2 steps."""
    import torch

    n_gpus = min(2, torch.cuda.device_count())
    assert n_gpus >= 2, "test should be skipped if <2 GPUs"

    from mdp.cli.train import run_train

    recipe_path, config_path = _make_yaml(tmp_path, smollm2, wikitext_tiny, gpus=n_gpus)
    run_train(recipe_path, config_path)

    ckpt_dir = tmp_path / "ckpt"
    saved = list(ckpt_dir.rglob("*.safetensors")) + list(ckpt_dir.rglob("*.bin"))
    assert saved, f"no checkpoint produced under {ckpt_dir}"
