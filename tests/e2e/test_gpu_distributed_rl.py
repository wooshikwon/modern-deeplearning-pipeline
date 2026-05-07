"""GPU distributed RL fixture e2e tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import make_checkpoint_callbacks_yaml, e2e_artifact_dir


def _make_dpo_ddp_yaml(
    tmp_path: Path,
    model_path: Path,
    data_path: Path,
    gpus: int,
) -> tuple[str, str]:
    recipe = {
        "name": "e2e-gpu-ddp-dpo",
        "task": "text_generation",
        "rl": {
            "algorithm": {"_component_": "DPO", "beta": 0.1},
            "models": {
                "policy": {
                    "_component_": "AutoModelForCausalLM",
                    "pretrained": str(model_path),
                    "optimizer": {"_component_": "AdamW", "lr": 1e-5},
                },
                "reference": {
                    "_component_": "AutoModelForCausalLM",
                    "pretrained": str(model_path),
                },
            },
        },
        "data": {
            "dataset": {
                "_component_": "HuggingFaceDataset",
                "source": str(data_path),
                "split": "train",
            },
            "collator": {
                "_component_": "PreferenceCollator",
                "tokenizer": str(model_path),
                "max_length": 32,
            },
            "dataloader": {
                "batch_size": 1,
                "num_workers": 0,
                "drop_last": True,
            },
        },
        "training": {
            "epochs": 1,
            "max_steps": 1,
            "precision": "fp32",
            "gradient_clip_max_norm": 1.0,
        },
        "metadata": {"author": "test", "description": "GPU DDP DPO fixture e2e"},
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
            "experiment_name": "gpu-ddp-dpo-e2e",
        },
        "storage": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "output_dir": str(tmp_path / "out"),
        },
        "job": {"resume": "disabled", "max_retries": 0},
    }
    recipe_path = tmp_path / "dpo_ddp_recipe.yaml"
    config_path = tmp_path / "dpo_ddp_config.yaml"
    yaml.safe_dump(recipe, recipe_path.open("w"))
    yaml.safe_dump(config, config_path.open("w"))
    return str(recipe_path), str(config_path)


@pytest.mark.distributed
@pytest.mark.fixtures
def test_dpo_ddp_2gpu(tmp_path, request, gpt2, preference_tiny):
    """DPO preference training runs under DDP and emits one rank-0 checkpoint."""
    import torch

    n_gpus = min(2, torch.cuda.device_count())
    assert n_gpus >= 2, "test should be skipped if <2 GPUs"

    from mdp.cli.rl_train import run_rl_train

    run_dir = e2e_artifact_dir(tmp_path, request.node.name)
    recipe_path, config_path = _make_dpo_ddp_yaml(run_dir, gpt2, preference_tiny, n_gpus)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir, save_top_k=1)
    run_rl_train(recipe_path, config_path, callbacks_file=callbacks_path)

    ckpt_dir = run_dir / "ckpt"
    assert (ckpt_dir / "latest").exists()
    checkpoints = [p for p in ckpt_dir.glob("checkpoint-*") if p.is_dir()]
    assert 1 <= len(checkpoints) <= 1
    assert list(ckpt_dir.rglob("model.safetensors")) or list(ckpt_dir.rglob("model.pt"))
