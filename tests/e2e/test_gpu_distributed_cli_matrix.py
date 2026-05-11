"""Distributed shell-CLI acceptance matrix over prepared cloud fixtures."""

from __future__ import annotations

from typing import Any, ClassVar
from pathlib import Path

import pytest
import torch
import yaml

from tests.e2e.conftest import e2e_artifact_dir, make_checkpoint_callbacks_yaml
from tests.e2e.gpu_cli_helpers import run_mdp_expect_failure, run_mdp_json
from tests.e2e.test_gpu_distributed_rl import _make_dpo_ddp_yaml
from tests.e2e.test_gpu_distributed_sft import _make_yaml as _make_distributed_sft_yaml


class HiddenOnlyLoss:
    """Test-only algorithm for the FSDP hidden-state fail-fast boundary."""

    needs_logits: ClassVar[bool] = False
    needs_hidden_states: ClassVar[bool] = True
    needs_generation: ClassVar[bool] = False

    def compute_loss(
        self,
        trainable_out: dict[str, Any],
        frozen_out: dict[str, Any],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        hidden = trainable_out["policy"]["hidden_states"]
        return {"policy": hidden.sum() * 0.0}


def _write_hidden_fsdp_yaml(tmp_path: Path, model_path: Path, data_path: Path, gpus: int) -> tuple[str, str]:
    recipe = {
        "name": "e2e-gpu-cli-fsdp-hidden-fail-fast",
        "task": "text_generation",
        "rl": {
            "algorithm": {
                "_component_": "tests.e2e.test_gpu_distributed_cli_matrix.HiddenOnlyLoss",
            },
            "models": {
                "policy": {
                    "_component_": "AutoModelForCausalLM",
                    "pretrained": str(model_path),
                    "optimizer": {"_component_": "AdamW", "lr": 1e-5},
                },
            },
        },
        "data": {
            "dataset": {
                "_component_": "HuggingFaceDataset",
                "source": str(data_path),
                "split": "train",
                "fields": {"text": "text"},
                "tokenizer": str(model_path),
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
            },
            "collator": {
                "_component_": "CausalLMCollator",
                "tokenizer": str(model_path),
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
        },
        "metadata": {"author": "test", "description": "FSDP hidden-state fail-fast acceptance"},
    }
    config = {
        "environment": {"name": "test"},
        "compute": {
            "target": "local",
            "gpus": gpus,
            "distributed": {"strategy": "fsdp"},
        },
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "gpu-cli-fsdp-hidden-fail-fast",
        },
        "storage": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "output_dir": str(tmp_path / "out"),
        },
        "job": {"resume": "disabled", "max_retries": 0},
    }
    recipe_path = tmp_path / "hidden_fsdp_recipe.yaml"
    config_path = tmp_path / "hidden_fsdp_config.yaml"
    yaml.safe_dump(recipe, recipe_path.open("w"))
    yaml.safe_dump(config, config_path.open("w"))
    return str(recipe_path), str(config_path)


@pytest.mark.distributed
@pytest.mark.fixtures
def test_cli_ddp_sft_resume_2gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    smollm2: Path,
    wikitext_tiny: Path,
) -> None:
    """DDP shell CLI can checkpoint and resume across two GPU ranks."""
    n_gpus = min(2, torch.cuda.device_count())
    assert n_gpus >= 2, "test should be skipped if <2 GPUs"

    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _make_distributed_sft_yaml(
        run_dir,
        smollm2,
        wikitext_tiny,
        gpus=n_gpus,
        strategy="ddp",
    )
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir)

    first = run_mdp_json(
        [
            "train", "-r", recipe_path, "-c", config_path,
            "--callbacks", callbacks_path,
            "--override", "training.max_steps=1",
        ],
        cwd=run_dir,
    )
    assert first["command"] == "train"
    assert first["status"] == "success"
    assert first["total_steps"] == 1
    assert (run_dir / "ckpt" / "latest").exists()

    resumed = run_mdp_json(
        [
            "train", "-r", recipe_path, "-c", config_path,
            "--callbacks", callbacks_path,
            "--override", "training.max_steps=2",
        ],
        cwd=run_dir,
    )
    assert resumed["command"] == "train"
    assert resumed["status"] == "success"
    assert resumed["total_steps"] == 2
    assert (run_dir / "ckpt" / "latest").exists()


@pytest.mark.distributed
@pytest.mark.fixtures
def test_cli_fsdp_sft_checkpoint_2gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    smollm2: Path,
    wikitext_tiny: Path,
) -> None:
    """FSDP shell CLI emits a full-state rank-0 checkpoint."""
    n_gpus = min(2, torch.cuda.device_count())
    assert n_gpus >= 2, "test should be skipped if <2 GPUs"

    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _make_distributed_sft_yaml(
        run_dir,
        smollm2,
        wikitext_tiny,
        gpus=n_gpus,
        strategy="fsdp",
    )
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir)

    payload = run_mdp_json(
        ["train", "-r", recipe_path, "-c", config_path, "--callbacks", callbacks_path],
        cwd=run_dir,
    )

    assert payload["command"] == "train"
    assert payload["status"] == "success"
    assert (run_dir / "ckpt" / "latest").exists()
    assert list((run_dir / "ckpt").rglob("model.safetensors"))


@pytest.mark.distributed
@pytest.mark.fixtures
def test_cli_ddp_dpo_2gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    gpt2: Path,
    preference_tiny: Path,
) -> None:
    """DPO preference shell CLI runs under DDP and emits one rank-0 checkpoint."""
    n_gpus = min(2, torch.cuda.device_count())
    assert n_gpus >= 2, "test should be skipped if <2 GPUs"

    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _make_dpo_ddp_yaml(run_dir, gpt2, preference_tiny, n_gpus)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir, save_top_k=1)

    payload = run_mdp_json(
        ["rl-train", "-r", recipe_path, "-c", config_path, "--callbacks", callbacks_path],
        cwd=run_dir,
    )

    assert payload["command"] == "rl-train"
    assert payload["status"] == "success"
    assert payload["algorithm"] == "DPOLoss"
    assert (run_dir / "ckpt" / "latest").exists()


@pytest.mark.distributed
@pytest.mark.fixtures
def test_cli_fsdp_needs_hidden_states_fails_fast_2gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    smollm2: Path,
    prompt_tiny: Path,
) -> None:
    """Unsafe FSDP hidden extraction fails through the actual shell CLI surface."""
    n_gpus = min(2, torch.cuda.device_count())
    assert n_gpus >= 2, "test should be skipped if <2 GPUs"

    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _write_hidden_fsdp_yaml(run_dir, smollm2, prompt_tiny, n_gpus)

    result = run_mdp_expect_failure(
        ["rl-train", "-r", recipe_path, "-c", config_path],
        cwd=run_dir,
    )

    combined = result.stdout + result.stderr
    assert "needs_hidden_states=True" in combined
    assert "FSDP" in combined

