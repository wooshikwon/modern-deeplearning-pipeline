"""GPU SFT (Supervised Fine-Tuning) e2e — single-GPU causal LM training step.

Validates: AutoModelForCausalLM + HF tokenizer + CausalLMCollator + AdamW
on a single CUDA device, 2 training steps, checkpoint saved.

Skips automatically when no CUDA GPU is present or fixtures are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import make_checkpoint_callbacks_yaml, e2e_artifact_dir


def _make_yaml(tmp_path: Path, model_path: Path, data_path: Path, gpus: int = 1) -> tuple[str, str]:
    """Write a minimal SFT recipe + config pair into tmp_path and return their paths."""
    recipe = {
        "name": "e2e-gpu-sft",
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
                "padding": "max_length",
                "truncation": True,
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
        "optimizer": {
            "_component_": "AdamW",
            "lr": 1e-4,
        },
        "metadata": {"author": "test", "description": "GPU SFT 1-step e2e"},
    }
    config: dict = {
        "environment": {"name": "test"},
        "compute": {"target": "local", "gpus": gpus},
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "gpu-sft-e2e",
        },
        "storage": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "output_dir": str(tmp_path / "out"),
        },
        "job": {"resume": "auto", "max_retries": 0},
    }
    if gpus >= 2:
        config["compute"]["distributed"] = {"strategy": "ddp"}

    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    yaml.safe_dump(recipe, recipe_path.open("w"))
    yaml.safe_dump(config, config_path.open("w"))
    return str(recipe_path), str(config_path)


@pytest.mark.gpu
@pytest.mark.fixtures
def test_sft_causal_lm_single_gpu(tmp_path, request, smollm2, wikitext_tiny):
    """End-to-end: HF causal LM SFT, 2 steps on single GPU, checkpoint saved."""
    from mdp.cli.train import run_train

    run_dir = e2e_artifact_dir(tmp_path, request.node.name)
    recipe_path, config_path = _make_yaml(run_dir, smollm2, wikitext_tiny, gpus=1)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir)
    run_train(recipe_path, config_path, callbacks_file=callbacks_path)

    ckpt_dir = run_dir / "ckpt"
    saved = list(ckpt_dir.rglob("*.safetensors")) + list(ckpt_dir.rglob("*.bin"))
    assert saved, f"no checkpoint produced under {ckpt_dir}"


@pytest.mark.gpu
@pytest.mark.fixtures
def test_sft_gpt2_single_gpu(tmp_path, request, gpt2, wikitext_tiny):
    """Same SFT path with a GPT2 architecture — exercises factory family routing."""
    from mdp.cli.train import run_train

    run_dir = e2e_artifact_dir(tmp_path, request.node.name)
    recipe_path, config_path = _make_yaml(run_dir, gpt2, wikitext_tiny, gpus=1)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir)
    run_train(recipe_path, config_path, callbacks_file=callbacks_path)

    ckpt_dir = run_dir / "ckpt"
    saved = list(ckpt_dir.rglob("*.safetensors")) + list(ckpt_dir.rglob("*.bin"))
    assert saved, f"no checkpoint produced under {ckpt_dir}"


@pytest.mark.gpu
@pytest.mark.fixtures
def test_sft_resume_export_inference_single_gpu(tmp_path, request, smollm2, wikitext_tiny, prompt_tiny):
    """Train step 1, resume to step 2, export latest checkpoint, then infer."""
    from mdp.cli.export import run_export
    from mdp.cli.inference import run_inference
    from mdp.cli.train import run_train

    run_dir = e2e_artifact_dir(tmp_path, request.node.name)
    recipe_path, config_path = _make_yaml(run_dir, smollm2, wikitext_tiny, gpus=1)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir)

    run_train(
        recipe_path,
        config_path,
        overrides=["training.max_steps=1"],
        callbacks_file=callbacks_path,
    )
    latest = run_dir / "ckpt" / "latest"
    assert latest.exists(), f"latest checkpoint link missing: {latest}"

    run_train(
        recipe_path,
        config_path,
        overrides=["training.max_steps=2"],
        callbacks_file=callbacks_path,
    )

    exported = run_dir / "exported"
    run_export(checkpoint=str(latest), output=str(exported))
    assert (exported / "recipe.yaml").exists()
    assert any(exported.glob("*.safetensors")) or any(exported.glob("model-*"))

    output_dir = run_dir / "preds"
    run_inference(
        run_id=None,
        model_dir=str(exported),
        data_source=str(prompt_tiny),
        cli_fields=["text=text"],
        output_format="jsonl",
        output_dir=str(output_dir),
        batch_size=2,
        max_length=32,
    )
    assert list(output_dir.glob("*.jsonl")), f"no inference output under {output_dir}"
