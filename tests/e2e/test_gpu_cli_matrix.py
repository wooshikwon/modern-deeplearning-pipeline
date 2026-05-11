"""Single-GPU shell-CLI acceptance matrix over prepared cloud fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import e2e_artifact_dir, make_checkpoint_callbacks_yaml
from tests.e2e.gpu_cli_helpers import run_mdp_json
from tests.e2e.test_gpu_rl import _make_dpo_yaml
from tests.e2e.test_gpu_sft import _make_yaml as _make_sft_yaml


def _write_grpo_yaml(tmp_path: Path, model_path: Path, data_path: Path) -> tuple[str, str]:
    recipe = {
        "name": "e2e-gpu-cli-grpo",
        "task": "text_generation",
        "rl": {
            "algorithm": {"_component_": "GRPO", "clip_range": 0.2, "kl_coeff": 0.01},
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
                "reward": {
                    "_component_": "AutoModelForCausalLM",
                    "pretrained": str(model_path),
                },
            },
            "generation": {
                "max_new_tokens": 4,
                "do_sample": False,
                "group_size": 1,
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
            "gradient_clip_max_norm": 1.0,
        },
        "metadata": {"author": "test", "description": "GPU GRPO shell CLI acceptance"},
    }
    config = {
        "environment": {"name": "test"},
        "compute": {"target": "local", "gpus": 1},
        "mlflow": {
            "tracking_uri": str(tmp_path / "mlruns"),
            "experiment_name": "gpu-cli-grpo-e2e",
        },
        "storage": {
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "output_dir": str(tmp_path / "out"),
        },
        "job": {"resume": "disabled", "max_retries": 0},
    }
    recipe_path = tmp_path / "grpo_recipe.yaml"
    config_path = tmp_path / "grpo_config.yaml"
    yaml.safe_dump(recipe, recipe_path.open("w"))
    yaml.safe_dump(config, config_path.open("w"))
    return str(recipe_path), str(config_path)


@pytest.mark.gpu
@pytest.mark.fixtures
def test_cli_sft_resume_export_inference_generate_single_gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    smollm2: Path,
    wikitext_tiny: Path,
    prompt_tiny: Path,
) -> None:
    """The core paid-GPU acceptance chain uses the real shell CLI surface."""
    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _make_sft_yaml(run_dir, smollm2, wikitext_tiny, gpus=1)
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

    latest = run_dir / "ckpt" / "latest"
    assert latest.exists(), f"latest checkpoint link missing: {latest}"

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

    exported = run_dir / "exported"
    export_payload = run_mdp_json(
        ["export", "--checkpoint", str(latest), "--output", str(exported)],
        cwd=run_dir,
    )
    assert export_payload["command"] == "export"
    assert export_payload["status"] == "success"
    assert Path(export_payload["output_dir"]) == exported
    assert (exported / "recipe.yaml").exists()

    preds = run_dir / "preds"
    inference_payload = run_mdp_json(
        [
            "inference",
            "--model-dir", str(exported),
            "--data", str(prompt_tiny),
            "--fields", "text=text",
            "--output-format", "jsonl",
            "--output-dir", str(preds),
        ],
        cwd=run_dir,
    )
    assert inference_payload["command"] == "inference"
    assert inference_payload["status"] == "success"
    assert Path(inference_payload["output_path"]).exists()

    generated = run_dir / "generated.jsonl"
    generate_payload = run_mdp_json(
        [
            "generate",
            "--model-dir", str(exported),
            "--prompts", str(prompt_tiny),
            "--prompt-field", "text",
            "--output", str(generated),
            "--max-new-tokens", "4",
            "--batch-size", "2",
        ],
        cwd=run_dir,
    )
    assert generate_payload["command"] == "generate"
    assert generate_payload["status"] == "success"
    assert generate_payload["num_generated"] > 0
    assert generated.exists()
    assert json.loads(generated.read_text().splitlines()[0])["generated_text"] is not None


@pytest.mark.gpu
@pytest.mark.fixtures
def test_cli_dpo_preference_single_gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    gpt2: Path,
    preference_tiny: Path,
) -> None:
    """DPO preference training is covered through ``mdp rl-train`` JSON CLI."""
    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _make_dpo_yaml(run_dir, gpt2, preference_tiny)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir, save_top_k=1)

    payload = run_mdp_json(
        ["rl-train", "-r", recipe_path, "-c", config_path, "--callbacks", callbacks_path],
        cwd=run_dir,
    )

    assert payload["command"] == "rl-train"
    assert payload["status"] == "success"
    assert payload["algorithm"] == "DPOLoss"
    assert (run_dir / "ckpt" / "latest").exists()


@pytest.mark.gpu
@pytest.mark.fixtures
def test_cli_grpo_generation_single_gpu(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    gpt2: Path,
    prompt_tiny: Path,
) -> None:
    """Generation-based RL path is covered through ``mdp rl-train`` JSON CLI."""
    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    recipe_path, config_path = _write_grpo_yaml(run_dir, gpt2, prompt_tiny)
    callbacks_path = make_checkpoint_callbacks_yaml(run_dir, save_top_k=1)

    payload = run_mdp_json(
        ["rl-train", "-r", recipe_path, "-c", config_path, "--callbacks", callbacks_path],
        cwd=run_dir,
    )

    assert payload["command"] == "rl-train"
    assert payload["status"] == "success"
    assert payload["algorithm"] == "GRPOLoss"
    assert (run_dir / "ckpt" / "latest").exists()


@pytest.mark.gpu
@pytest.mark.fixtures
def test_cli_bert_text_classification_pretrained_inference(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    bert_tiny: Path,
    classification_text_tiny: Path,
) -> None:
    """Encoder text inference/evaluation fixture is covered by the shell CLI."""
    run_dir = e2e_artifact_dir(tmp_path, request.node.name, clean=True)
    out_dir = run_dir / "bert-preds"

    payload = run_mdp_json(
        [
            "inference",
            "--pretrained", str(bert_tiny),
            "--tokenizer", str(bert_tiny),
            "--data", str(classification_text_tiny),
            "--fields", "text=text",
            "--fields", "label=label",
            "--output-format", "jsonl",
            "--output-dir", str(out_dir),
            "--batch-size", "2",
            "--max-length", "32",
        ],
        cwd=run_dir,
    )

    assert payload["command"] == "inference"
    assert payload["status"] == "success"
    assert payload["task"].endswith("SequenceClassification")
    assert Path(payload["output_path"]).exists()
