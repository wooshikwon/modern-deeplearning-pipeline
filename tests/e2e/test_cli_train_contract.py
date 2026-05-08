"""CLI train command contract tests through the real module entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data.dataloader import default_collate


class SyntheticVisionDataset:
    """Tiny deterministic dataset importable by subprocess CLI smoke tests."""

    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        label = idx % 2
        return {
            "pixel_values": torch.full((3, 8, 8), float(label)),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SyntheticVisionCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return default_collate(features)


def _run_mdp(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.run(
        [sys.executable, "-m", "mdp", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_train_yaml(tmp_path: Path) -> tuple[Path, Path]:
    recipe = {
        "name": "cli-train-contract",
        "task": "image_classification",
        "model": {
            "_component_": "tests.e2e.models.TinyVisionModel",
            "num_classes": 2,
            "hidden_dim": 16,
        },
        "data": {
            "dataset": {
                "_component_": "tests.e2e.test_cli_train_contract.SyntheticVisionDataset",
                "length": 2,
            },
            "collator": {
                "_component_": "tests.e2e.test_cli_train_contract.SyntheticVisionCollator",
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
                "drop_last": False,
            },
        },
        "training": {
            "max_steps": 1,
            "precision": "fp32",
        },
        "optimizer": {
            "_component_": "AdamW",
            "lr": 0.001,
        },
        "metadata": {
            "author": "contract",
            "description": "CLI train JSON contract smoke",
        },
    }
    config = {
        "environment": {"name": "local"},
        "compute": {"target": "local", "gpus": 0},
        "storage": {
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "output_dir": str(tmp_path / "outputs"),
        },
        "job": {"resume": "disabled"},
    }

    recipe_path = tmp_path / "recipe.yaml"
    config_path = tmp_path / "config.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return recipe_path, config_path


def _assert_train_success_payload(payload: dict[str, Any], tmp_path: Path) -> None:
    assert payload["status"] == "success"
    assert payload["command"] == "train"
    assert payload["total_steps"] == 1
    assert payload["stopped_reason"] == "max_steps_reached"
    assert payload["checkpoint_dir"] == str(tmp_path / "checkpoints")
    assert payload["output_dir"] == str(tmp_path / "outputs")
    assert "checkpoints_saved" in payload


def test_python_m_mdp_top_level_json_train_success(tmp_path: Path) -> None:
    recipe_path, config_path = _write_train_yaml(tmp_path)

    result = _run_mdp(
        ["--format", "json", "train", "-r", str(recipe_path), "-c", str(config_path)],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert '"status"' not in result.stderr
    assert '"command"' not in result.stderr
    _assert_train_success_payload(json.loads(result.stdout), tmp_path)


def test_python_m_mdp_subcommand_json_train_success(tmp_path: Path) -> None:
    recipe_path, config_path = _write_train_yaml(tmp_path)

    result = _run_mdp(
        ["train", "-r", str(recipe_path), "-c", str(config_path), "--format", "json"],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert '"status"' not in result.stderr
    assert '"command"' not in result.stderr
    _assert_train_success_payload(json.loads(result.stdout), tmp_path)
