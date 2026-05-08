"""Local CLI error UX contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mdp.__main__ import app


def _run_mdp(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    return subprocess.run(
        [sys.executable, "-m", "mdp", *args],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_json_error(
    result: subprocess.CompletedProcess[str],
    *,
    command: str,
    error_type: str,
    message_contains: str,
) -> None:
    assert result.returncode == 1
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["command"] == command
    assert payload["error"]["type"] == error_type
    assert message_contains in payload["error"]["message"]
    assert payload["error"]["details"] == {}


def test_train_missing_yaml_error_json_mode() -> None:
    """JSON mode emits a structured error payload on validation failure."""
    result = _run_mdp(
        ["--format", "json", "train", "-r", "missing-recipe.yaml", "-c", "missing-config.yaml"],
    )

    _assert_json_error(
        result,
        command="train",
        error_type="ValidationError",
        message_contains="missing-recipe.yaml",
    )


def test_rl_train_missing_yaml_error_json_mode() -> None:
    result = _run_mdp(
        ["--format", "json", "rl-train", "-r", "missing-rl.yaml", "-c", "missing-config.yaml"],
    )

    _assert_json_error(
        result,
        command="rl-train",
        error_type="ValidationError",
        message_contains="missing-rl.yaml",
    )


def test_train_schema_error_json_mode_includes_yaml_path_details(tmp_path: Path) -> None:
    recipe_path = tmp_path / "bad-recipe.yaml"
    config_path = tmp_path / "config.yaml"
    recipe_path.write_text(
        """
name: bad-schema
task: text_generation
model:
  _component_: transformers.AutoModelForCausalLM
  pretrained: hf://gpt2
data:
  dataset:
    _component_: HuggingFaceDataset
    source: wikitext
  collator:
    _component_: CausalLMCollator
    tokenizer: gpt2
training:
  epochs: 1
  val_check_units: step
optimizer:
  _component_: AdamW
metadata:
  author: test
  description: schema error
""".lstrip()
    )
    config_path.write_text("compute:\n  gpus: 0\n")

    result = _run_mdp(
        ["--format", "json", "train", "-r", str(recipe_path), "-c", str(config_path)],
    )

    assert result.returncode == 1
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["command"] == "train"
    assert payload["error"]["type"] == "ValidationError"
    assert payload["error"]["details"]["schema_errors"] == [
        {
            "path": "$.training.val_check_units",
            "message": "Extra inputs are not permitted",
        }
    ]


@pytest.mark.parametrize(
    ("command", "args", "message_contains"),
    [
        ("estimate", ["estimate", "-r", "missing-recipe.yaml"], "missing-recipe.yaml"),
        ("inference", ["inference", "--data", "data.jsonl"], "--run-id, --model-dir, --pretrained"),
        ("generate", ["generate", "--prompts", "prompts.jsonl"], "--run-id, --model-dir, --pretrained"),
        ("export", ["export"], "--run-id, --model-dir, --pretrained"),
        ("serve", ["serve"], "--run-id, --model-dir, --pretrained"),
    ],
)
def test_missing_source_error_json_mode(
    command: str,
    args: list[str],
    message_contains: str,
) -> None:
    result = _run_mdp(["--format", "json", *args])

    _assert_json_error(
        result,
        command=command,
        error_type="ValidationError",
        message_contains=message_contains,
    )


def test_train_missing_yaml_error_text_mode() -> None:
    """Text mode keeps human-facing progress on stdout and errors on stderr."""
    result = CliRunner(mix_stderr=False).invoke(
        app,
        ["train", "-r", "missing-recipe.yaml", "-c", "missing-config.yaml"],
    )

    assert result.exit_code == 1
    assert "Recipe: missing-recipe.yaml" in result.stdout
    assert "[error] Settings 로딩 실패:" in result.stderr
    assert "missing-recipe.yaml" in result.stderr
