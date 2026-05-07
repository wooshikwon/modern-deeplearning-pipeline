"""Local CLI error UX contract tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from mdp.__main__ import app


def test_train_missing_yaml_error_json_mode() -> None:
    """JSON mode emits a structured error payload on validation failure."""
    result = CliRunner().invoke(
        app,
        ["--format", "json", "train", "-r", "missing-recipe.yaml", "-c", "missing-config.yaml"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["command"] == "train"
    assert payload["error"]["type"] == "ValidationError"
    assert "missing-recipe.yaml" in payload["error"]["message"]


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
