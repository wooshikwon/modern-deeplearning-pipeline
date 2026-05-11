"""Shared helpers for GPU shell-CLI acceptance tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_mdp_json(args: list[str], *, cwd: Path) -> dict:
    """Run ``python -m mdp --format json`` and return the JSON payload."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(
        [sys.executable, "-m", "mdp", "--format", "json", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"mdp {' '.join(args)} failed with rc={result.returncode}\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert '"status"' not in result.stderr
    assert '"command"' not in result.stderr
    return json.loads(result.stdout)


def run_mdp_expect_failure(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m mdp --format json`` and assert it fails."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(
        [sys.executable, "-m", "mdp", "--format", "json", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    return result
