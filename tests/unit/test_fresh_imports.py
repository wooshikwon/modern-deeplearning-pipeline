"""Fresh-process import smoke tests for public runtime entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "statement",
    [
        "from mdp.factory.factory import Factory; print(Factory)",
        "from mdp.runtime.engine import ExecutionEngine; print(ExecutionEngine)",
        "from mdp.training import Trainer; print(Trainer)",
    ],
)
def test_public_entry_points_import_in_fresh_process(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
