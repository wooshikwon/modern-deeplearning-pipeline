"""Global test fixtures for MDP test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Always use CPU for test reliability."""
    return torch.device("cpu")


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Temporary directory for checkpoint tests."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir
