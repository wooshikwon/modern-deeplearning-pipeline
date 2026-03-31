"""E2E tests for CLI commands (init, list)."""

from __future__ import annotations

from pathlib import Path

from mdp.cli.init import init_project
from mdp.cli.list_cmd import run_list


# ---------------------------------------------------------------------------
# init_project
# ---------------------------------------------------------------------------


def test_init_project(tmp_path: Path, monkeypatch) -> None:
    """init_project creates the expected directory structure."""
    monkeypatch.chdir(tmp_path)
    project_name = "my-test-project"

    init_project(project_name)

    root = tmp_path / project_name
    assert root.is_dir()
    assert (root / "configs").is_dir()
    assert (root / "recipes").is_dir()
    assert (root / "data").is_dir()
    assert (root / "checkpoints").is_dir()
    assert (root / "configs" / "local.yaml").is_file()
    assert (root / "recipes" / "example.yaml").is_file()
    assert (root / ".gitignore").is_file()

    # Verify file content is non-empty
    assert (root / "configs" / "local.yaml").stat().st_size > 0
    assert (root / "recipes" / "example.yaml").stat().st_size > 0


# ---------------------------------------------------------------------------
# run_list
# ---------------------------------------------------------------------------


def test_list_models(capsys) -> None:
    """run_list('models') executes without error."""
    run_list("models")
    # No assertion on output content -- just verify it doesn't crash


def test_list_tasks(capsys) -> None:
    """run_list('tasks') executes without error."""
    run_list("tasks")
    captured = capsys.readouterr()
    assert "image_classification" in captured.out


def test_list_strategies(capsys) -> None:
    """run_list('strategies') executes without error."""
    run_list("strategies")
    captured = capsys.readouterr()
    assert "ddp" in captured.out or "DDP" in captured.out or "fsdp" in captured.out
