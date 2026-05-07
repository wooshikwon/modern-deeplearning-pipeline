"""E2E tests for CLI commands (init, list)."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
import yaml

from mdp.cli.init import (
    _adapter_from_catalog_default,
    _find_catalog_entry,
    _list_models_for_task,
    init_project,
)
from mdp.cli.list_cmd import run_list
from mdp.settings.schema import Recipe
from mdp.task_taxonomy import TASK_PRESETS


def _catalog_task_model_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for task in sorted(TASK_PRESETS):
        for model in _list_models_for_task(task):
            name = model.get("name")
            if isinstance(name, str) and name:
                cases.append((task, name))
    assert cases, "catalog-backed mdp init matrix is empty"
    return cases


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


@pytest.mark.parametrize(
    ("task", "model_name"),
    _catalog_task_model_cases(),
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_init_project_catalog_matrix_parseable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    model_name: str,
) -> None:
    """Every catalog-backed task/model init output is parseable as a Recipe."""
    monkeypatch.chdir(tmp_path)
    project_name = f"proj-{task}-{model_name}".replace(".", "-")

    init_project(project_name, task=task, model=model_name)

    recipe_path = tmp_path / project_name / "recipes" / "example.yaml"
    raw = yaml.safe_load(recipe_path.read_text())
    recipe = Recipe(**raw)

    assert recipe.task == task
    assert recipe.name
    assert isinstance(recipe.model, dict)
    assert isinstance(recipe.data.dataset, dict)
    assert isinstance(recipe.data.collator, dict)


@pytest.mark.parametrize(
    ("task", "model_name"),
    _catalog_task_model_cases(),
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_init_project_catalog_model_routing_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    model_name: str,
) -> None:
    """Generated model blocks keep URI routing compatible with Factory."""
    monkeypatch.chdir(tmp_path)
    project_name = f"route-{task}-{model_name}".replace(".", "-")

    init_project(project_name, task=task, model=model_name)

    recipe_path = tmp_path / project_name / "recipes" / "example.yaml"
    raw = yaml.safe_load(recipe_path.read_text())
    model = raw["model"]
    pretrained = model.get("pretrained")
    assert isinstance(pretrained, str) and pretrained

    if pretrained.startswith(("timm://", "ultralytics://")):
        assert "_component_" not in model
    else:
        assert "_component_" in model


def test_init_project_unsupported_task_model_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported catalog combinations must not silently emit fallback recipes."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        init_project("bad-combo", task="image_classification", model="gpt2")

    assert exc_info.value.exit_code == 1
    assert not (tmp_path / "bad-combo" / "recipes" / "example.yaml").exists()


@pytest.mark.parametrize("adapter_name", ["lora", "qlora"])
def test_catalog_adapter_shorthand_expands_to_component(adapter_name: str) -> None:
    """Catalog adapter shorthand expands into component-unified adapter config."""
    catalog = _find_catalog_entry("gpt2")
    assert catalog is not None

    adapter = _adapter_from_catalog_default(adapter_name, catalog)

    assert adapter["_component_"] == ("QLoRA" if adapter_name == "qlora" else "LoRA")
    assert "target" in adapter
    if adapter_name == "qlora":
        assert adapter["quantization"]["bits"] == 4


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
