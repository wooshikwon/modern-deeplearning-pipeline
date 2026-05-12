"""E2E tests for CLI commands (init, list)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
from mdp.settings.components import ComponentSpec, ModelComponentSpec
from mdp.settings.schema import Recipe
from mdp.task_taxonomy import TASK_PRESETS


def _run_mdp(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    return subprocess.run(
        [sys.executable, "-m", "mdp", *args],
        cwd=cwd or Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


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
    from mdp.settings.loader import SettingsLoader

    monkeypatch.chdir(tmp_path)
    project_name = "my-test-project"

    init_project(project_name)

    root = tmp_path / project_name
    assert root.is_dir()
    assert (root / "configs").is_dir()
    assert (root / "recipes").is_dir()
    assert (root / "data").is_dir()
    assert (root / "checkpoints").is_dir()
    assert (root / "docs").is_dir()
    assert (root / "configs" / "local.yaml").is_file()
    assert (root / "recipes" / "example.yaml").is_file()
    assert (root / "AGENT.md").is_file()
    assert (root / "docs" / "getting-started.md").is_file()
    assert (root / "docs" / "cli-reference.md").is_file()
    assert (root / "docs" / "configuration.md").is_file()
    assert (root / ".gitignore").is_file()

    # Verify file content is non-empty
    assert (root / "configs" / "local.yaml").stat().st_size > 0
    assert (root / "recipes" / "example.yaml").stat().st_size > 0
    agent_text = (root / "AGENT.md").read_text()
    assert "docs/getting-started.md" in agent_text
    assert "docs/configuration.md" in agent_text

    settings = SettingsLoader().load_estimation_settings(str(root / "recipes" / "example.yaml"))
    assert settings.recipe.name
    assert settings.recipe.task
    assert settings.recipe.model.component is not None
    assert settings.recipe.model.pretrained is None
    assert settings.recipe.data.dataset.component
    assert settings.recipe.data.collator.component


def test_cli_entry_init_json_creates_project(tmp_path: Path) -> None:
    result = _run_mdp(
        ["--format", "json", "init", "entry-project", "--task", "image_classification", "--model", "resnet18"],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["command"] == "init"
    assert payload["project_name"] == "entry-project"
    assert payload["task"] == "image_classification"
    assert payload["model"] == "resnet18"
    assert (tmp_path / "entry-project" / "recipes" / "example.yaml").is_file()
    assert (tmp_path / "entry-project" / "AGENT.md").is_file()
    assert (tmp_path / "entry-project" / "docs" / "getting-started.md").is_file()


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
    expected_collators = {
        "image_classification": "VisionCollator",
        "object_detection": "VisionCollator",
        "semantic_segmentation": "VisionCollator",
        "text_classification": "ClassificationCollator",
        "token_classification": "TokenClassificationCollator",
        "text_generation": "CausalLMCollator",
        "seq2seq": "Seq2SeqCollator",
    }
    expected_datasets = {
        "image_classification": "ImageClassificationDataset",
    }
    monkeypatch.chdir(tmp_path)
    project_name = f"proj-{task}-{model_name}".replace(".", "-")

    init_project(project_name, task=task, model=model_name)

    recipe_path = tmp_path / project_name / "recipes" / "example.yaml"
    raw = yaml.safe_load(recipe_path.read_text())
    recipe = Recipe(**raw)

    assert recipe.task == task
    assert recipe.name
    assert isinstance(recipe.model, ModelComponentSpec)
    assert isinstance(recipe.data.dataset, ComponentSpec)
    assert isinstance(recipe.data.collator, ComponentSpec)

    expected_collator = expected_collators.get(task)
    if expected_collator is not None:
        assert recipe.data.collator.component.rsplit(".", 1)[-1] == expected_collator
    expected_dataset = expected_datasets.get(task)
    if expected_dataset is not None:
        assert recipe.data.dataset.component.rsplit(".", 1)[-1] == expected_dataset


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
    """Generated model blocks keep URI routing compatible with AssemblyMaterializer."""
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


def test_cli_entry_version_json() -> None:
    result = _run_mdp(["--format", "json", "version"])

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["command"] == "version"
    assert isinstance(payload["version"], str)


@pytest.mark.parametrize(
    ("target", "expected_key"),
    [
        ("tasks", "tasks"),
        ("models", "models"),
        ("callbacks", "callbacks"),
        ("strategies", "strategies"),
    ],
)
def test_cli_entry_list_discovery_json(target: str, expected_key: str) -> None:
    result = _run_mdp(["--format", "json", "list", target])

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["command"] == "list"
    assert isinstance(payload[expected_key], list)
    assert payload[expected_key]


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
