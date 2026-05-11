"""Artifact layout inventory contracts for the serving/checkpoint boundary."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mdp.cli.export import run_export
from mdp.artifacts.layout import detect_weight_layout, get_adapter_name
from mdp.artifacts.serving import ServingArtifactManager
from mdp.serving.model_loader import resolve_artifact_load_plan
from mdp.settings.components import ComponentSpec
from mdp.training._checkpoint import CheckpointContext, CheckpointManager, ModelSlot


@dataclass(frozen=True)
class LayoutInventoryCase:
    name: str
    files: dict[str, str]
    artifact_kind: str
    weight_format: str | None
    adapter_policy: str
    lifecycle: str


@pytest.mark.parametrize(
    "case",
    [
        LayoutInventoryCase(
            name="flat_serving_safetensors",
            files={"recipe.yaml": "name: flat\n", "model.safetensors": ""},
            artifact_kind="serving_artifact",
            weight_format="safetensors",
            adapter_policy="suppress_recipe_adapter",
            lifecycle="write_read",
        ),
        LayoutInventoryCase(
            name="hf_save_pretrained_unsharded_directory",
            files={
                "recipe.yaml": "name: hf\n",
                "config.json": "{}",
                "model.safetensors": "",
                "tokenizer.json": "{}",
            },
            artifact_kind="serving_artifact",
            weight_format="hf_pretrained_dir",
            adapter_policy="suppress_recipe_adapter",
            lifecycle="write_read",
        ),
        LayoutInventoryCase(
            name="peft_adapter_artifact",
            files={
                "recipe.yaml": "name: peft\n",
                "adapter_config.json": json.dumps({"adapter_name": "trained"}),
                "adapter_model.safetensors": "",
            },
            artifact_kind="serving_artifact",
            weight_format="peft_adapter",
            adapter_policy="load_peft_adapter_artifact",
            lifecycle="write_read_adapter_only",
        ),
        LayoutInventoryCase(
            name="custom_export_artifact",
            files={
                "recipe.yaml": "name: custom\n",
                "export_info.json": json.dumps({"format": "custom"}),
            },
            artifact_kind="serving_artifact",
            weight_format="export",
            adapter_policy="suppress_recipe_adapter",
            lifecycle="write_read_custom_hook",
        ),
        LayoutInventoryCase(
            name="manifestless_legacy_training_checkpoint",
            files={"trainer_state.json": "{}", "model.pt": ""},
            artifact_kind="legacy_checkpoint",
            weight_format="torch_state_dict",
            adapter_policy="preserve_recipe_adapter",
            lifecycle="read_only_compatibility",
        ),
    ],
    ids=lambda case: case.name,
)
def test_current_artifact_layout_inventory(
    tmp_path: Path,
    case: LayoutInventoryCase,
) -> None:
    """Inventory layouts that U2's descriptor must continue to represent."""
    for filename, content in case.files.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    plan = resolve_artifact_load_plan(tmp_path)

    assert plan.artifact_kind == case.artifact_kind
    assert plan.weight_format == case.weight_format
    assert plan.adapter_policy == case.adapter_policy
    assert plan.legacy_policy == (
        "read_only" if case.lifecycle == "read_only_compatibility" else None
    )
    assert case.lifecycle in {
        "write_read",
        "write_read_adapter_only",
        "write_read_custom_hook",
        "read_only_compatibility",
    }


def test_rl_multi_role_manifest_layout_inventory(tmp_path: Path) -> None:
    """RL checkpoints expose role-specific model records through one manifest."""
    policy_dir = tmp_path / "policy"
    reward_dir = tmp_path / "reward"
    policy_dir.mkdir()
    reward_dir.mkdir()
    (policy_dir / "model.safetensors").touch()
    (reward_dir / "model.safetensors").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "layout_version": 2,
            "kind": "rl",
            "saved_at": "validation",
            "global_step": 7,
            "models": {
                "policy": {
                    "role": "policy",
                    "format": "safetensors",
                    "path": "policy/model.safetensors",
                    "trainable": True,
                },
                "reward": {
                    "role": "reward",
                    "format": "safetensors",
                    "path": "reward/model.safetensors",
                    "trainable": False,
                },
            },
        })
    )

    reward_plan = resolve_artifact_load_plan(tmp_path, role="reward")

    assert reward_plan.artifact_kind == "training_checkpoint"
    assert reward_plan.role == "reward"
    assert reward_plan.weights_dir == reward_dir
    assert reward_plan.adapter_policy == "preserve_recipe_adapter"


def test_hf_sharded_save_pretrained_directory_is_not_flat_primary_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "recipe.yaml").write_text("name: sharded\n")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    (tmp_path / "model-00002-of-00002.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 2},
            "weight_map": {
                "layer_a.weight": "model-00001-of-00002.safetensors",
                "layer_b.weight": "model-00002-of-00002.safetensors",
            },
        })
    )

    plan = resolve_artifact_load_plan(tmp_path)
    layout = detect_weight_layout(tmp_path)

    assert plan.artifact_kind == "serving_artifact"
    assert plan.weight_format == "hf_pretrained_dir"
    assert layout.kind == "hf_pretrained_dir"
    assert layout.is_sharded is True
    assert {path.name for path in layout.files} == {
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_adapter_name_detection_is_neutral_layout_utility(tmp_path: Path) -> None:
    assert get_adapter_name(tmp_path) == "default"

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"adapter_name": "trained"})
    )

    assert get_adapter_name(tmp_path) == "trained"


def test_mlflow_snapshot_adapter_layout_is_adapter_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training-time MLflow snapshot keeps adapter-only/no-merge semantics."""

    class _AdapterConfig:
        def save_pretrained(self, output_dir: str) -> None:
            (Path(output_dir) / "adapter_config.json").write_text(
                json.dumps({"adapter_name": "default"})
            )

    class _AdapterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.peft_config = {"default": _AdapterConfig()}

    recipe = SimpleNamespace(
        data=SimpleNamespace(
            collator=ComponentSpec(
                component="mdp.data.collators.ClassificationCollator",
                kwargs={},
            )
        ),
        model_dump=lambda mode: {"name": "adapter-snapshot", "mode": mode},
    )
    settings = SimpleNamespace(recipe=recipe)

    def _get_peft_model_state_dict(
        model: Any,
        *,
        state_dict: dict[str, torch.Tensor],
        adapter_name: str,
    ) -> dict[str, torch.Tensor]:
        assert adapter_name == "default"
        assert state_dict
        return {"adapter.weight": torch.ones(1)}

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(get_peft_model_state_dict=_get_peft_model_state_dict),
    )

    ServingArtifactManager().write(
        _AdapterModel(),
        settings,
        tmp_path,
        mode="mlflow_snapshot",
        policy_state_dict={"base.weight": torch.ones(1)},
    )

    assert (tmp_path / "adapter_model.safetensors").exists()
    assert (tmp_path / "adapter_config.json").exists()
    assert (tmp_path / "recipe.yaml").exists()
    assert not (tmp_path / "model.safetensors").exists()


def test_deployment_export_requests_merged_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`mdp export` is a deployment package path and asks reconstruction to merge."""
    from mdp.serving import model_loader

    source_dir = tmp_path / "source"
    output_dir = tmp_path / "exported"
    source_dir.mkdir()
    (source_dir / "recipe.yaml").write_text("name: export\n")

    calls: list[tuple[Path, bool]] = []

    settings = SimpleNamespace(
        recipe=SimpleNamespace(
            data=SimpleNamespace(
                collator=ComponentSpec(
                    component="mdp.data.collators.ClassificationCollator",
                    kwargs={},
                )
            )
        )
    )

    def _reconstruct_model(source: Path, merge: bool) -> tuple[torch.nn.Module, Any]:
        calls.append((source, merge))
        return torch.nn.Linear(1, 1), settings

    monkeypatch.setattr(model_loader, "reconstruct_model", _reconstruct_model)

    run_export(checkpoint=str(source_dir), output=str(output_dir))

    assert calls == [(source_dir, True)]
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "recipe.yaml").exists()


def test_new_writers_do_not_create_legacy_layouts(tmp_path: Path) -> None:
    settings = SimpleNamespace(
        recipe=SimpleNamespace(
            data=SimpleNamespace(
                collator=ComponentSpec(
                    component="mdp.data.collators.ClassificationCollator",
                    kwargs={},
                )
            ),
            model_dump=lambda mode: {"name": "new-writer", "mode": mode},
        )
    )

    serving_dir = tmp_path / "serving"
    ServingArtifactManager().write(
        torch.nn.Linear(2, 1),
        settings,
        serving_dir,
        mode="mlflow_snapshot",
    )
    serving_plan = resolve_artifact_load_plan(serving_dir)

    checkpoint_dir = tmp_path / "checkpoint"
    CheckpointManager().save(
        CheckpointContext(kind="sft", ckpt_dir=checkpoint_dir, global_step=1),
        [
            ModelSlot(
                name="",
                role="model",
                model=torch.nn.Linear(2, 1),
                trainable=True,
            )
        ],
    )

    assert serving_plan.artifact_kind == "serving_artifact"
    assert serving_plan.legacy_policy is None
    assert (checkpoint_dir / "manifest.json").exists()
