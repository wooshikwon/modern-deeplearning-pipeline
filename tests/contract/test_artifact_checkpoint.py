"""Artifact and checkpoint public-internal contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import typer

from mdp.cli.output import ModelSourcePlan, resolve_model_source_plan
from mdp.serving.model_loader import ArtifactLoadPlan, resolve_artifact_load_plan
from mdp.training._checkpoint import (
    CHECKPOINT_LAYOUT_VERSION,
    MANIFEST_FILE,
    CheckpointContext,
    CheckpointManager,
    ModelSlot,
)
from mdp.training.strategies.base import (
    BaseStrategy,
    StrategyCheckpointCapability,
)
from mdp.training.strategies.ddp import DDPStrategy
from mdp.training.strategies.deepspeed import DeepSpeedStrategy
from mdp.training.strategies.fsdp import FSDPStrategy


def test_checkpoint_manager_writes_manifest_layout_contract(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ckpt_dir = tmp_path / "checkpoint-12"

    result = CheckpointManager().save(
        CheckpointContext(
            kind="sft",
            ckpt_dir=ckpt_dir,
            global_step=12,
            epoch=2,
            step_in_epoch=4,
            saved_at="validation",
            metrics={"val_loss": 0.25},
            recipe_dict={"name": "contract", "task": "image_classification"},
        ),
        [
            ModelSlot(
                name="",
                role="model",
                model=model,
                trainable=True,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        ],
        scaler={"scale": 128.0},
    )

    manifest = json.loads((result / MANIFEST_FILE).read_text())

    assert result == ckpt_dir
    assert manifest["layout_version"] == CHECKPOINT_LAYOUT_VERSION
    assert manifest["kind"] == "sft"
    assert manifest["saved_at"] == "validation"
    assert manifest["global_step"] == 12
    assert manifest["trainer_state_file"] == "trainer_state.json"
    assert manifest["recipe_file"] == "recipe.yaml"
    assert manifest["scaler"] == "scaler.pt"
    assert manifest["models"] == {
        "model": {
            "role": "model",
            "format": "torch_state_dict",
            "path": "model.pt",
            "trainable": True,
            "optimizer": "optimizer.pt",
            "scheduler": "scheduler.pt",
        }
    }


def test_checkpoint_manager_restores_manifest_and_legacy_contract(
    tmp_path: Path,
) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    for param in target.parameters():
        param.data.zero_()
    ckpt_dir = tmp_path / "checkpoint-3"

    CheckpointManager().save(
        CheckpointContext(kind="sft", ckpt_dir=ckpt_dir, global_step=3),
        [
            ModelSlot(
                name="",
                role="policy",
                model=source,
                trainable=True,
            )
        ],
    )

    loaded = CheckpointManager().restore(
        ckpt_dir,
        [
            ModelSlot(
                name="",
                role="policy",
                model=target,
                trainable=True,
            )
        ],
    )

    assert loaded.legacy is False
    assert loaded.manifest is not None
    assert torch.allclose(target.weight, source.weight)

    legacy_dir = tmp_path / "legacy-checkpoint"
    legacy_dir.mkdir()
    (legacy_dir / "trainer_state.json").write_text(json.dumps({"global_step": 9}))
    legacy = CheckpointManager().load(legacy_dir)

    assert legacy.legacy is True
    assert legacy.manifest is None
    assert legacy.trainer_state == {"global_step": 9}


@pytest.mark.parametrize(
    ("command", "supports_pretrained"),
    [
        ("inference", True),
        ("generate", True),
        ("serve", False),
        ("export", False),
    ],
)
def test_model_source_plan_artifact_command_matrix(
    command: str,
    supports_pretrained: bool,
) -> None:
    plan = resolve_model_source_plan(None, "/models/exported", command)

    assert isinstance(plan, ModelSourcePlan)
    assert plan.kind == "artifact"
    assert plan.command == command
    assert str(plan.path) == "/models/exported"
    assert plan.uri is None
    assert plan.supports_pretrained is supports_pretrained
    assert plan.is_artifact
    assert not plan.is_pretrained


def test_model_source_plan_run_id_downloads_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mlflow.artifacts.download_artifacts",
        lambda run_id, artifact_path: f"/tmp/mlruns/{run_id}/{artifact_path}",
    )

    plan = resolve_model_source_plan("abc123", None, "inference")

    assert plan.kind == "artifact"
    assert str(plan.path) == "/tmp/mlruns/abc123/model"


def test_model_source_plan_pretrained_is_command_scoped() -> None:
    plan = resolve_model_source_plan(
        None,
        None,
        "generate",
        pretrained="hf://gpt2",
    )

    assert plan.kind == "pretrained"
    assert plan.uri == "hf://gpt2"
    assert plan.path is None
    assert plan.is_pretrained

    with pytest.raises(typer.BadParameter, match="inference, generate"):
        resolve_model_source_plan(None, None, "serve", pretrained="hf://gpt2")


def test_model_source_plan_rejects_ambiguous_sources() -> None:
    with pytest.raises(typer.BadParameter, match="하나만"):
        resolve_model_source_plan(
            "abc123",
            "/models/exported",
            "inference",
            pretrained="hf://gpt2",
        )


def test_artifact_load_plan_selects_manifest_role_path(tmp_path: Path) -> None:
    policy_dir = tmp_path / "policy"
    reward_dir = tmp_path / "reward"
    policy_dir.mkdir()
    reward_dir.mkdir()
    (policy_dir / "model.safetensors").touch()
    (reward_dir / "model.safetensors").touch()
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps({
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
            }
        })
    )

    plan = resolve_artifact_load_plan(tmp_path, role="reward", merge=True)

    assert isinstance(plan, ArtifactLoadPlan)
    assert plan.artifact_kind == "training_checkpoint"
    assert plan.role == "reward"
    assert plan.weight_format == "safetensors"
    assert plan.weights_dir == reward_dir
    assert plan.merge is True
    assert plan.adapter_policy == "preserve_recipe_adapter"


@pytest.mark.parametrize(
    ("files", "artifact_kind", "weight_format", "adapter_policy"),
    [
        (
            {"recipe.yaml": "", "model.safetensors": ""},
            "serving_artifact",
            "safetensors",
            "suppress_recipe_adapter",
        ),
        (
            {"recipe.yaml": "", "adapter_model.safetensors": ""},
            "serving_artifact",
            "peft_adapter",
            "load_peft_adapter_artifact",
        ),
        (
            {"model.pt": ""},
            "legacy_checkpoint",
            "torch_state_dict",
            "preserve_recipe_adapter",
        ),
        ({}, "legacy_checkpoint", None, "preserve_recipe_adapter"),
    ],
)
def test_artifact_load_plan_classifies_artifact_layouts(
    tmp_path: Path,
    files: dict[str, str],
    artifact_kind: str,
    weight_format: str | None,
    adapter_policy: str,
) -> None:
    for name, content in files.items():
        (tmp_path / name).write_text(content)

    plan = resolve_artifact_load_plan(tmp_path)

    assert plan.artifact_kind == artifact_kind
    assert plan.role == "policy"
    assert plan.weight_format == weight_format
    assert plan.weights_dir == tmp_path
    assert plan.adapter_policy == adapter_policy


def test_artifact_load_plan_rejects_missing_manifest_role(tmp_path: Path) -> None:
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps({
            "models": {
                "policy": {
                    "role": "policy",
                    "format": "safetensors",
                    "path": "policy/model.safetensors",
                    "trainable": True,
                }
            }
        })
    )

    with pytest.raises(ValueError, match="role='reward'"):
        resolve_artifact_load_plan(tmp_path, role="reward")


class _ImplicitUnsupportedStrategy(BaseStrategy):
    def setup(self, model, device, optimizer=None):  # noqa: ANN001, ANN201, ARG002
        return model

    def save_checkpoint(self, model, path: str) -> None:  # noqa: ANN001, ARG002
        raise AssertionError("unsupported")

    def load_checkpoint(self, model, path: str):  # noqa: ANN001, ANN201, ARG002
        return model


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            DDPStrategy(),
            StrategyCheckpointCapability(
                supports_managed_checkpoint=True,
                weight_format="safetensors",
            ),
        ),
        (
            FSDPStrategy(),
            StrategyCheckpointCapability(
                supports_managed_checkpoint=True,
                requires_all_ranks_for_save=True,
                weight_format="safetensors_full_state_dict",
            ),
        ),
        (
            DeepSpeedStrategy(),
            StrategyCheckpointCapability(
                unsupported_reason=(
                    "DeepSpeedStrategy does not declare manifest checkpoint "
                    "compatibility"
                )
            ),
        ),
        (
            _ImplicitUnsupportedStrategy(),
            StrategyCheckpointCapability(
                unsupported_reason=(
                    "_ImplicitUnsupportedStrategy does not declare manifest "
                    "checkpoint compatibility"
                )
            ),
        ),
    ],
)
def test_strategy_checkpoint_capability_matrix(
    strategy: BaseStrategy,
    expected: StrategyCheckpointCapability,
) -> None:
    assert strategy.checkpoint_capability == expected
