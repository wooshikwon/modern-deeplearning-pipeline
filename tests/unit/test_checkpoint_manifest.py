"""Manifest-aware checkpoint manager tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mdp.training._checkpoint import (
    CHECKPOINT_LAYOUT_VERSION,
    MANIFEST_FILE,
    CheckpointContext,
    CheckpointManager,
    CheckpointManifest,
    LoadedCheckpoint,
    ModelRecord,
    ModelSlot,
)
from mdp.training.callbacks.checkpoint import ModelCheckpoint
from mdp.training.strategies.base import StrategyCheckpointCapability


class _UnsupportedStrategy:
    @property
    def checkpoint_capability(self) -> StrategyCheckpointCapability:
        return StrategyCheckpointCapability(
            unsupported_reason="DeepSpeed ZeRO engine checkpoint"
        )

    def save_checkpoint(self, model: torch.nn.Module, path: str) -> None:  # noqa: ARG002
        raise AssertionError("unsupported strategies must fail before save")


class _CollectiveStrategy:
    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    @property
    def checkpoint_capability(self) -> StrategyCheckpointCapability:
        return StrategyCheckpointCapability(
            supports_managed_checkpoint=True,
            requires_all_ranks_for_save=True,
            weight_format="test_collective",
        )

    def save_checkpoint(self, model: torch.nn.Module, path: str) -> None:  # noqa: ARG002
        self.saved_paths.append(path)


def test_checkpoint_manifest_json_round_trip(tmp_path: Path) -> None:
    manifest = CheckpointManifest(
        kind="rl",
        saved_at="step",
        global_step=120,
        trainer_state_file="trainer_state.json",
        recipe_file="recipe.yaml",
        models={
            "policy": ModelRecord(
                role="policy",
                format="torch_state_dict",
                path="policy/model.pt",
                trainable=True,
                optimizer="policy/optimizer.pt",
                scheduler="policy/scheduler.pt",
            )
        },
        scaler="scaler.pt",
        epoch=3,
        step_in_epoch=7,
        metrics={"loss": 1.25},
    )

    encoded = json.dumps(manifest.to_dict())
    decoded = CheckpointManifest.from_dict(json.loads(encoded))
    decoded.write(tmp_path)

    restored = CheckpointManifest.read(tmp_path)
    assert restored == decoded
    assert restored.layout_version == CHECKPOINT_LAYOUT_VERSION
    assert restored.models["policy"].optimizer == "policy/optimizer.pt"


def test_checkpoint_manifest_rejects_unknown_layout_version() -> None:
    with pytest.raises(ValueError, match="layout_version"):
        CheckpointManifest.from_dict(
            {
                "layout_version": 999,
                "kind": "sft",
                "saved_at": "manual",
                "global_step": 0,
            }
        )


def test_checkpoint_manager_save_writes_manifest_and_loads(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager()
    ckpt_dir = tmp_path / "checkpoint-10"

    result = manager.save(
        CheckpointContext(
            kind="sft",
            ckpt_dir=ckpt_dir,
            global_step=10,
            epoch=1,
            step_in_epoch=4,
            saved_at="step",
            metrics={"loss": 0.5},
            recipe_dict={"task": "sft"},
        ),
        [
            ModelSlot(
                name="",
                role="model",
                model=model,
                trainable=True,
                optimizer=optimizer,
            )
        ],
    )

    assert result == ckpt_dir
    assert (ckpt_dir / MANIFEST_FILE).exists()
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "optimizer.pt").exists()

    loaded = manager.load(ckpt_dir)
    assert isinstance(loaded, LoadedCheckpoint)
    assert loaded.legacy is False
    assert loaded.manifest is not None
    assert loaded.trainer_state is not None
    assert loaded.trainer_state["global_step"] == 10
    assert loaded.manifest.models["model"].path == "model.pt"
    assert loaded.manifest.models["model"].optimizer == "optimizer.pt"


def test_checkpoint_manager_restore_uses_manifest_record_paths(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    for param in target.parameters():
        param.data.zero_()
    optimizer = torch.optim.SGD(target.parameters(), lr=0.1)
    manager = CheckpointManager()
    ckpt_dir = tmp_path / "checkpoint-10"

    manager.save(
        CheckpointContext(
            kind="sft",
            ckpt_dir=ckpt_dir,
            global_step=10,
        ),
        [
            ModelSlot(
                name="",
                role="policy",
                model=source,
                trainable=True,
                optimizer=optimizer,
            )
        ],
    )

    manager.restore(
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

    assert torch.allclose(target.weight, source.weight)


def test_checkpoint_manager_restore_hf_pretrained_bin_record(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    for param in target.parameters():
        param.data.zero_()
    ckpt_dir = tmp_path / "checkpoint-10"
    ckpt_dir.mkdir()
    torch.save(source.state_dict(), ckpt_dir / "pytorch_model.bin")
    CheckpointManifest(
        kind="sft",
        global_step=10,
        models={
            "model": ModelRecord(
                role="policy",
                format="hf_pretrained",
                path="pytorch_model.bin",
                trainable=True,
            )
        },
    ).write(ckpt_dir)

    CheckpointManager().restore(
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

    assert torch.allclose(target.weight, source.weight)


def test_checkpoint_manager_rejects_unsupported_strategy(tmp_path: Path) -> None:
    manager = CheckpointManager()
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="DeepSpeed ZeRO"):
        manager.save(
            CheckpointContext(
                kind="sft",
                ckpt_dir=tmp_path / "checkpoint-10",
                global_step=10,
            ),
            [
                ModelSlot(
                    name="",
                    role="model",
                    model=model,
                    trainable=True,
                )
            ],
            strategy=_UnsupportedStrategy(),
        )


def test_checkpoint_manager_non_main_rank_enters_collective_save(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager()
    model = torch.nn.Linear(2, 1)
    strategy = _CollectiveStrategy()
    ckpt_dir = tmp_path / "checkpoint-10"

    manager.save(
        CheckpointContext(
            kind="sft",
            ckpt_dir=ckpt_dir,
            global_step=10,
            is_main_process=False,
        ),
        [
            ModelSlot(
                name="",
                role="model",
                model=model,
                trainable=True,
            )
        ],
        strategy=strategy,
    )

    assert strategy.saved_paths == [str(ckpt_dir / "model.safetensors")]
    assert not (ckpt_dir / MANIFEST_FILE).exists()


def test_model_checkpoint_non_main_rank_preserves_collective_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    strategy = _CollectiveStrategy()
    callback = ModelCheckpoint(dirpath=tmp_path)
    callback.save_checkpoint(
        torch.nn.Linear(2, 1),
        None,
        None,
        epoch=0,
        global_step=10,
        strategy=strategy,
    )

    assert strategy.saved_paths == [str(tmp_path / "checkpoint-10" / "model.safetensors")]
    assert callback.saved_checkpoints == []


def test_checkpoint_manager_reads_manifestless_legacy_checkpoint(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "legacy"
    ckpt_dir.mkdir()
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    torch.save({"scale": 128.0}, ckpt_dir / "scaler.pt")

    loaded = CheckpointManager().load(ckpt_dir)

    assert loaded.legacy is True
    assert loaded.manifest is None
    assert loaded.trainer_state == {"global_step": 2}
    assert loaded.scaler == {"scale": 128.0}
    assert loaded.to_legacy_state() == {
        "ckpt_dir": ckpt_dir,
        "trainer_state": {"global_step": 2},
        "scaler": {"scale": 128.0},
    }
