"""Manifest-aware checkpoint manager tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    resolve_checkpoint_dir,
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

    def unwrap(self, wrapped_model: torch.nn.Module) -> torch.nn.Module:
        return wrapped_model


class _ManagedStrategy:
    def __init__(self) -> None:
        self.unwrap_called = False

    @property
    def checkpoint_capability(self) -> StrategyCheckpointCapability:
        return StrategyCheckpointCapability(
            supports_managed_checkpoint=True,
            weight_format="safetensors",
        )

    def unwrap(self, wrapped_model: torch.nn.Module) -> torch.nn.Module:
        self.unwrap_called = True
        return wrapped_model.inner  # type: ignore[attr-defined]

    def save_checkpoint(self, model: torch.nn.Module, path: str) -> None:  # noqa: ARG002
        raise AssertionError("PEFT adapter checkpoints must not save full state_dict")


class _WrappedModel(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner


def test_checkpoint_manifest_json_round_trip(tmp_path: Path) -> None:
    manifest = CheckpointManifest(
        kind="rl",
        saved_at="step",
        global_step=120,
        trainer_state_file="trainer_state.json",
        recipe_file="recipe.yaml",
        recipe_name="wntp_baseline",
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
    assert restored.recipe_name == "wntp_baseline"
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


def test_resolve_checkpoint_dir_uses_recipe_and_job_namespace() -> None:
    assert resolve_checkpoint_dir("./checkpoints") == Path("./checkpoints")
    assert resolve_checkpoint_dir(
        "./checkpoints",
        recipe_name="critic_pretrain",
        job_name="sanity-c8",
    ) == Path("./checkpoints/critic_pretrain/sanity-c8")


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
            recipe_dict={"name": "sft", "task": "sft"},
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
    assert loaded.manifest.recipe_name == "sft"


def test_checkpoint_manager_rejects_resume_from_different_recipe(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    ckpt_dir = tmp_path / "checkpoint"

    CheckpointManager().save(
        CheckpointContext(
            kind="sft",
            ckpt_dir=ckpt_dir,
            global_step=1,
            recipe_dict={"name": "wntp_baseline"},
        ),
        [
            ModelSlot(
                name="",
                role="policy",
                model=source,
                trainable=True,
            )
        ],
    )

    with pytest.raises(ValueError, match="recipe mismatch"):
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
            expected_recipe_name="critic_pretrain",
        )


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


def test_checkpoint_manager_restore_owns_manifestless_legacy_slots(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    for param in target.parameters():
        param.data.zero_()

    optimizer = torch.optim.SGD(target.parameters(), lr=0.5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler_state = {"scale": 128.0}
    ckpt_dir = tmp_path / "legacy"
    ckpt_dir.mkdir()
    torch.save(source.state_dict(), ckpt_dir / "model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    torch.save(scaler_state, ckpt_dir / "scaler.pt")
    (ckpt_dir / "trainer_state.json").write_text(json.dumps({"global_step": 2}))

    class _Scaler:
        def __init__(self) -> None:
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    scaler = _Scaler()

    loaded = CheckpointManager().restore(
        ckpt_dir,
        [
            ModelSlot(
                name="",
                role="policy",
                model=target,
                trainable=True,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        ],
        scaler=scaler,
    )

    assert loaded.legacy is True
    assert torch.allclose(target.weight, source.weight)
    assert scaler.loaded == scaler_state


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
        "legacy": True,
        "legacy_policy": "read_only",
    }


def test_checkpoint_manager_restores_named_pretrained_dir_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ShardedPretrained(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def save_pretrained(self, output_dir: Path) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text("{}")
            (output_dir / "model-00001-of-00002.safetensors").touch()
            (output_dir / "model.safetensors.index.json").write_text(
                json.dumps({
                    "metadata": {"total_size": 1},
                    "weight_map": {"weight": "model-00001-of-00002.safetensors"},
                })
            )

    calls: list[tuple[Any, str, bool, bool]] = []

    def _load_sharded_checkpoint(
        model: Any,
        folder: str,
        *,
        strict: bool = True,
        prefer_safe: bool = True,
    ) -> None:
        calls.append((model, folder, strict, prefer_safe))

    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers.modeling_utils",
        SimpleNamespace(load_sharded_checkpoint=_load_sharded_checkpoint),
    )

    ckpt_dir = tmp_path / "checkpoint"
    CheckpointManager().save(
        CheckpointContext(kind="rl", ckpt_dir=ckpt_dir, global_step=3),
        [
            ModelSlot(
                name="policy",
                role="policy",
                model=_ShardedPretrained(),
                trainable=True,
            )
        ],
    )
    manifest = json.loads((ckpt_dir / MANIFEST_FILE).read_text())

    assert manifest["models"]["policy"]["format"] == "pretrained_dir"
    assert manifest["models"]["policy"]["path"] == "policy"

    target = _ShardedPretrained()
    CheckpointManager().restore(
        ckpt_dir,
        [
            ModelSlot(
                name="policy",
                role="policy",
                model=target,
                trainable=True,
            )
        ],
    )

    assert len(calls) == 1
    assert calls[0][1:] == (str(ckpt_dir / "policy"), False, True)


def test_checkpoint_manager_prefers_peft_adapter_under_managed_strategy(tmp_path: Path) -> None:
    class _PeftLike(torch.nn.Module):
        peft_config = {"default": object()}

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def save_pretrained(self, output_dir: Path) -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "adapter_config.json").write_text("{}")
            (output_dir / "adapter_model.safetensors").touch()

    ckpt_dir = tmp_path / "checkpoint"
    strategy = _ManagedStrategy()
    CheckpointManager().save(
        CheckpointContext(kind="rl", ckpt_dir=ckpt_dir, global_step=4),
        [
            ModelSlot(
                name="policy",
                role="policy",
                model=_WrappedModel(_PeftLike()),
                trainable=True,
            )
        ],
        strategy=strategy,
    )

    manifest = json.loads((ckpt_dir / MANIFEST_FILE).read_text())

    assert strategy.unwrap_called is True
    assert manifest["models"]["policy"]["format"] == "peft_adapter"
    assert manifest["models"]["policy"]["path"] == "policy/adapter_model.safetensors"
    assert (ckpt_dir / "policy" / "adapter_model.safetensors").exists()
    assert not (ckpt_dir / "policy" / "model.safetensors").exists()


def test_checkpoint_manager_keeps_collective_strategy_save_for_peft(tmp_path: Path) -> None:
    class _PeftLike(torch.nn.Module):
        peft_config = {"default": object()}

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def save_pretrained(self, output_dir: Path) -> None:  # noqa: ARG002
            raise AssertionError("collective strategy must own PEFT checkpoint save")

    ckpt_dir = tmp_path / "checkpoint"
    strategy = _CollectiveStrategy()
    CheckpointManager().save(
        CheckpointContext(kind="rl", ckpt_dir=ckpt_dir, global_step=5),
        [
            ModelSlot(
                name="policy",
                role="policy",
                model=_PeftLike(),
                trainable=True,
            )
        ],
        strategy=strategy,
    )

    manifest = json.loads((ckpt_dir / MANIFEST_FILE).read_text())

    assert strategy.saved_paths == [str(ckpt_dir / "policy" / "model.safetensors")]
    assert manifest["models"]["policy"]["format"] == "test_collective"
