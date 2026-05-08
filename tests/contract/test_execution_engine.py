"""ExecutionEngine contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from mdp.models.base import BaseModel
from mdp.runtime.engine import ExecutionEngine
from mdp.settings.components import ComponentSpec
from mdp.settings.distributed import has_distributed_intent
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.run_plan_builder import normalize_callback_configs
from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    Recipe,
    RLSpec,
    Settings,
    TrainingSpec,
)
from tests.e2e.datasets import ListDataLoader, make_vision_batches


class TinyCausalLM(BaseModel):
    """BaseModel-compatible LM with the HF-style forward signature RLTrainer uses."""

    _block_classes = None

    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def _sft_settings(tmp_path: Path, *, max_steps: int = 1) -> Settings:
    recipe = Recipe(
        name="engine-sft",
        task="image_classification",
        model={
            "_component_": "tests.e2e.models.TinyVisionModel",
            "num_classes": 2,
            "hidden_dim": 16,
        },
        data=DataSpec(
            dataset={
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": str(tmp_path / "train.jsonl"),
                "split": "train",
            },
            collator={"_component_": "mdp.data.collators.VisionCollator"},
            dataloader={"batch_size": 2, "num_workers": 0},
        ),
        training=TrainingSpec(max_steps=max_steps),
        optimizer={"_component_": "AdamW", "lr": 1e-3},
        metadata=MetadataSpec(author="contract", description="engine sft"),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"
    settings.config.storage.checkpoint_dir = str(tmp_path / "checkpoints")
    return settings


def _dpo_settings(tmp_path: Path, *, max_steps: int = 1) -> Settings:
    recipe = Recipe(
        name="engine-dpo",
        task="text_generation",
        model={"_component_": "unused-top-level-model"},
        data=DataSpec(
            dataset={
                "_component_": "mdp.data.datasets.HuggingFaceDataset",
                "source": str(tmp_path / "train.jsonl"),
                "split": "train",
            },
            collator={
                "_component_": "mdp.data.collators.PreferenceCollator",
                "tokenizer": "gpt2",
                "max_length": 32,
            },
            dataloader={"batch_size": 2, "num_workers": 0},
        ),
        training=TrainingSpec(max_steps=max_steps),
        metadata=MetadataSpec(author="contract", description="engine dpo"),
        rl=RLSpec(
            algorithm={"_component_": "DPO", "beta": 0.1},
            models={
                "policy": {
                    "_component_": "tests.contract.test_execution_engine.TinyCausalLM",
                    "optimizer": {"_component_": "AdamW", "lr": 1e-3},
                },
                "reference": {
                    "_component_": "tests.contract.test_execution_engine.TinyCausalLM",
                },
            },
        ),
    )
    settings = Settings(recipe=recipe, config=Config())
    settings.config.job.resume = "disabled"
    settings.config.storage.checkpoint_dir = str(tmp_path / "checkpoints")
    return settings


def _pref_batches(n: int, batch_size: int, seq_len: int = 8, vocab: int = 32) -> list[dict]:
    return [
        {
            "chosen_input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
            "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "chosen_labels": torch.randint(1, vocab, (batch_size, seq_len)),
            "rejected_input_ids": torch.randint(1, vocab, (batch_size, seq_len)),
            "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "rejected_labels": torch.randint(1, vocab, (batch_size, seq_len)),
        }
        for _ in range(n)
    ]


def _sft_plan(settings: Settings, cb_configs: list[dict] | None = None) -> RunPlan:
    return RunPlan(
        command="train",
        mode="sft",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=normalize_callback_configs(cb_configs),
        validation_scope="training",
        distributed_intent=has_distributed_intent(settings),
    )


def _rl_plan(settings: Settings, cb_configs: list[dict] | None = None) -> RunPlan:
    return RunPlan(
        command="rl-train",
        mode="rl",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=normalize_callback_configs(cb_configs),
        validation_scope="training",
        distributed_intent=has_distributed_intent(settings),
    )


def test_engine_runs_sft_cpu_smoke_through_bundle_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _sft_settings(tmp_path)
    loaders = {
        "train": ListDataLoader(make_vision_batches(2, 2, 2, 8)),
        "val": None,
    }
    monkeypatch.setattr(
        "mdp.data.dataloader.create_dataloaders",
        lambda **kwargs: loaders,
    )

    result = ExecutionEngine().run(_sft_plan(settings))

    assert result["total_steps"] == 1
    assert result["stopped_reason"] == "max_steps_reached"
    assert "metrics" in result


def test_engine_runs_dpo_cpu_smoke_through_bundle_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _dpo_settings(tmp_path)
    monkeypatch.setattr(
        "mdp.data.dataloader.create_dataloaders",
        lambda **kwargs: {"train": ListDataLoader(_pref_batches(2, 2)), "val": None},
    )

    result = ExecutionEngine().run(_rl_plan(settings))

    assert result["total_steps"] == 1
    assert result["stopped_reason"] == "max_steps"
    assert result["algorithm"] == "DPOLoss"
    assert torch.isfinite(torch.tensor(result["metrics"]["loss"]))


def test_callback_configs_are_resolved_once_and_injected_into_sft_bundle(
    tmp_path: Path,
) -> None:
    run_plan = _sft_plan(
        _sft_settings(tmp_path),
        cb_configs=[
            {
                "_component_": "mdp.training.callbacks.checkpoint.ModelCheckpoint",
                "monitor": "loss",
                "save_top_k": 1,
            }
        ],
    )
    assembly_plan = SimpleNamespace(kind="sft_training")
    captured: dict[str, object] = {}

    class FakeMaterializer:
        def __init__(self, plan):
            captured["assembly_plan"] = plan
            self._callbacks = ["resolved-callback"]

        def materialize_callbacks(self):
            captured["callback_materialize_count"] = (
                captured.get("callback_materialize_count", 0) + 1
            )
            return self._callbacks

        def materialize_sft_training_bundle(self, *, callbacks=None):
            captured["bundle_callbacks"] = callbacks
            return "sft-bundle"

    class FakePlanner:
        @classmethod
        def from_run_plan(cls, plan):
            captured["run_plan"] = plan
            return assembly_plan

    fake_trainer = mock.Mock()
    fake_trainer.train.return_value = {"trained": True}

    with mock.patch(
        "mdp.training.trainer.Trainer.from_bundle",
        return_value=fake_trainer,
    ) as from_bundle:
        result = ExecutionEngine(
            assembly_planner=FakePlanner,
            materializer_cls=FakeMaterializer,
            callbacks_observer=lambda callbacks, settings: captured.setdefault(
                "observed_callbacks", callbacks
            ),
        ).run(run_plan)

    assert result == {"trained": True}
    assert captured["run_plan"].callback_configs == run_plan.callback_configs
    assert captured["assembly_plan"] is assembly_plan
    assert captured["callback_materialize_count"] == 1
    assert captured["bundle_callbacks"] == ["resolved-callback"]
    assert captured["observed_callbacks"] == ["resolved-callback"]
    from_bundle.assert_called_once_with("sft-bundle")


def test_engine_dispatches_rl_by_assembly_kind(tmp_path: Path) -> None:
    run_plan = _rl_plan(_dpo_settings(tmp_path))
    assembly_plan = SimpleNamespace(kind="rl_training")

    class FakePlanner:
        @classmethod
        def from_run_plan(cls, plan):
            return assembly_plan

    class FakeMaterializer:
        def __init__(self, plan):
            self.plan = plan

        def materialize_callbacks(self):
            return []

        def materialize_rl_training_bundle(self, *, callbacks=None):
            return "rl-bundle"

    fake_trainer = mock.Mock()
    fake_trainer.train.return_value = {"rl": True}

    with mock.patch(
        "mdp.training.rl_trainer.RLTrainer.from_bundle",
        return_value=fake_trainer,
    ) as from_bundle:
        result = ExecutionEngine(
            assembly_planner=FakePlanner,
            materializer_cls=FakeMaterializer,
        ).run(run_plan)

    assert result == {"rl": True}
    from_bundle.assert_called_once_with("rl-bundle")


def test_runtime_training_helper_runs_run_plan(
    tmp_path: Path,
) -> None:
    from mdp.runtime.training import run_training

    settings = _sft_settings(tmp_path)
    run_plan = _sft_plan(
        settings,
        cb_configs=[
            {"_component_": "ModelCheckpoint", "monitor": "loss", "save_top_k": 1}
        ],
    )
    calls: list[str] = []
    captured: dict[str, object] = {}
    engine = mock.Mock()
    engine.run.return_value = {"total_steps": 1, "stopped_reason": "completed"}

    def observer(callbacks, settings):
        pass

    def fake_liger() -> None:
        calls.append("liger")

    class FakeEngine:
        def __init__(self, *, callbacks_observer=None):
            calls.append("engine-init")
            captured["callbacks_observer"] = callbacks_observer
            self.callbacks_observer = callbacks_observer

        def run(self, plan):
            calls.append("engine-run")
            engine.run(plan)
            return engine.run.return_value

    with mock.patch(
        "mdp.runtime.worker.apply_liger_patches_for_training",
        side_effect=fake_liger,
    ), mock.patch("mdp.runtime.engine.ExecutionEngine", FakeEngine):
        result = run_training(run_plan, callbacks_observer=observer)

    plan = engine.run.call_args.args[0]
    assert result == {"total_steps": 1, "stopped_reason": "completed"}
    assert calls == ["liger", "engine-init", "engine-run"]
    assert captured["callbacks_observer"] is observer
    assert plan.command == "train"
    assert plan.mode == "sft"
    assert plan.settings is settings
    assert plan.callback_configs == (
        ComponentSpec(
            component="ModelCheckpoint",
            kwargs={"monitor": "loss", "save_top_k": 1},
            path="callbacks[0]",
        ),
    )


def test_torchrun_run_training_adapter_delegates_to_runtime_helper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mdp.cli._torchrun_entry as entry_mod

    run_plan = _sft_plan(_sft_settings(tmp_path), cb_configs=[{"_component_": "ProgressBar"}])
    captured: dict[str, object] = {}

    def fake_runtime_run_training(run_plan_arg, *, callbacks_observer=None):
        captured["run_plan"] = run_plan_arg
        captured["callbacks_observer"] = callbacks_observer
        return {"total_steps": 1}

    monkeypatch.setattr(
        "mdp.runtime.training.run_training",
        fake_runtime_run_training,
    )

    result = entry_mod.run_training(run_plan)
    from mdp.cli.callback_output import print_callbacks_log

    assert result == {"total_steps": 1}
    assert captured["run_plan"] is run_plan
    assert captured["callbacks_observer"] is print_callbacks_log


def test_torchrun_main_worker_contract_bootstraps_before_runtime_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import mdp.cli._torchrun_entry as entry_mod
    from mdp.cli import _logging_bootstrap

    run_plan = _sft_plan(_sft_settings(tmp_path))
    run_plan_path = tmp_path / "run_plan.json"
    result_path = tmp_path / "result.json"
    from mdp.runtime.payload import RunPlanPayload

    run_plan_path.write_text(
        json.dumps(RunPlanPayload.from_run_plan(run_plan).to_json_dict(), default=str)
    )
    events: list[str] = []

    def fake_bootstrap(settings_arg=None):
        events.append("bootstrap-settings" if settings_arg is not None else "bootstrap-env")

    def fake_dist_init(settings_arg):
        assert settings_arg.recipe.name == "engine-sft"
        events.append("dist-init")

    def fake_run_training(run_plan_arg):
        assert run_plan_arg.settings.recipe.name == "engine-sft"
        events.append("run-training")
        return {"total_steps": 1, "stopped_reason": "completed"}

    monkeypatch.setattr(_logging_bootstrap, "bootstrap_logging", fake_bootstrap)
    monkeypatch.setattr(entry_mod, "_init_distributed_if_torchrun", fake_dist_init)
    monkeypatch.setattr(entry_mod, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_torchrun_entry.py",
            "--run-plan-path",
            str(run_plan_path),
            "--result-path",
            str(result_path),
        ],
    )

    entry_mod.main()

    assert events == [
        "bootstrap-env",
        "bootstrap-settings",
        "dist-init",
        "run-training",
    ]
    assert json.loads(result_path.read_text()) == {
        "total_steps": 1,
        "stopped_reason": "completed",
    }
