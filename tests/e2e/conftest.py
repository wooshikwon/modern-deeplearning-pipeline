"""e2e 테스트 공용 fixture."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    Recipe,
    Settings,
    TrainingSpec,
)


def e2e_artifact_dir(tmp_path: Path, test_name: str, *parts: str) -> Path:
    """Return per-test artifacts under the cloud artifact root when configured."""
    root = os.environ.get("MDP_TEST_ARTIFACT_DIR")
    if root:
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in test_name)
        path = Path(root) / "tests" / safe_name
    else:
        path = tmp_path
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_checkpoint_callbacks_yaml(tmp_path: Path, *, save_top_k: int = 2) -> str:
    callbacks = [
        {
            "_component_": "mdp.training.callbacks.checkpoint.ModelCheckpoint",
            "every_n_steps": 1,
            "save_top_k": save_top_k,
            "monitor": "loss",
            "mode": "min",
        }
    ]
    path = tmp_path / "callbacks.yaml"
    yaml.safe_dump(callbacks, path.open("w"))
    return str(path)


def make_test_settings(
    *,
    task: str = "image_classification",
    epochs: int = 3,
    max_steps: int | None = None,
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    val_check_interval: float = 1.0,
    val_check_unit: str = "epoch",
    model_class: str = "tests.e2e.models.TinyVisionModel",
    optimizer: dict | None = None,
    scheduler: dict | None = None,
    monitoring_enabled: bool = False,
    checkpoint_dir: str = "./checkpoints",
    name: str = "test-experiment",
) -> Settings:
    """e2e 테스트용 최소 Settings를 생성한다."""
    recipe = Recipe(
        name=name,
        task=task,
        model={"_component_": model_class},
        data=DataSpec(
            dataset={"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            collator={"_component_": "mdp.data.collators.CausalLMCollator", "tokenizer": "gpt2"},
        ),
        training=TrainingSpec(
            epochs=epochs,
            max_steps=max_steps,
            precision=precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            val_check_interval=val_check_interval,
            val_check_unit=val_check_unit,
        ),
        optimizer=optimizer or {"_component_": "AdamW", "lr": 1e-3},
        scheduler=scheduler,
        metadata=MetadataSpec(author="test", description="e2e test"),
    )

    if monitoring_enabled:
        recipe.monitoring.enabled = True

    config = Config()
    config.job.resume = "disabled"
    config.storage.checkpoint_dir = checkpoint_dir

    return Settings(recipe=recipe, config=config)
