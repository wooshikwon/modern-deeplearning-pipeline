"""e2e 테스트 공용 fixture."""

from __future__ import annotations

from mdp.settings.schema import (
    Config,
    DataSpec,
    MetadataSpec,
    ModelSpec,
    Recipe,
    Settings,
    TrainingSpec,
)


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
        model=ModelSpec(class_path=model_class),
        data=DataSpec(source="/tmp/fake", label_strategy="causal"),
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
