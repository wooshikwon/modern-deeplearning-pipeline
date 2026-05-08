"""AssemblyMaterializer public model/head materialization behavior tests."""

from __future__ import annotations

from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.planner import AssemblyPlanner
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.schema import Config, DataSpec, MetadataSpec, Recipe, Settings, TrainingSpec


def _settings_with_head(head_config: dict) -> Settings:
    return Settings(
        recipe=Recipe(
            name="materializer-head-e2e",
            task="image_classification",
            model={
                "_component_": "tests.e2e.models.TinyVisionModel",
                "num_classes": 2,
                "hidden_dim": 16,
            },
            head=head_config,
            data=DataSpec(
                dataset={"_component_": "tests.e2e.datasets.TinyVisionDataset"},
                collator={"_component_": "mdp.data.collators.VisionCollator"},
            ),
            training=TrainingSpec(epochs=1),
            metadata=MetadataSpec(author="test", description="materializer head materialization"),
        ),
        config=Config(),
    )


def _materializer(settings: Settings) -> AssemblyMaterializer:
    run_plan = RunPlan(
        command="train",
        mode="sft",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=(),
        validation_scope="training",
        distributed_intent=False,
    )
    return AssemblyMaterializer(AssemblyPlanner.from_run_plan(run_plan))


def test_materializer_materializes_head_at_target_attr() -> None:
    settings = _settings_with_head({
        "_component_": "ClassificationHead",
        "_target_attr": "head",
        "num_classes": 10,
        "hidden_dim": 16,
        "dropout": 0.0,
    })

    model = _materializer(settings).materialize_policy_model()

    assert type(model.head).__name__ == "ClassificationHead"
    assert model.head.classifier.out_features == 10


def test_materializer_materializes_head_at_existing_alternative_attr() -> None:
    settings = _settings_with_head({
        "_component_": "torch.nn.Linear",
        "_target_attr": "classifier",
        "in_features": 8,
        "out_features": 32,
    })

    model = _materializer(settings).materialize_policy_model()

    assert model.classifier.out_features == 32


def test_materializer_rejects_head_without_target_attr() -> None:
    settings = _settings_with_head({
        "_component_": "ClassificationHead",
        "num_classes": 10,
        "hidden_dim": 16,
        "dropout": 0.0,
    })

    try:
        _materializer(settings).materialize_policy_model()
    except ValueError as exc:
        assert "_target_attr" in str(exc)
    else:  # pragma: no cover - assertion failure path
        raise AssertionError("AssemblyMaterializer accepted a head config without _target_attr")
