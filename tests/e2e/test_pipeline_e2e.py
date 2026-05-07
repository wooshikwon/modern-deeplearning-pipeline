"""파이프라인 통합 테스트: YAML → Settings → Factory → Trainer → 결과.

3 tests:
- test_yaml_to_training_e2e: 실제 YAML 파일에서 학습 완료까지 전체 경로
- test_init_generates_parseable_yaml: mdp init 생성 파일이 파싱 가능한지
- test_train_json_output_schema: TrainResult 스키마 필드가 결과에 존재하는지
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

_FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_yaml_to_training_e2e() -> None:
    """실제 YAML → SettingsFactory → Factory → Trainer → result 전체 흐름."""
    from mdp.cli._torchrun_entry import run_training
    from mdp.factory.factory import Factory
    from mdp.settings.factory import SettingsFactory
    from tests.e2e.datasets import ListDataLoader, make_vision_batches

    recipe_path = str(_FIXTURES / "recipes" / "tiny-vision-e2e.yaml")
    config_path = str(_FIXTURES / "configs" / "local-cpu.yaml")

    # YAML → Settings
    settings = SettingsFactory().for_training(recipe_path, config_path)
    assert settings.recipe.name == "tiny-vision-e2e"
    assert settings.recipe.task == "image_classification"

    # Factory → Model (TinyVisionModel은 init_args로 생성 가능)
    from mdp.factory.factory import Factory

    factory = Factory(settings)
    model = factory.create_model()
    assert model is not None

    # Trainer (데이터는 fake source라 수동 주입)
    from mdp.training.trainer import Trainer

    train_batches = make_vision_batches(5, 4, 2, 8)
    val_batches = make_vision_batches(2, 4, 2, 8, seed=99)
    trainer = Trainer(
        settings=settings,
        model=model,
        train_loader=ListDataLoader(train_batches),
        val_loader=ListDataLoader(val_batches),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    result = trainer.train()

    # 결과 검증
    assert "metrics" in result
    assert "total_epochs" in result
    assert "total_steps" in result
    assert "stopped_reason" in result
    assert result["total_epochs"] == 2
    assert result["stopped_reason"] == "completed"


def test_init_generates_parseable_yaml(tmp_path: Path) -> None:
    """mdp init으로 생성된 recipe+config가 SettingsFactory로 파싱 가능한지."""
    from mdp.cli.init import init_project
    from mdp.settings.factory import SettingsFactory

    project_dir = tmp_path / "test-project"
    init_project(str(project_dir))

    recipe_path = project_dir / "recipes" / "example.yaml"
    config_path = project_dir / "configs" / "local.yaml"

    assert recipe_path.exists(), f"Recipe not found: {recipe_path}"
    assert config_path.exists(), f"Config not found: {config_path}"

    # 파싱 성공 여부만 확인 (모델 로딩은 안 함)
    factory = SettingsFactory()
    # for_estimation은 검증을 건너뛰므로 recipe만으로 파싱 가능
    settings = factory.for_estimation(str(recipe_path))
    assert settings.recipe.name is not None
    assert settings.recipe.task is not None

    # S7 회귀 방지: init 템플릿은 component-unified schema를 따라야 한다.
    # model/dataset/collator가 모두 _component_ dict로 선언되어야 한다.
    assert "_component_" in settings.recipe.model, (
        "init 템플릿의 model 블록에 _component_ 키가 없다 — 구 class_path 회귀"
    )
    assert "_component_" in settings.recipe.data.dataset
    assert "_component_" in settings.recipe.data.collator

    assert "pretrained" not in settings.recipe.model, (
        "fallback init 템플릿은 torchvision 모델에 HF pretrained URI를 섞으면 안 된다"
    )


def test_catalog_init_recipes_are_parseable_for_all_supported_tasks() -> None:
    """catalog의 모든 supported_tasks 조합이 최신 Recipe 스키마로 파싱되어야 한다."""
    from mdp.cli.init import _build_recipe_from_catalog
    from mdp.settings.schema import Recipe

    catalog_dir = Path(__file__).resolve().parents[2] / "mdp" / "models" / "catalog"
    failures: list[str] = []

    for catalog_path in sorted(catalog_dir.rglob("*.yaml")):
        catalog = yaml.safe_load(catalog_path.read_text())
        if not catalog:
            continue
        for task in catalog.get("supported_tasks", []):
            recipe_yaml = _build_recipe_from_catalog(task, catalog, "agent-project")
            recipe_dict = yaml.safe_load(recipe_yaml)
            try:
                Recipe(**recipe_dict)
            except Exception as exc:  # pragma: no cover - assertion detail path
                failures.append(f"{catalog_path.relative_to(catalog_dir)}:{task}: {exc}")

    assert not failures, "\n".join(failures)


def test_catalog_init_recipes_use_task_specific_data_components() -> None:
    """mdp init은 text/vision/token/seq2seq task별 collator를 섞으면 안 된다."""
    from mdp.cli.init import _build_recipe_from_catalog

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

    catalog_dir = Path(__file__).resolve().parents[2] / "mdp" / "models" / "catalog"
    failures: list[str] = []

    for catalog_path in sorted(catalog_dir.rglob("*.yaml")):
        catalog = yaml.safe_load(catalog_path.read_text())
        if not catalog:
            continue
        for task in catalog.get("supported_tasks", []):
            recipe = yaml.safe_load(_build_recipe_from_catalog(task, catalog, "agent-project"))
            collator = recipe["data"]["collator"]["_component_"].rsplit(".", 1)[-1]
            expected = expected_collators.get(task)
            if expected is not None and collator != expected:
                failures.append(
                    f"{catalog['name']}:{task}: collator={collator}, expected={expected}"
                )

            dataset = recipe["data"]["dataset"]["_component_"].rsplit(".", 1)[-1]
            expected_dataset = expected_datasets.get(task)
            if expected_dataset is not None and dataset != expected_dataset:
                failures.append(
                    f"{catalog['name']}:{task}: dataset={dataset}, expected={expected_dataset}"
                )

    assert not failures, "\n".join(failures)


def test_train_json_output_schema() -> None:
    """TrainResult 스키마가 Trainer 결과와 호환되는지."""
    from mdp.cli.schemas import TrainResult
    from tests.e2e.conftest import make_test_settings
    from tests.e2e.datasets import ListDataLoader, make_vision_batches
    from tests.e2e.models import TinyVisionModel
    from mdp.training.trainer import Trainer

    settings = make_test_settings(epochs=1)
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    trainer = Trainer(
        settings=settings, model=model,
        train_loader=ListDataLoader(make_vision_batches(3, 4, 2, 8)),
    )
    trainer.device = torch.device("cpu")
    trainer.amp_enabled = False

    train_result = trainer.train()

    # TrainResult가 Trainer 반환값에서 정상 생성되는지
    result = TrainResult(
        checkpoint_dir=settings.config.storage.checkpoint_dir,
        output_dir=settings.config.storage.output_dir,
        metrics=train_result.get("metrics", {}),
        total_epochs=train_result.get("total_epochs"),
        total_steps=train_result.get("total_steps"),
        stopped_reason=train_result.get("stopped_reason"),
        duration_seconds=train_result.get("training_duration_seconds"),
        monitoring=train_result.get("monitoring"),
    )

    # 스키마 직렬화가 성공하고 핵심 필드가 존재
    dumped = result.model_dump(exclude_none=True)
    assert "total_epochs" in dumped
    assert "total_steps" in dumped
    assert "stopped_reason" in dumped
    assert dumped["total_epochs"] == 1
    assert dumped["stopped_reason"] == "completed"
