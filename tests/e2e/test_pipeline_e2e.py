"""파이프라인 통합 테스트: YAML → Settings → Factory → Trainer → 결과.

3 tests:
- test_yaml_to_training_e2e: 실제 YAML 파일에서 학습 완료까지 전체 경로
- test_init_generates_parseable_yaml: mdp init 생성 파일이 파싱 가능한지
- test_train_json_output_schema: TrainResult 스키마 필드가 결과에 존재하는지
"""

from __future__ import annotations

from pathlib import Path

import torch

_FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_yaml_to_training_e2e() -> None:
    """실제 YAML → SettingsFactory → Factory → Trainer → result 전체 흐름."""
    from mdp.cli._torchrun_entry import run_training
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
