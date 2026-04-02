"""추론 전체 흐름 테스트 — artifact에서 모델 재구성 → 배치 추론.

MLflow 없이 artifact 디렉토리를 직접 구성하여 테스트.
검증: recipe.yaml → 모델 재구성 → 가중치 로딩 → forward → 결과.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from tests.e2e.datasets import ListDataLoader, make_vision_batches
from tests.e2e.models import TinyVisionModel


def _create_artifact(tmp_path: Path) -> Path:
    """학습 완료 상태를 시뮬레이션하는 artifact 디렉토리."""
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    # 학습된 모델 저장
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    # 몇 step 학습 시뮬레이션
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for batch in make_vision_batches(3, 4, 2, 8):
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    from safetensors.torch import save_file
    save_file(model.state_dict(), str(artifact_dir / "model.safetensors"))

    # recipe.yaml
    recipe = {
        "name": "inference-test",
        "task": "image_classification",
        "model": {
            "class_path": "tests.e2e.models.TinyVisionModel",
            "init_args": {"num_classes": 2, "hidden_dim": 16},
        },
        "data": {"source": "/tmp/fake", "fields": {"image": "pixel_values", "label": "labels"}},
        "training": {"epochs": 1},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3},
        "metadata": {"author": "test", "description": "inference test"},
    }
    (artifact_dir / "recipe.yaml").write_text(yaml.dump(recipe))

    return artifact_dir


def test_reconstruct_model_from_artifact(tmp_path: Path) -> None:
    """artifact 디렉토리에서 모델 재구성 + 가중치 로딩."""
    from mdp.serving.model_loader import reconstruct_model

    artifact_dir = _create_artifact(tmp_path)
    model, settings = reconstruct_model(artifact_dir)

    assert settings.recipe.name == "inference-test"
    assert settings.recipe.task == "image_classification"

    # 로딩된 모델이 forward 가능한지
    model.eval()
    batch = {"pixel_values": torch.randn(2, 3, 8, 8)}
    with torch.no_grad():
        outputs = model(batch)
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 2)


def test_artifact_to_batch_inference(tmp_path: Path) -> None:
    """artifact → 모델 재구성 → 배치 추론 → 예측 결과."""
    from mdp.serving.model_loader import reconstruct_model

    artifact_dir = _create_artifact(tmp_path)
    model, settings = reconstruct_model(artifact_dir)
    model.eval()

    # 추론용 데이터
    test_batches = make_vision_batches(3, 4, 2, 8, seed=42)
    test_loader = ListDataLoader(test_batches)

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.tolist())

    # 12 samples (3 batches × 4)
    assert len(all_preds) == 12
    assert all(p in (0, 1) for p in all_preds)


def test_settings_factory_from_artifact(tmp_path: Path) -> None:
    """SettingsFactory.from_artifact()가 recipe.yaml에서 Settings를 복원하는지."""
    from mdp.settings.factory import SettingsFactory

    artifact_dir = _create_artifact(tmp_path)
    settings = SettingsFactory().from_artifact(str(artifact_dir))

    assert settings.recipe.name == "inference-test"
    assert settings.recipe.model.class_path == "tests.e2e.models.TinyVisionModel"
