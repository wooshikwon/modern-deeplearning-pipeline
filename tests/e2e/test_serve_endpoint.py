"""서빙 endpoint 통합 테스트 — FastAPI TestClient로 /health, /predict 검증.

실제 모델 서버를 띄우지 않고 TestClient로 동기 테스트.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import yaml

from tests.e2e.models import TinyVisionModel


def _create_model_artifact(tmp_path: Path) -> Path:
    """테스트용 model artifact 디렉토리를 생성한다."""
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()

    # 모델 가중치 저장
    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    from safetensors.torch import save_file

    save_file(model.state_dict(), str(artifact_dir / "model.safetensors"))

    # recipe.yaml 저장
    recipe = {
        "name": "serve-test",
        "task": "image_classification",
        "model": {
            "_component_": "tests.e2e.models.TinyVisionModel",
            "num_classes": 2,
            "hidden_dim": 16,
        },
        "data": {
            "dataset": {"_component_": "mdp.data.datasets.HuggingFaceDataset", "source": "/tmp/fake", "split": "train"},
            "collator": {"_component_": "mdp.data.collators.ClassificationCollator", "tokenizer": "gpt2"},
        },
        "training": {"epochs": 1},
        "optimizer": {"_component_": "AdamW", "lr": 1e-3},
        "metadata": {"author": "test", "description": "serve test"},
    }
    (artifact_dir / "recipe.yaml").write_text(yaml.dump(recipe))

    return artifact_dir


def test_health_endpoint(tmp_path: Path) -> None:
    """/health 엔드포인트가 200과 모델 정보를 반환하는지."""
    from mdp.serving.server import create_app
    from mdp.serving.model_loader import reconstruct_model

    artifact_dir = _create_model_artifact(tmp_path)
    model, settings = reconstruct_model(artifact_dir)
    model.eval()

    from mdp.serving.handlers import PredictHandler

    handler = PredictHandler(model, None, None, settings.recipe)
    app = create_app(handler, settings.recipe)

    from starlette.testclient import TestClient

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["task"] == "image_classification"


def test_predict_endpoint_classification(tmp_path: Path) -> None:
    """/predict가 classification 입력을 처리하고 결과를 반환하는지.

    PredictHandler는 batch_loop(asyncio task)으로 비동기 처리하므로,
    직접 _preprocess → model forward를 동기적으로 검증한다.
    """
    from mdp.serving.model_loader import reconstruct_model

    artifact_dir = _create_model_artifact(tmp_path)
    model, settings = reconstruct_model(artifact_dir)
    model.eval()

    from mdp.serving.handlers import PredictHandler

    handler = PredictHandler(model, None, None, settings.recipe)

    # _preprocess가 transform/tokenizer 없이 raw dict를 그대로 반환하는지
    raw_input = {"pixel_values": torch.randn(1, 3, 8, 8)}
    preprocessed = handler._preprocess(raw_input)
    assert "pixel_values" in preprocessed

    # 모델이 preprocessed 입력으로 forward 가능한지
    with torch.no_grad():
        outputs = model(preprocessed)
    assert "logits" in outputs
    assert outputs["logits"].shape[1] == 2  # num_classes
