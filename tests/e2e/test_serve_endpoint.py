"""서빙 endpoint 통합 테스트 — FastAPI TestClient로 /health, /predict 검증.

실제 모델 서버를 띄우지 않고 TestClient로 동기 테스트.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from mdp.models.base import BaseModel
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


def test_predict_handler_hf_style_forward() -> None:
    """PredictHandler calls HF-style modules with keyword batch arguments."""
    from mdp.serving.handlers import PredictHandler

    class _HFStyleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.proj = nn.Linear(4, 2)

        def forward(self, input_ids=None, attention_mask=None):
            hidden = self.embed(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1)
            return {"logits": self.proj(hidden[:, -1])}

    class _Tokenizer:
        def __call__(self, text, return_tensors=None, padding=True, truncation=True):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

    recipe = SimpleNamespace(task="text_classification")
    handler = PredictHandler(_HFStyleModel(), _Tokenizer(), None, recipe)

    result = asyncio.run(handler.handle({"text": "hello"}))

    assert "prediction" in result
    assert "probabilities" in result


def test_predict_handler_basemodel_style_forward() -> None:
    """PredictHandler keeps BaseModel modules on the ``model(batch)`` contract."""
    from mdp.serving.handlers import PredictHandler

    class _BaseStyleModel(BaseModel):
        _block_classes = None

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 2)

        def forward(self, batch: dict) -> dict[str, torch.Tensor]:
            return {"logits": self.proj(batch["features"])}

        def validation_step(self, batch: dict) -> dict[str, float]:
            return {"val_loss": 0.0}

    recipe = SimpleNamespace(task="feature_extraction")
    handler = PredictHandler(_BaseStyleModel(), None, None, recipe)

    result = asyncio.run(handler.handle({"features": torch.ones(1, 3)}))

    assert "prediction" in result
    assert "probabilities" in result


def test_parse_request_applies_fields_to_json_body() -> None:
    """JSON requests use the same role/column routing as multipart requests."""
    from mdp.serving.server import _parse_request

    class _Request:
        headers = {"content-type": "application/json"}

        async def json(self):
            return {"review": "great"}

    result = asyncio.run(_parse_request(_Request(), {"text": "review"}, "text_classification"))

    assert result["text"] == "great"


def test_parse_request_keeps_canonical_json_key() -> None:
    """Canonical role key wins when both role and source column are present."""
    from mdp.serving.server import _parse_request

    class _Request:
        headers = {"content-type": "application/json"}

        async def json(self):
            return {"text": "canonical", "review": "source"}

    result = asyncio.run(_parse_request(_Request(), {"text": "review"}, "text_classification"))

    assert result["text"] == "canonical"


def test_serving_runtime_options_cli_overrides_config() -> None:
    from mdp.cli.serve import _resolve_serving_runtime_options
    from mdp.settings.schema import ServingConfig

    serving_config = ServingConfig(
        device_map="auto",
        max_memory={"0": "24GiB"},
    )

    device_map, max_memory = _resolve_serving_runtime_options(
        serving_config,
        device_map="balanced",
        max_memory='{"0": "12GiB"}',
    )

    assert device_map == "balanced"
    assert max_memory == {"0": "12GiB"}


def test_serving_runtime_options_falls_back_to_config() -> None:
    from mdp.cli.serve import _resolve_serving_runtime_options
    from mdp.settings.schema import ServingConfig

    serving_config = ServingConfig(
        device_map="auto",
        max_memory={"0": "24GiB"},
    )

    device_map, max_memory = _resolve_serving_runtime_options(
        serving_config,
        device_map=None,
        max_memory=None,
    )

    assert device_map == "auto"
    assert max_memory == {"0": "24GiB"}
