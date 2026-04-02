"""공용 모델 가중치 로딩 테스트."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from mdp.serving.model_loader import load_checkpoint_weights
from tests.e2e.models import TinyVisionModel


def test_load_safetensors(tmp_path: Path) -> None:
    """model.safetensors에서 가중치를 로드."""
    from safetensors.torch import save_file

    model_orig = TinyVisionModel(num_classes=2, hidden_dim=16)
    save_file(model_orig.state_dict(), tmp_path / "model.safetensors")

    model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
    load_checkpoint_weights(model_new, tmp_path)

    for (n1, p1), (n2, p2) in zip(
        model_orig.named_parameters(), model_new.named_parameters()
    ):
        assert torch.equal(p1, p2), f"Parameter {n1} mismatch"


def test_load_model_pt_fallback(tmp_path: Path) -> None:
    """model.safetensors 없을 때 model.pt로 fallback."""
    model_orig = TinyVisionModel(num_classes=2, hidden_dim=16)
    torch.save(model_orig.state_dict(), tmp_path / "model.pt")

    model_new = TinyVisionModel(num_classes=2, hidden_dim=16)
    load_checkpoint_weights(model_new, tmp_path)

    for (_, p1), (_, p2) in zip(
        model_orig.named_parameters(), model_new.named_parameters()
    ):
        assert torch.equal(p1, p2)


def test_load_no_weights_warns(tmp_path: Path, caplog) -> None:
    """가중치 파일 없으면 warning."""
    model = TinyVisionModel(num_classes=2, hidden_dim=16)

    with caplog.at_level(logging.WARNING):
        load_checkpoint_weights(model, tmp_path)

    assert "가중치 파일이 없습니다" in caplog.text
