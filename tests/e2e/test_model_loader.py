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


# ---------------------------------------------------------------------------
# device_map 관련 테스트
# ---------------------------------------------------------------------------


def test_dispatch_model_with_accelerate(tmp_path: Path) -> None:
    """_dispatch_model이 accelerate로 모델을 분산 배치한다 (CPU 환경에서도 동작)."""
    from safetensors.torch import save_file

    from mdp.serving.model_loader import _dispatch_model

    model = TinyVisionModel(num_classes=2, hidden_dim=16)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    dispatched = _dispatch_model(
        TinyVisionModel(num_classes=2, hidden_dim=16),
        str(tmp_path / "model.safetensors"),
        device_map="auto",
    )
    assert hasattr(dispatched, "hf_device_map")


def test_find_checkpoint_path_safetensors(tmp_path: Path) -> None:
    """safetensors 파일을 우선 반환한다."""
    from mdp.serving.model_loader import _find_checkpoint_path

    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.pt").touch()
    assert _find_checkpoint_path(tmp_path) == str(tmp_path / "model.safetensors")


def test_find_checkpoint_path_pt_fallback(tmp_path: Path) -> None:
    """safetensors 없으면 model.pt로 fallback."""
    from mdp.serving.model_loader import _find_checkpoint_path

    (tmp_path / "model.pt").touch()
    assert _find_checkpoint_path(tmp_path) == str(tmp_path / "model.pt")


def test_find_checkpoint_path_none(tmp_path: Path) -> None:
    """가중치 파일 없으면 None."""
    from mdp.serving.model_loader import _find_checkpoint_path

    assert _find_checkpoint_path(tmp_path) is None


def test_serving_config_device_map_fields() -> None:
    """ServingConfig에 device_map, max_memory 필드가 존재한다."""
    from mdp.settings.schema import ServingConfig

    config = ServingConfig(device_map="auto", max_memory={"0": "24GiB"})
    assert config.device_map == "auto"
    assert config.max_memory == {"0": "24GiB"}

    default = ServingConfig()
    assert default.device_map is None
    assert default.max_memory is None
