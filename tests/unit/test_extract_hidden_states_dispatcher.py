"""RLTrainer framework-agnostic hidden-states dispatcher 테스트.

spec-algorithm-hidden-states-support U1 의 핵심 검증:
  1. BaseModel.extract_features_and_head 기본 NotImplementedError 계약.
  2. RLTrainer._extract_hidden_states_and_head 4단 dispatcher:
      (1) 모델 override → (2) HF PreTrainedModel → (3) timm →
      (4) torchvision ResNet → (5) NotImplementedError.
  3. 실제 프레임워크 기본 구현의 shape 계약.

여기서는 RLTrainer 전체를 띄우지 않고, 메서드를 언바운드로 호출하여
dispatcher 로직만 격리 검증한다. Trainer 생성은 무거운 의존성
(Settings/Recipe/dataloader)이 필요하므로 단위 테스트에 부적합.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from mdp.models.base import BaseModel
from mdp.training.rl_trainer import RLTrainer


# ────────────────────────────────────────────────────────────── #
#  BaseModel.extract_features_and_head 계약                     #
# ────────────────────────────────────────────────────────────── #


class _DummyBaseModel(BaseModel):
    """NotImplementedError fallthrough 테스트용 BaseModel 서브클래스."""

    _block_classes = None

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # pragma: no cover
        return {}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:  # pragma: no cover
        return torch.tensor(0.0)

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:  # pragma: no cover
        return {}


def test_basemodel_extract_features_default_raises() -> None:
    """BaseModel.extract_features_and_head 기본 구현은 override 안내 메시지와
    함께 NotImplementedError를 던진다."""
    model = _DummyBaseModel()
    with pytest.raises(NotImplementedError, match="override하지 않았습니다"):
        model.extract_features_and_head({"input_ids": torch.tensor([[1, 2, 3]])})


# ────────────────────────────────────────────────────────────── #
#  Dispatcher priority 1 — 모델의 override 우선                  #
# ────────────────────────────────────────────────────────────── #


class _CustomModelWithOverride(BaseModel):
    """extract_features_and_head를 override한 BaseModel 서브클래스."""

    _block_classes = None

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 16)
        self.called_with: dict[str, Any] = {}

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # pragma: no cover
        return {}

    def training_step(self, batch: dict[str, Tensor]) -> Tensor:  # pragma: no cover
        return torch.tensor(0.0)

    def validation_step(self, batch: dict[str, Tensor]) -> dict[str, float]:  # pragma: no cover
        return {}

    def extract_features_and_head(
        self, batch: dict[str, Tensor], layer_idx: int = -1
    ) -> tuple[Tensor, Tensor]:
        self.called_with = {"batch_keys": sorted(batch.keys()), "layer_idx": layer_idx}
        B = batch["input_ids"].shape[0] if "input_ids" in batch else 2
        hidden = torch.zeros(B, 4, 16)
        head_weight = torch.zeros(100, 16)
        return hidden, head_weight


def _call_dispatcher(model: nn.Module, batch: dict, layer_idx: int = -1) -> tuple[Tensor, Tensor]:
    """RLTrainer.__init__을 우회해 dispatcher만 호출."""
    trainer_stub = SimpleNamespace(
        _extract_hf_pretrained=RLTrainer._extract_hf_pretrained.__get__(SimpleNamespace()),
        _extract_timm=RLTrainer._extract_timm.__get__(SimpleNamespace()),
        _extract_torchvision_resnet=RLTrainer._extract_torchvision_resnet.__get__(
            SimpleNamespace()
        ),
    )
    return RLTrainer._extract_hidden_states_and_head(
        trainer_stub, model, batch, layer_idx=layer_idx
    )


def test_dispatcher_priority_1_custom_override() -> None:
    """모델이 extract_features_and_head를 가지면 최우선으로 호출된다."""
    model = _CustomModelWithOverride()
    batch = {"input_ids": torch.arange(6).view(2, 3)}

    hidden, head_weight = _call_dispatcher(model, batch, layer_idx=-2)

    assert hidden.shape == (2, 4, 16)
    assert head_weight.shape == (100, 16)
    assert model.called_with == {"batch_keys": ["input_ids"], "layer_idx": -2}


def test_dispatcher_priority_1_override_raising_notimplemented_falls_through() -> None:
    """모델의 override가 NotImplementedError를 던지면 dispatcher가 fallback으로 이동.

    BaseModel의 기본 구현이 NotImplementedError이므로 이 경로가 실제로 동작해야
    한다(BaseModel subclass가 override를 하지 않으면 dispatcher가 framework 경로를
    찾는다). _DummyBaseModel은 HF/timm/torchvision이 아니므로 최종적으로 priority 5
    에서 다시 NotImplementedError.
    """
    model = _DummyBaseModel()
    with pytest.raises(NotImplementedError, match="기본 구현이 없습니다"):
        _call_dispatcher(model, {"input_ids": torch.tensor([[1]])})


# ────────────────────────────────────────────────────────────── #
#  Dispatcher priority 2 — HF PreTrainedModel                    #
# ────────────────────────────────────────────────────────────── #


def _build_tiny_hf_model():
    """GPT2-like tiny model (v4 매개변수 최소화)."""
    transformers = pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=128,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        n_ctx=32,
    )
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model, cfg


def test_dispatcher_priority_2_hf_shapes() -> None:
    """HF PreTrainedModel 경로: hidden=(B,S,H) + head_weight=(V,H)."""
    model, cfg = _build_tiny_hf_model()
    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    hidden, head_weight = _call_dispatcher(model, batch)

    assert hidden.shape == (2, 5, cfg.n_embd)
    assert head_weight.shape == (cfg.vocab_size, cfg.n_embd)
    # HF GPT-2는 lm_head.weight이 wte.weight과 tied되어 있음
    assert head_weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def test_dispatcher_priority_2_hf_layer_idx() -> None:
    """layer_idx 인자가 hidden_states tuple 인덱싱에 반영된다."""
    model, cfg = _build_tiny_hf_model()
    batch = {"input_ids": torch.randint(0, cfg.vocab_size, (1, 4))}

    hidden_last, _ = _call_dispatcher(model, batch, layer_idx=-1)
    hidden_first, _ = _call_dispatcher(model, batch, layer_idx=0)

    # 둘 다 (1, 4, H) shape이지만 내용은 달라야 함 (embedding vs 마지막 block)
    assert hidden_last.shape == hidden_first.shape == (1, 4, cfg.n_embd)
    assert not torch.allclose(hidden_last, hidden_first)


# ────────────────────────────────────────────────────────────── #
#  Dispatcher priority 3 — timm                                  #
# ────────────────────────────────────────────────────────────── #


def test_dispatcher_priority_3_timm_shapes() -> None:
    """timm 모델 경로: forward_features + get_classifier().weight."""
    timm = pytest.importorskip("timm")

    # ResNet18 (CNN): (B,3,224,224) → feature (B,C,H,W), head Linear
    model = timm.create_model("resnet18", pretrained=False, num_classes=10)
    model.eval()
    batch = {"pixel_values": torch.randn(2, 3, 224, 224)}

    hidden, head_weight = _call_dispatcher(model, batch)

    # timm resnet18: forward_features → (B, 512, 7, 7), classifier Linear(512, 10)
    assert hidden.ndim >= 2
    assert hidden.shape[0] == 2
    assert head_weight.shape == (10, 512)


def test_dispatcher_priority_3_timm_identity_classifier_raises() -> None:
    """timm 모델의 classifier가 Identity(num_classes=0)인 경우 NotImplementedError."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False, num_classes=0)
    model.eval()
    batch = {"pixel_values": torch.randn(1, 3, 224, 224)}

    with pytest.raises(NotImplementedError, match="Identity"):
        _call_dispatcher(model, batch)


# ────────────────────────────────────────────────────────────── #
#  Dispatcher priority 4 — torchvision ResNet                    #
# ────────────────────────────────────────────────────────────── #


def test_dispatcher_priority_4_torchvision_resnet_shapes() -> None:
    """torchvision ResNet 경로: manual forward + model.fc.weight."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    batch = {"pixel_values": torch.randn(2, 3, 224, 224)}

    hidden, head_weight = _call_dispatcher(model, batch)

    # avgpool 후 flatten → (B, 512), fc.weight → (1000, 512)
    assert hidden.shape == (2, 512)
    assert head_weight.shape == (1000, 512)
    assert head_weight.data_ptr() == model.fc.weight.data_ptr()


# ────────────────────────────────────────────────────────────── #
#  Dispatcher priority 5 — unsupported                           #
# ────────────────────────────────────────────────────────────── #


class _PlainModule(nn.Module):
    """BaseModel/HF/timm/torchvision 중 아무것도 아닌 모듈."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        return self.linear(x)


def test_dispatcher_priority_5_unsupported_raises() -> None:
    """어떤 framework에도 해당하지 않는 모델은 안내 메시지와 함께 NotImplementedError."""
    model = _PlainModule()
    with pytest.raises(NotImplementedError) as exc:
        _call_dispatcher(model, {"input_ids": torch.tensor([[1]])})
    msg = str(exc.value)
    assert "_PlainModule" in msg
    assert "BaseModel" in msg
    # HF/timm/torchvision 3종 언급으로 사용자 안내
    assert "PreTrainedModel" in msg or "timm" in msg or "torchvision" in msg


# ────────────────────────────────────────────────────────────── #
#  DDP/FSDP wrapper unwrapping                                   #
# ────────────────────────────────────────────────────────────── #


class _FakeWrapper(nn.Module):
    """DDP처럼 `.module`에 실제 모델을 보관하는 wrapper."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner


def test_dispatcher_unwraps_ddp_style_wrapper() -> None:
    """.module 래핑된 모델도 내부 모델 기준으로 dispatch한다."""
    inner = _CustomModelWithOverride()
    wrapped = _FakeWrapper(inner)
    batch = {"input_ids": torch.arange(4).view(1, 4)}

    hidden, head_weight = _call_dispatcher(wrapped, batch)

    assert hidden.shape == (1, 4, 16)
    assert head_weight.shape == (100, 16)
    assert inner.called_with["batch_keys"] == ["input_ids"]
