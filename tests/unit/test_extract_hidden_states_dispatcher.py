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


# ────────────────────────────────────────────────────────────── #
#  Priority 2 — PEFT 래핑 투과 (review 1-1)                      #
# ────────────────────────────────────────────────────────────── #
#
#  PeftModel은 HF PreTrainedModel을 상속하지 않기 때문에 `isinstance(PreTrainedModel)`
#  직접 검사만으로는 dispatcher 경로가 설정되지 않는다. 실제 LoRA 경로에서는
#  `peft_model.base_model.model`이 진짜 PreTrainedModel이므로 dispatcher가 2단계
#  duck-typing 언래핑으로 내려가야 한다. 실제 `peft` 패키지의 PeftModel은 GPU
#  환경에서만 검증 가능하지만, dispatcher의 판정 로직은 `getattr(base_model,
#  "model")`로만 구현되어 있으므로 동일 shape의 minimal stub으로도 충분히 회귀
#  고정 가능하다.


class _FakePeftModel(nn.Module):
    """PeftModel.base_model.model 체인을 흉내 내는 최소 스텁.

    ``isinstance(self, PreTrainedModel)``은 False지만 ``base_model.model``은
    진짜 PreTrainedModel이므로 dispatcher가 Priority 2의 PEFT 분기로 내려가야
    한다. 실제 LoRA처럼 parameter grafting을 하지는 않지만, dispatcher가
    "PEFT 래핑을 인식하고 투과하여 내부 HF 모델의 `_extract_hf_pretrained`로
    라우팅하는가"만 격리 검증한다.
    """

    def __init__(self, inner_hf: nn.Module) -> None:
        super().__init__()
        # 실제 PeftModel 구조: self.base_model = LoraModel, base_model.model = PreTrainedModel
        self.base_model = SimpleNamespace(model=inner_hf)


def test_dispatcher_priority_2_peft_wrapping_routes_to_inner_hf_model() -> None:
    """PEFT 스타일 래핑(`base_model.model` 체인)은 내부 HF 모델로 dispatch된다.

    review-2026-04-18-U6-c1 §1-1에서 확정한 회귀 고정. 이 가드가 빠지면
    weighted-ntp Phase 3 Baseline(LoRA+HF Llama) 전체가
    ``NotImplementedError: Model PeftModel...``로 실패한다.
    """
    inner, cfg = _build_tiny_hf_model()
    wrapped = _FakePeftModel(inner)
    # PeftModel은 PreTrainedModel을 상속하지 않는다는 전제 확인.
    from transformers import PreTrainedModel

    assert not isinstance(wrapped, PreTrainedModel)

    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }

    hidden, head_weight = _call_dispatcher(wrapped, batch)

    # inner의 HF 경로 결과와 동일해야 함.
    assert hidden.shape == (2, 5, cfg.n_embd)
    assert head_weight.shape == (cfg.vocab_size, cfg.n_embd)
    # head_weight은 inner.get_output_embeddings().weight을 참조 (tied-weights)
    assert head_weight.data_ptr() == inner.get_output_embeddings().weight.data_ptr()


def test_dispatcher_priority_2_peft_wrapping_inside_ddp_wrapper() -> None:
    """DDP(`.module`) + PEFT(`base_model.model`) 2중 래핑도 정상 투과한다.

    DDP 언래핑(`getattr(model, "module", model)`)이 먼저 적용된 후 PEFT 분기가
    동작해야 한다. 실서버 4-rank 학습의 전형적 조합.
    """
    inner, cfg = _build_tiny_hf_model()
    peft_wrapped = _FakePeftModel(inner)
    ddp_wrapped = _FakeWrapper(peft_wrapped)

    batch = {"input_ids": torch.randint(0, cfg.vocab_size, (1, 4))}
    hidden, head_weight = _call_dispatcher(ddp_wrapped, batch)

    assert hidden.shape == (1, 4, cfg.n_embd)
    assert head_weight.shape == (cfg.vocab_size, cfg.n_embd)


class _NonPeftDuckStub(nn.Module):
    """`base_model.model`을 가지지만 내부가 PreTrainedModel이 아닌 경우.

    PEFT 분기가 `isinstance(peft_inner, PreTrainedModel)` 재검사 없이 무조건
    dispatch하면 Priority 3/4 경로가 죽는다. 비-HF duck-type이 우연히 같은
    속성 체인을 가져도 Priority 5(NotImplementedError)로 떨어져야 한다.
    """

    def __init__(self) -> None:
        super().__init__()
        self.base_model = SimpleNamespace(model=_PlainModule())


def test_dispatcher_peft_chain_requires_hf_inner() -> None:
    """`base_model.model`이 있어도 그 내부가 PreTrainedModel이 아니면 HF로
    라우팅하지 않고 이후 Priority로 fall-through 한다."""
    stub = _NonPeftDuckStub()
    with pytest.raises(NotImplementedError) as exc:
        _call_dispatcher(stub, {"input_ids": torch.tensor([[1]])})
    # PlainModule이 최종 miss 대상이어야 한다 (내부가 아닌 wrapper 이름도 허용).
    assert "기본 구현이 없습니다" in str(exc.value)


# ────────────────────────────────────────────────────────────── #
#  Dispatcher 반환 계약 — (hidden, head_weight) 동일 device        #
#  (review 1-2)                                                   #
# ────────────────────────────────────────────────────────────── #
#
#  Liger-Kernel FLCE는 Triton 커널이라 모든 인자가 동일 GPU device에 있어야
#  하는데, PEFT 래핑이나 tied-weights 경로에서 `lm_head.weight`이 CPU에 남는
#  케이스가 실서버에서 관찰됐다(review-2026-04-18-U6-c1 §1-2 v4·v5 실패).
#  dispatcher가 반환 직전에 `head_weight.to(hidden.device)`를 강제하므로,
#  hidden이 어느 device에 있든 head_weight 역시 같은 device로 따라가야 한다.


class _HeadOnCpuHFModel(nn.Module):
    """PreTrainedModel-like 껍데기 — ``get_output_embeddings().weight``이 CPU에
    남아 있는 상황을 모사한다.

    실제 HF 모델 forward를 흉내 내되 ``lm_head.weight``은 CPU에 고정하고
    forward hidden_states 튜플은 caller가 지정한 device로 반환한다.
    """

    def __init__(self, vocab: int = 17, hidden: int = 8, hidden_device: str = "cpu") -> None:
        super().__init__()
        self._vocab = vocab
        self._hidden = hidden
        self._hidden_device = torch.device(hidden_device)
        # head_weight은 항상 CPU에 두어 device mismatch를 유발.
        self._head_weight = torch.randn(vocab, hidden, device="cpu")

    # transformers.PreTrainedModel isinstance가 되도록 실제 클래스에 동적 등록 —
    # 아래 fixture에서 monkeypatch.
    def get_output_embeddings(self):
        return SimpleNamespace(weight=self._head_weight)

    def forward(self, *, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        hidden = torch.randn(B, S, self._hidden, device=self._hidden_device)
        # HF ModelOutput-like 튜플: hidden_states는 layer+1 tuple. 마지막만 필요.
        return SimpleNamespace(hidden_states=(hidden,), logits=None)


def test_extract_hf_pretrained_returns_same_device_tensors() -> None:
    """``_extract_hf_pretrained``가 ``(hidden, head_weight)``을 같은 device로
    정렬해 반환한다(review 1-2 계약).

    hidden은 ``meta`` device에, ``get_output_embeddings().weight``은 CPU에
    두고 dispatcher 하위 헬퍼를 직접 호출한다. PeftModel 래핑이나 tied-weights
    경로에서 관찰된 device mismatch(Liger FLCE Triton kernel 실패 원인)가
    dispatcher 반환 직전에 `.to(hidden.device)`로 흡수되는지를 고정 검증.
    """
    # `_extract_hf_pretrained`만 격리 호출하므로 isinstance 통과용 상속은 필요 없다.
    model = _HeadOnCpuHFModel(vocab=17, hidden=8, hidden_device="meta")
    batch = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

    trainer_stub = SimpleNamespace()
    hidden, head_weight = RLTrainer._extract_hf_pretrained(
        trainer_stub, model, batch, layer_idx=-1
    )
    # 핵심 계약: 동일 device.
    assert hidden.device == head_weight.device, (
        f"dispatcher가 device 일관성 계약을 어겼다: hidden={hidden.device}, "
        f"head_weight={head_weight.device}"
    )
    assert hidden.device.type == "meta"


def test_extract_hf_pretrained_noop_when_already_same_device() -> None:
    """이미 같은 device면 .to() 호출이 identity여야 한다 (ptr 유지).

    GPT-2 path는 tied embedding이라 head_weight.data_ptr()이 wte.weight과
    동일하다. 기본 CPU 환경에서 수행되므로 추가 복사 없이 같은 tensor가
    반환되어야 한다(기존 성공 경로 보존).
    """
    model, cfg = _build_tiny_hf_model()
    batch = {"input_ids": torch.randint(0, cfg.vocab_size, (1, 3))}
    hidden, head_weight = _call_dispatcher(model, batch)

    assert hidden.device == head_weight.device
    assert head_weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()


def test_extract_torchvision_resnet_returns_same_device_tensors() -> None:
    """torchvision ResNet 경로도 dispatcher 반환 계약을 준수한다 (대칭 확인)."""
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    batch = {"pixel_values": torch.randn(1, 3, 224, 224)}
    hidden, head_weight = _call_dispatcher(model, batch)
    assert hidden.device == head_weight.device


def test_extract_timm_returns_same_device_tensors() -> None:
    """timm 경로도 dispatcher 반환 계약을 준수한다 (대칭 확인)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    model.eval()
    batch = {"pixel_values": torch.randn(1, 3, 224, 224)}
    hidden, head_weight = _call_dispatcher(model, batch)
    assert hidden.device == head_weight.device
