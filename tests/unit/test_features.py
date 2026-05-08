"""mdp.training._features 모듈의 free function 단위 테스트.

spec-training-restructure U1 테스트 요구:
  "free function 호출이 trainer 인스턴스 없이 동작하는지 직접 검증"

이 파일은 _features.py의 퍼블릭·프라이빗 free function을 RLTrainer 없이 직접 호출하여
검증한다. dispatcher 분기 로직의 전수 회귀는 기존
``tests/unit/test_extract_hidden_states_dispatcher.py``가 담당하므로, 이 파일은
"free function 시그니처·동작이 trainer 인스턴스 없이 독립적으로 사용 가능한가"에
집중한다.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from mdp.training._features import (
    _extract_hf_pretrained,
    _extract_timm,
    _extract_torchvision_resnet,
    extract_hidden_states_and_head,
    extract_logits,
    forward_model,
)


# ────────────────────────────────────────────────────────────── #
#  extract_logits                                                #
# ────────────────────────────────────────────────────────────── #


def test_extract_logits_from_hf_model_output() -> None:
    """HF ModelOutput 형태 (logits 속성 보유) — .logits 반환."""
    logits = torch.randn(2, 5, 32)
    out = SimpleNamespace(logits=logits)
    result = extract_logits(out)
    assert result is logits


def test_extract_logits_from_dict_logits_key() -> None:
    """dict 출력의 'logits' 키 반환."""
    logits = torch.randn(1, 3, 16)
    result = extract_logits({"logits": logits})
    assert result is logits


def test_extract_logits_from_dict_output_key_fallback() -> None:
    """dict에 'logits' 없으면 'output' fallback."""
    out = torch.randn(2, 8)
    result = extract_logits({"output": out})
    assert result is out


def test_extract_logits_passthrough_tensor() -> None:
    """dict도 아니고 logits 속성도 없는 경우 그대로 반환."""
    raw = torch.randn(4, 4)
    result = extract_logits(raw)
    assert result is raw


# ────────────────────────────────────────────────────────────── #
#  forward_model                                                 #
# ────────────────────────────────────────────────────────────── #


class _TinyLinearModel(nn.Module):
    """input_ids를 받아 (B, S, vocab) logits를 반환하는 최소 모델."""

    def __init__(self, vocab: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, 8)
        self.proj = nn.Linear(8, vocab)
        self.vocab = vocab

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> dict:
        h = self.embed(input_ids)
        logits = self.proj(h)
        return {"logits": logits}


def test_forward_model_policy_role_returns_logits() -> None:
    """role='policy' → {"logits": ...}."""
    model = _TinyLinearModel()
    batch = {"input_ids": torch.randint(0, 32, (2, 4))}
    out = forward_model(model, batch, role="policy")
    assert "logits" in out
    assert out["logits"].shape == (2, 4, 32)


def test_forward_model_reference_role_same_as_policy() -> None:
    """role='reference' → {"logits": ...} (policy와 동일 구조)."""
    model = _TinyLinearModel()
    batch = {"input_ids": torch.randint(0, 32, (1, 3))}
    out = forward_model(model, batch, role="reference")
    assert "logits" in out


def test_forward_model_value_role_returns_values() -> None:
    """role='value' + 3D logits (B, S, 1) → {"values": (B, S)}."""

    class _ValueModel(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            B, S = input_ids.shape
            return {"logits": torch.zeros(B, S, 1)}

    batch = {"input_ids": torch.randint(0, 4, (2, 5))}
    out = forward_model(_ValueModel(), batch, role="value")
    assert "values" in out
    assert out["values"].shape == (2, 5)


def test_forward_model_reward_role_with_explicit_reward_key() -> None:
    """role='reward' + 모델 출력에 'reward' 키 → out에 "reward" + "logits" 모두."""

    class _RewardModel(nn.Module):
        def forward(self, input_ids, attention_mask=None):
            B = input_ids.shape[0]
            return {"reward": torch.zeros(B), "logits": torch.zeros(B, 1)}

    batch = {"input_ids": torch.randint(0, 4, (3, 2))}
    out = forward_model(_RewardModel(), batch, role="reward")
    assert "reward" in out
    assert "logits" in out


def test_forward_model_empty_batch_returns_empty_dict() -> None:
    """input_ids 없는 batch → 빈 dict (pixel_values만 있는 경우 등)."""
    model = _TinyLinearModel()
    out = forward_model(model, {}, role="policy")
    assert out == {}


def test_forward_model_no_trainer_instance_needed() -> None:
    """forward_model은 trainer 인스턴스 없이 직접 호출 가능."""
    model = _TinyLinearModel(vocab=16)
    batch = {"input_ids": torch.randint(0, 16, (2, 3))}
    # RLTrainer 없이 직접 free function 호출
    result = forward_model(model, batch)
    assert "logits" in result
    assert result["logits"].shape == (2, 3, 16)


# ────────────────────────────────────────────────────────────── #
#  extract_hidden_states_and_head — trainer 없이 호출 가능 확인  #
# ────────────────────────────────────────────────────────────── #


def test_extract_hidden_states_and_head_no_trainer_hf() -> None:
    """HF 경로: trainer 인스턴스 없이 extract_hidden_states_and_head 직접 호출."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=2, n_positions=32, n_ctx=32)
    model = GPT2LMHeadModel(cfg)
    model.eval()

    batch = {"input_ids": torch.randint(0, 64, (2, 5))}

    # RLTrainer 없이 free function 직접 호출
    hidden, head_weight = extract_hidden_states_and_head(model, batch)

    assert hidden.shape == (2, 5, cfg.n_embd)
    assert head_weight.shape == (cfg.vocab_size, cfg.n_embd)


def test_extract_hidden_states_and_head_no_trainer_torchvision() -> None:
    """torchvision ResNet 경로: trainer 없이 직접 호출."""
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    batch = {"pixel_values": torch.randn(2, 3, 224, 224)}

    hidden, head_weight = extract_hidden_states_and_head(model, batch)

    assert hidden.shape == (2, 512)
    assert head_weight.shape == (1000, 512)


def test_extract_hidden_states_and_head_no_trainer_timm() -> None:
    """timm 경로: trainer 없이 직접 호출."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    model.eval()
    batch = {"pixel_values": torch.randn(1, 3, 224, 224)}

    hidden, head_weight = extract_hidden_states_and_head(model, batch)

    assert hidden.ndim >= 2
    assert head_weight.shape == (5, 512)


# ────────────────────────────────────────────────────────────── #
#  _extract_hf_pretrained free function 직접 호출               #
# ────────────────────────────────────────────────────────────── #


def test_extract_hf_pretrained_direct_call_without_trainer() -> None:
    """_extract_hf_pretrained를 trainer 없이 직접 호출 — 시그니처 (model, batch, layer_idx)."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=2, n_positions=32, n_ctx=32)
    model = GPT2LMHeadModel(cfg)
    model.eval()

    batch = {"input_ids": torch.randint(0, 64, (1, 4))}
    hidden, head_weight = _extract_hf_pretrained(model, batch, layer_idx=-1)

    assert hidden.shape == (1, 4, cfg.n_embd)
    assert head_weight.shape == (cfg.vocab_size, cfg.n_embd)


def test_extract_hf_pretrained_layer_idx_applied() -> None:
    """layer_idx 인자가 _extract_hf_pretrained free function에 전달된다."""
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=2, n_positions=32, n_ctx=32)
    model = GPT2LMHeadModel(cfg)
    model.eval()

    batch = {"input_ids": torch.randint(0, 64, (1, 4))}
    hidden_last, _ = _extract_hf_pretrained(model, batch, layer_idx=-1)
    hidden_first, _ = _extract_hf_pretrained(model, batch, layer_idx=0)

    assert hidden_last.shape == hidden_first.shape
    # 서로 다른 layer — 내용이 달라야 한다
    assert not torch.allclose(hidden_last, hidden_first)


# ────────────────────────────────────────────────────────────── #
#  _extract_timm free function 직접 호출                         #
# ────────────────────────────────────────────────────────────── #


def test_extract_timm_direct_call_without_trainer() -> None:
    """_extract_timm을 trainer 없이 직접 호출 — 시그니처 (model, batch)."""
    timm = pytest.importorskip("timm")
    model = timm.create_model("resnet18", pretrained=False, num_classes=10)
    model.eval()

    batch = {"pixel_values": torch.randn(2, 3, 224, 224)}
    hidden, head_weight = _extract_timm(model, batch)

    assert hidden.ndim >= 2
    assert hidden.shape[0] == 2
    assert head_weight.shape == (10, 512)


# ────────────────────────────────────────────────────────────── #
#  _extract_torchvision_resnet free function 직접 호출           #
# ────────────────────────────────────────────────────────────── #


def test_extract_torchvision_resnet_direct_call_without_trainer() -> None:
    """_extract_torchvision_resnet을 trainer 없이 직접 호출 — 시그니처 (model, batch)."""
    pytest.importorskip("torchvision")
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()

    batch = {"pixel_values": torch.randn(1, 3, 224, 224)}
    hidden, head_weight = _extract_torchvision_resnet(model, batch)

    assert hidden.shape == (1, 512)
    assert head_weight.shape == (1000, 512)
    assert head_weight.data_ptr() == model.fc.weight.data_ptr()
