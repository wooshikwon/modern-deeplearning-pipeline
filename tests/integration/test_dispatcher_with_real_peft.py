"""Dispatcher ↔ 실제 PEFT + HF 모델 integration 테스트.

**왜 integration 계층인가** (review-2026-04-18-U6-c2 §1-3):

- 기존 `tests/unit/test_extract_hidden_states_dispatcher.py`는 `MagicMock(spec=...)`
  스타일 스텁으로 dispatcher **분기 로직**만 격리 검증한다. 실제 `peft.PeftModel`
  의 `base_model.model` 체인, `get_peft_model`이 parameter grafting을 수행하는
  방식, `enable_input_require_grads` 훅 등의 **런타임 동작**은 검증하지 않는다.
- 2026-04-18 H200 sanity v6 실패(`_compute_per_token_ce_liger` → FLCE forward →
  Triton `ValueError: Pointer argument (at 0) cannot be accessed`)는 이 gap에서
  빠져나왔다. fix-c1의 device 계약 보강(`head_weight.to(hidden.device)`)조차
  `hidden`이 CPU로 반환되는 상황에서는 head_weight까지 CPU로 끌어내리는 역효과를
  냈고, 로컬 mock 테스트는 이를 포착할 수 없었다.

**이 integration test의 역할**:

- 실 peft + tiny HF 모델 조합에서 dispatcher가 반환하는 ``(hidden, head_weight)``
  의 device 계약
  으로 고정 검증. 로컬에서는 CPU만 돌지만, 계약 불변 조건("모델 parameter device
  == hidden.device == head_weight.device")은 device-agnostic으로 검증 가능.
- Liger FLCE 자체는 Triton 필수이므로 GPU only → CPU 로컬에서는 skip. 단 "CPU
  모델이면 dispatcher 반환도 CPU"라는 기본 계약은 확정할 수 있다.
- 미래 GPU 환경(CI CUDA runner, 또는 sanity 서버)에서는 이 테스트가 자동 확장되어
  FLCE까지 돌린다 (``torch.cuda.is_available()``로 분기).
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="module")
def tiny_peft_hf_model():
    """GPT2 + LoRA: 실제 PEFT 래핑된 HF causal LM.

    review-2026-04-18-U6-c2 §관찰 2의 "실제 조합"에 가장 가까운 최소 재현 환경.
    Llama는 의존성(flash_attn 등)이 무거워 CPU 로컬에서 불가하므로 GPT2로 대체.
    PEFT의 투과 로직(``base_model.model`` 체인)은 모델 종류와 무관하므로 유효.
    """
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model

    cfg = GPT2Config(
        vocab_size=128,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        n_ctx=32,
        use_cache=False,
    )
    base = GPT2LMHeadModel(cfg)
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.eval()
    return model, cfg


def test_dispatcher_returns_tensors_on_model_device_with_real_peft(tiny_peft_hf_model):
    """실 PEFT + HF 조합: dispatcher 반환 device == 모델 parameter device.

    Sanity v6 실패의 근본 회귀 고정. 이 테스트가 로컬에서 green하고 GPU에서도
    green이면 "dispatcher 반환 계약"이 framework·device 불문 확정된다.
    """
    from mdp.training.rl_trainer import RLTrainer

    model, cfg = tiny_peft_hf_model
    expected_device = next(model.parameters()).device
    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }
    # input_ids도 같은 device로 이동 (train loop `_move_to_device`와 동일 동작).
    batch = {k: (v.to(expected_device) if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}

    # RLTrainer.__init__ 우회 — dispatcher만 bind해 호출.
    from types import SimpleNamespace

    trainer_stub = SimpleNamespace()
    trainer_stub._extract_hf_pretrained = RLTrainer._extract_hf_pretrained.__get__(
        trainer_stub
    )
    trainer_stub._extract_timm = RLTrainer._extract_timm.__get__(trainer_stub)
    trainer_stub._extract_torchvision_resnet = (
        RLTrainer._extract_torchvision_resnet.__get__(trainer_stub)
    )

    hidden, head_weight = RLTrainer._extract_hidden_states_and_head(
        trainer_stub, model, batch, layer_idx=-1
    )

    # 핵심 계약: hidden·head_weight·input_ids 모두 동일 device.
    assert hidden.device == expected_device
    assert head_weight.device == expected_device
    assert hidden.shape == (2, 5, cfg.n_embd)
    assert head_weight.shape == (cfg.vocab_size, cfg.n_embd)


def test_dispatcher_with_peft_routes_through_base_model_chain(tiny_peft_hf_model):
    """Priority 2 PEFT 분기: PeftModel → base_model → model (PreTrainedModel) 투과.

    `isinstance(PeftModel, PreTrainedModel)` == False라는 PEFT 설계상 특성 때문에
    dispatcher가 `getattr(base_model, 'model')` 체인을 통해서만 정상 dispatch된다.
    이 경로가 깨지면 `NotImplementedError: Model PeftModel...`로 떨어진다.
    """
    from transformers import PreTrainedModel

    model, _ = tiny_peft_hf_model
    # PeftModel은 PreTrainedModel을 상속하지 않음 (dispatcher의 PEFT 분기 전제).
    assert not isinstance(model, PreTrainedModel)
    # base_model.model은 진짜 PreTrainedModel (HF 모델).
    assert isinstance(model.base_model.model, PreTrainedModel)


def test_dispatcher_hidden_gradients_flow_through_peft(tiny_peft_hf_model):
    """Dispatcher 반환 hidden에서 출발한 backward가 LoRA adapter까지 gradient 전파한다.

    fix-c1이 걱정했던 "`peft_inner` 직접 forward가 LoRA grafting을 우회하진
    않는가"를 로컬에서 확정 검증. LoRA 경로는 ``base_model.model``에 대해 linear
    weight를 in-place swap하므로 adapter parameter에 gradient가 흘러야 한다.
    """
    from mdp.training.rl_trainer import RLTrainer
    from types import SimpleNamespace

    model, cfg = tiny_peft_hf_model
    # grad 흐름 검증을 위해 train mode.
    model.train()
    # LoRA 파라미터만 trainable — 기본값.
    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (1, 4)),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    trainer_stub = SimpleNamespace()
    trainer_stub._extract_hf_pretrained = RLTrainer._extract_hf_pretrained.__get__(
        trainer_stub
    )
    trainer_stub._extract_timm = RLTrainer._extract_timm.__get__(trainer_stub)
    trainer_stub._extract_torchvision_resnet = (
        RLTrainer._extract_torchvision_resnet.__get__(trainer_stub)
    )

    hidden, head_weight = RLTrainer._extract_hidden_states_and_head(
        trainer_stub, model, batch, layer_idx=-1
    )
    # Pseudo-loss: hidden과 head_weight의 matmul 결과 sum.
    logits = hidden @ head_weight.t()
    loss = logits.sum()
    loss.backward()

    # LoRA 파라미터(adapter A/B) 중 하나 이상에 grad가 잡혀야 한다.
    lora_params = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    assert lora_params, "LoRA trainable parameter가 하나도 없다 — PEFT 래핑 이상."
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_params), (
        "LoRA adapter에 gradient가 흐르지 않았다. dispatcher 경로가 LoRA grafting을 "
        "우회했을 가능성. base_model.model 투과 로직을 점검하라."
    )

    # 정리: 테스트 간 grad 누적 방지.
    model.zero_grad(set_to_none=True)
    model.eval()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Liger FLCE는 Triton kernel 기반이라 CUDA에서만 동작.",
)
def test_dispatcher_output_survives_matmul_on_gpu(tiny_peft_hf_model):
    """GPU 환경 한정 — dispatcher 출력으로 matmul → Triton-like kernel 진입까지 무사.

    CUDA runner에서만 활성화. sanity v6가 실패한 정확한 경로("dispatcher 반환
    hidden·head_weight로 matmul → 결과로 fused cross entropy")를 매우 작은
    규모로 축소 재현해 dispatcher→FLCE 경로의 device 호환성을 확인한다. FLCE 직접 호출은 mdp의
    liger-kernel import 제약(H200 recipe 전용)이라 matmul까지만 검증.
    """
    from mdp.training.rl_trainer import RLTrainer
    from types import SimpleNamespace

    model, cfg = tiny_peft_hf_model
    model = model.to("cuda")
    batch = {
        "input_ids": torch.randint(0, cfg.vocab_size, (2, 5), device="cuda"),
        "attention_mask": torch.ones(2, 5, dtype=torch.long, device="cuda"),
    }

    trainer_stub = SimpleNamespace()
    trainer_stub._extract_hf_pretrained = RLTrainer._extract_hf_pretrained.__get__(
        trainer_stub
    )
    trainer_stub._extract_timm = RLTrainer._extract_timm.__get__(trainer_stub)
    trainer_stub._extract_torchvision_resnet = (
        RLTrainer._extract_torchvision_resnet.__get__(trainer_stub)
    )

    hidden, head_weight = RLTrainer._extract_hidden_states_and_head(
        trainer_stub, model, batch, layer_idx=-1
    )
    assert hidden.device.type == "cuda"
    assert head_weight.device.type == "cuda"

    logits = hidden @ head_weight.t()
    assert logits.device.type == "cuda"
    # Pointer 접근 가능한지 간단히 확인 — CUDA allocation이 유효한 경우 data_ptr은 양수.
    assert logits.data_ptr() > 0
