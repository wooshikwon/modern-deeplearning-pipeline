"""Feature extractor free functions — framework-agnostic (hidden, head_weight) dispatcher.

이 모듈은 trainer 인스턴스 없이 호출 가능한 stateless free function을 제공한다.
소비자: ``RLTrainer`` (직접 호출).

원칙 4 (spec-training-restructure §설계 원칙):
  ``(model, batch, layer_idx)`` 입력만으로 동작하며, trainer 상태(self.algorithm,
  self.trainable 등)에 의존하지 않는다. 향후 inference callback 등에서도 재사용 가능.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def extract_logits(model_output) -> torch.Tensor:
    """model forward 출력에서 logits 텐서를 꺼낸다.

    지원 형태:
    - HF ModelOutput (``out.logits``)
    - dict (``out["logits"]`` 또는 ``out["output"]`` fallback)
    - 그 외 — 그대로 반환 (이미 tensor인 경우)
    """
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict):
        return model_output.get("logits", model_output.get("output"))
    return model_output


def forward_model(model: nn.Module, batch: dict, role: str = "policy") -> dict:
    """Causal forward를 수행하고 role별 출력 키를 정규화한다.

    role에 따라 출력 키가 달라진다:
    - "policy", "reference" → {"logits": (batch, seq, vocab)}
    - "value" → {"values": (batch, seq)} — scalar head 또는 LM head[:, :, 0]
    - "reward" → {"reward": (batch,)} 우선, fallback으로 {"logits": tensor}

    Preference 배치(chosen/rejected)는 caller에서 분리하여 2회 호출한다.
    """
    result = {}

    if "input_ids" in batch:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        logits = extract_logits(out)

        if role == "value":
            # value model: logits → (batch, seq) scalar values
            if logits.dim() == 3 and logits.shape[-1] == 1:
                result["values"] = logits.squeeze(-1)
            elif logits.dim() == 3:
                result["values"] = logits[:, :, 0]
            else:
                result["values"] = logits
        elif role == "reward":
            # reward model: 명시적 "reward" 키 우선, 없으면 logits 반환
            if isinstance(out, dict) and "reward" in out:
                result["reward"] = out["reward"]
            result["logits"] = logits
        else:
            result["logits"] = logits

    return result


def _extract_hf_pretrained(
    model: nn.Module,
    batch: dict,
    layer_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``PreTrainedModel`` 기본 구현.

    ``hidden_states[layer_idx]``와 ``get_output_embeddings().weight``을 반환한다.
    기본(``layer_idx=-1``)에서는 ``base_model`` property로 inner encoder를 직접 호출해
    ``last_hidden_state``만 꺼내는 효율 경로를 사용한다 (메모리 절감, 아래 주석 참조).
    """
    # 효율 경로 vs full-tuple 경로 분기.
    #
    # ``output_hidden_states=True``는 HF가 **모든 layer의 hidden state**를
    # tuple로 저장하여 forward 중 activation graph에 상주시킨다. Llama-3-8B
    # (32 layer) + bs=32 + seq=1879 + bf16 기준 약 14.5 GiB 오버헤드
    # (2026-04-19 U6 sanity snapshot 실측: rl_trainer.py:1579 at 14.56 GiB).
    # 현재 첫 consumer(weighted_ntp)는 항상 layer_idx=-1만 사용하므로 이
    # 전부를 저장할 이유가 없다.
    #
    # PreTrainedModel의 ``base_model`` property는 ``base_model_prefix``에
    # 해당하는 inner encoder(LlamaForCausalLM.model → LlamaModel,
    # GPT2LMHeadModel.transformer → GPT2Model 등)를 반환. 이 base encoder를
    # 직접 호출하면 ``last_hidden_state``만 받고 중간 layer tuple은 생성되지
    # 않는다. LoRA 는 linear module in-place swap이므로 base encoder forward
    # 에도 LoRA delta가 정상 반영된다.
    #
    # layer_idx != -1 (드문 케이스)이거나 base_model이 self를 가리키는 희귀
    # 모델에서는 full-tuple fallback.
    base_encoder = getattr(model, "base_model", None)
    use_efficient_path = (
        layer_idx == -1
        and base_encoder is not None
        and base_encoder is not model
    )
    hidden: torch.Tensor | None = None
    if use_efficient_path:
        assert base_encoder is not None  # guaranteed by use_efficient_path condition above
        base_out = base_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        hidden = getattr(base_out, "last_hidden_state", None)
        if hidden is None:
            # base encoder가 last_hidden_state를 안 내는 edge case → fallback
            use_efficient_path = False
    if not use_efficient_path:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
        )
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise NotImplementedError(
                f"HF 모델 {type(model).__name__}이 hidden_states를 반환하지 않습니다. "
                "extract_features_and_head를 override하세요."
            )
        hidden = hidden_states[layer_idx]

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or getattr(output_embeddings, "weight", None) is None:
        raise NotImplementedError(
            f"HF 모델 {type(model).__name__}에는 output embedding이 없습니다 "
            "(encoder-only 모델 등). extract_features_and_head를 override하세요."
        )
    head_weight = output_embeddings.weight
    return hidden, head_weight


def _extract_timm(
    model: nn.Module,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """timm 모델 기본 구현.

    ``forward_features`` + ``get_classifier``의 weight을 반환한다.
    ``Identity`` classifier(num_classes=0)는 지원 불가.
    """
    pixel_values = batch.get("pixel_values")
    if pixel_values is None:
        raise NotImplementedError(
            "timm 모델은 batch['pixel_values']를 요구합니다. "
            "현재 batch keys: " + ", ".join(sorted(batch.keys()))
        )
    features = model.forward_features(pixel_values)
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Identity):
        raise NotImplementedError(
            f"timm 모델 {type(model).__name__}의 classifier가 Identity입니다 "
            "(num_classes=0). extract_features_and_head를 override하세요."
        )
    weight = getattr(classifier, "weight", None)
    if weight is None:
        raise NotImplementedError(
            f"timm 모델 {type(model).__name__}의 classifier에 weight 속성이 "
            "없습니다. extract_features_and_head를 override하세요."
        )
    return features, weight


def _extract_torchvision_resnet(
    model: nn.Module,
    batch: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torchvision ``ResNet`` 기본 구현.

    ``conv1``~``avgpool``까지 수동 forward 후 ``flatten`` 결과를 hidden으로,
    ``model.fc.weight``을 head로 반환한다. Custom ResNet variant는
    ``extract_features_and_head`` override 필요.
    """
    x = batch.get("pixel_values")
    if x is None:
        raise NotImplementedError(
            "torchvision ResNet은 batch['pixel_values']를 요구합니다. "
            "현재 batch keys: " + ", ".join(sorted(batch.keys()))
        )
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    hidden = torch.flatten(x, 1)

    fc = getattr(model, "fc", None)
    if fc is None or getattr(fc, "weight", None) is None:
        raise NotImplementedError(
            f"torchvision 모델 {type(model).__name__}의 fc가 없습니다. "
            "extract_features_and_head를 override하세요."
        )
    head_weight = fc.weight
    return hidden, head_weight


def extract_hidden_states_and_head(
    model: nn.Module,
    batch: dict,
    layer_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Framework-agnostic dispatcher for (hidden, head_weight) extraction.

    Priority:
        1. Model의 ``extract_features_and_head`` override (BaseModel subclass 등).
        2. HF ``PreTrainedModel`` 기본 — ``output_hidden_states=True`` 경유
           (PEFT 래핑 ``base_model.model`` 체인 투과 포함).
        3. timm 모델 기본 — ``forward_features`` + ``get_classifier``.
        4. torchvision ``ResNet`` 계열 기본 — manual layer forward + ``model.fc``.
        5. 그 외 — ``NotImplementedError`` with guidance.

    Returns:
        (hidden, head_weight):
          - hidden: framework-dependent shape. NLP causal LM은 ``(B, S, H)``.
          - head_weight: output projection weight. ``(V, H)`` 또는 ``(C, H)``.
    """
    # DDP/FSDP wrapper 언래핑 — 래핑된 모델은 .module으로 실제 모델에 접근
    unwrapped = getattr(model, "module", model)

    # Priority 1: 모델의 override (BaseModel subclass 또는 framework wrapper)
    if hasattr(unwrapped, "extract_features_and_head"):
        try:
            return unwrapped.extract_features_and_head(batch, layer_idx=layer_idx)
        except NotImplementedError:
            # BaseModel의 기본 구현이 NotImplementedError이므로 dispatcher로 위임
            pass

    # Priority 2: HF PreTrainedModel (직접 또는 PEFT 래핑 투과).
    # PeftModel은 HF PreTrainedModel을 상속하지 않으므로 isinstance 직접 체크에
    # 걸리지 않는다. PeftModel.base_model(LoraModel).model(PreTrainedModel) 경로로
    # 내려가면 LM head가 살아있는 진짜 PreTrainedModel이 나온다.
    # 비-PEFT HF 모델(LlamaForCausalLM 등)은 먼저 isinstance에 걸려 언래핑 skip.
    try:
        from transformers import PreTrainedModel

        if isinstance(unwrapped, PreTrainedModel):
            return _extract_hf_pretrained(unwrapped, batch, layer_idx)
        peft_base = getattr(unwrapped, "base_model", None)
        peft_inner = getattr(peft_base, "model", None) if peft_base is not None else None
        if peft_inner is not None and isinstance(peft_inner, PreTrainedModel):
            return _extract_hf_pretrained(peft_inner, batch, layer_idx)
    except ImportError:
        pass

    # Priority 3: timm 모델
    try:
        import timm.models

        # timm 모델은 일반적으로 `default_cfg` 속성을 가지며,
        # `forward_features`/`get_classifier` 메서드를 보유한다.
        is_timm = (
            hasattr(unwrapped, "default_cfg")
            and hasattr(unwrapped, "forward_features")
            and hasattr(unwrapped, "get_classifier")
        )
        # timm.models 모듈 import가 성공한 경우에만 timm 경로 진입
        _ = timm.models  # 린터 무시 + 명시적 import 의존
        if is_timm:
            return _extract_timm(unwrapped, batch)
    except ImportError:
        pass

    # Priority 4: torchvision ResNet 계열
    try:
        from torchvision.models.resnet import ResNet

        if isinstance(unwrapped, ResNet):
            return _extract_torchvision_resnet(unwrapped, batch)
    except ImportError:
        pass

    # Priority 5: 미지원 모델 — 명확한 안내 메시지
    raise NotImplementedError(
        f"Model {type(unwrapped).__name__}의 extract_features_and_head "
        "기본 구현이 없습니다. BaseModel을 상속하고 override하거나, "
        "지원되는 framework(HF PreTrainedModel / timm / torchvision ResNet)로 "
        "감싸세요."
    )
