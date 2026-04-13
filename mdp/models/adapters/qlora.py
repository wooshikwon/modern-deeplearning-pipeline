"""QLoRA 어댑터 적용 (양자화 + LoRA)."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def _resolve_model_class(class_path: str | None) -> type:
    if class_path is None:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
    import importlib
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def apply_qlora(
    model_name_or_path: str,
    r: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    target_modules: list[str] | str = "all-linear",
    task_type: str | None = None,
    bits: int = 4,
    class_path: str | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    quantization: dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """양자화된 모델에 LoRA 어댑터를 적용한다.

    모델을 BitsAndBytes로 양자화 로딩 후 LoRA를 적용한다.

    Args:
        model_name_or_path: HuggingFace 모델 이름 또는 로컬 경로.
        r: LoRA rank.
        lora_alpha: LoRA 스케일링 계수. alpha로도 지정 가능.
        lora_dropout: LoRA 드롭아웃 비율. dropout으로도 지정 가능.
        target_modules: 적용 대상 모듈 이름 또는 패턴.
        task_type: PEFT TaskType 문자열 (예: "CAUSAL_LM").
        bits: 양자화 비트 수 (4 또는 8).
        alpha: lora_alpha의 단축 이름.
        dropout: lora_dropout의 단축 이름.
        quantization: {bits, type, ...} 설정 dict. bits를 여기서도 받을 수 있다.
        **kwargs: 추가 인자.

    Returns:
        QLoRA가 적용된 PeftModel.
    """
    # quantization dict 플래트닝 — bits를 직접 받거나 quantization.bits에서 가져옴
    if quantization is not None and isinstance(quantization, dict):
        bits = quantization.get("bits", bits)

    # 단축 이름 → PEFT 이름 매핑
    resolved_alpha = lora_alpha if lora_alpha is not None else (alpha if alpha is not None else 16)
    resolved_dropout = lora_dropout if lora_dropout is not None else (dropout if dropout is not None else 0.05)
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "transformers 패키지가 필요합니다: pip install transformers"
        ) from e

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "peft 패키지가 필요합니다: pip install peft"
        ) from e

    # BitsAndBytes 양자화 설정
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"bits는 4 또는 8이어야 합니다, 받은 값: {bits}")

    # from_pretrained에 전달하지 않을 키를 먼저 분리
    modules_to_save = kwargs.pop("modules_to_save", None) or None

    # 양자화된 모델 로딩
    model_cls = _resolve_model_class(class_path)
    model = model_cls.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        **kwargs,
    )

    # kbit 학습 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정 및 적용
    peft_task_type = getattr(TaskType, task_type) if task_type else None

    lora_config = LoraConfig(
        r=r,
        lora_alpha=resolved_alpha,
        lora_dropout=resolved_dropout,
        target_modules=target_modules,
        task_type=peft_task_type,
        modules_to_save=modules_to_save,
    )

    model = get_peft_model(model, lora_config)

    from mdp.models.adapters import log_trainable_params
    log_trainable_params(model)

    return model
