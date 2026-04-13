"""LoRA 어댑터 적용."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int | None = None,
    lora_dropout: float | None = None,
    target_modules: list[str] | str = "all-linear",
    task_type: str | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    **kwargs: Any,
) -> nn.Module:
    """모델에 LoRA 어댑터를 적용한다.

    Args:
        model: 기반 모델.
        r: LoRA rank.
        lora_alpha: LoRA 스케일링 계수. alpha로도 지정 가능.
        lora_dropout: LoRA 드롭아웃 비율. dropout으로도 지정 가능.
        target_modules: 적용 대상 모듈 이름 또는 패턴.
        task_type: PEFT TaskType 문자열 (예: "CAUSAL_LM").
        alpha: lora_alpha의 단축 이름.
        dropout: lora_dropout의 단축 이름.
        **kwargs: LoraConfig에 전달할 추가 인자.

    Returns:
        LoRA가 적용된 PeftModel.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "peft 패키지가 필요합니다: pip install peft"
        ) from e

    # 단축 이름 → PEFT 이름 매핑 (명시적 lora_alpha/lora_dropout이 우선)
    resolved_alpha = lora_alpha if lora_alpha is not None else (alpha if alpha is not None else 16)
    resolved_dropout = lora_dropout if lora_dropout is not None else (dropout if dropout is not None else 0.0)

    peft_task_type = getattr(TaskType, task_type) if task_type else None

    config = LoraConfig(
        r=r,
        lora_alpha=resolved_alpha,
        lora_dropout=resolved_dropout,
        target_modules=target_modules,
        task_type=peft_task_type,
        **kwargs,
    )

    model = get_peft_model(model, config)

    from mdp.models.adapters import log_trainable_params
    log_trainable_params(model)

    return model
