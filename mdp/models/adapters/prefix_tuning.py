"""Prefix Tuning 어댑터 적용."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def apply_prefix_tuning(
    model: nn.Module,
    num_virtual_tokens: int | None = None,
    task_type: str | None = None,
    r: int | None = None,
    alpha: int | None = None,
    dropout: float | None = None,
    target_modules: list[str] | str | None = None,
    modules_to_save: list[str] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """모델에 Prefix Tuning 어댑터를 적용한다.

    각 트랜스포머 레이어 입력 앞에 학습 가능한 가상 토큰(prefix)을 삽입한다.
    원래 모델 파라미터는 freeze되고, prefix 파라미터만 학습된다.

    Args:
        model: 기반 모델.
        num_virtual_tokens: prefix 길이 (가상 토큰 수). r로도 지정 가능.
        task_type: PEFT TaskType 문자열 (예: "CAUSAL_LM", "SEQ_CLS").
        r: num_virtual_tokens의 단축 이름.
        alpha: LoRA 전용 — 무시됨.
        dropout: LoRA 전용 — 무시됨.
        target_modules: LoRA 전용 — 무시됨.
        modules_to_save: LoRA 전용 — 무시됨.
        **kwargs: PrefixTuningConfig에 전달할 추가 인자.

    Returns:
        Prefix Tuning이 적용된 PeftModel.
    """
    try:
        from peft import PrefixTuningConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "peft 패키지가 필요합니다: pip install peft"
        ) from e

    # r → num_virtual_tokens 매핑
    resolved_tokens = num_virtual_tokens if num_virtual_tokens is not None else (r if r is not None else 16)

    peft_task_type = getattr(TaskType, task_type) if task_type else None

    config = PrefixTuningConfig(
        num_virtual_tokens=resolved_tokens,
        task_type=peft_task_type,
        **kwargs,
    )

    model = get_peft_model(model, config)

    from mdp.models.adapters import log_trainable_params
    log_trainable_params(model)

    return model
