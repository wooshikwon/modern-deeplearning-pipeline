"""Prefix Tuning 어댑터 적용."""

from __future__ import annotations

import logging
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def apply_prefix_tuning(
    model: nn.Module,
    num_virtual_tokens: int = 16,
    task_type: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """모델에 Prefix Tuning 어댑터를 적용한다.

    각 트랜스포머 레이어 입력 앞에 학습 가능한 가상 토큰(prefix)을 삽입한다.
    원래 모델 파라미터는 freeze되고, prefix 파라미터만 학습된다.

    Args:
        model: 기반 모델.
        num_virtual_tokens: prefix 길이 (가상 토큰 수).
        task_type: PEFT TaskType 문자열 (예: "CAUSAL_LM", "SEQ_CLS").
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

    peft_task_type = getattr(TaskType, task_type) if task_type else None

    config = PrefixTuningConfig(
        num_virtual_tokens=num_virtual_tokens,
        task_type=peft_task_type,
        **kwargs,
    )

    model = get_peft_model(model, config)

    return model
