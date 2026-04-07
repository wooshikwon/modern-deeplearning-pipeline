"""어댑터 공개 API."""

from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


def log_trainable_params(model: nn.Module) -> None:
    """학습 가능 파라미터 수를 로깅한다."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / max(total, 1) if total > 0 else 0.0
    logger.info(f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
