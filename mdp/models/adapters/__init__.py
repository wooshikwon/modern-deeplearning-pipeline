"""어댑터 공개 API — method별 라우팅."""

from __future__ import annotations

import logging
from typing import Any

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


def apply_adapter(
    model: nn.Module | None,
    adapter_config: dict[str, Any],
) -> nn.Module:
    """adapter_config의 method에 따라 적절한 어댑터를 적용한다.

    Args:
        model: 기반 모델.
        adapter_config: method, r, alpha, dropout 등을 포함하는 설정 dict.

    Returns:
        어댑터가 적용된 모델.
    """
    method = adapter_config.get("method", "").lower()
    config = {k: v for k, v in adapter_config.items() if k != "method"}

    # quantization dict 플래트닝 (QLoRA)
    if "quantization" in config:
        quant = config.pop("quantization")
        if isinstance(quant, dict):
            for qk, qv in quant.items():
                config.setdefault(qk, qv)

    # alpha → lora_alpha 매핑 (lora/qlora 전용)
    if method in ("lora", "qlora"):
        if "alpha" in config:
            config["lora_alpha"] = config.pop("alpha")
        if "dropout" in config:
            config["lora_dropout"] = config.pop("dropout")
    else:
        # prefix_tuning 등에는 lora 키 매핑을 적용하지 않음
        pass

    if method == "lora":
        if model is None:
            raise ValueError("LoRA에는 model이 필요합니다")
        from mdp.models.adapters.lora import apply_lora
        result = apply_lora(model, **config)
    elif method == "qlora":
        from mdp.models.adapters.qlora import apply_qlora
        model_name = config.pop("model_name_or_path", None)
        if model_name is None:
            raise ValueError("QLoRA에는 model_name_or_path가 필요합니다")
        result = apply_qlora(model_name, **config)
    elif method == "prefix_tuning":
        if model is None:
            raise ValueError("PrefixTuning에는 model이 필요합니다")
        from mdp.models.adapters.prefix_tuning import apply_prefix_tuning
        if "r" in config:
            config["num_virtual_tokens"] = config.pop("r")
        for k in ("dropout", "target_modules", "modules_to_save", "alpha"):
            config.pop(k, None)
        result = apply_prefix_tuning(model, **config)
    else:
        raise ValueError(f"지원하지 않는 어댑터 method: {method!r}")

    log_trainable_params(result)
    return result
