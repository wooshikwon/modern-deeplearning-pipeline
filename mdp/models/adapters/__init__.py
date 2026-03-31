"""어댑터 공개 API — method별 라우팅."""

from __future__ import annotations

from typing import Any

from torch import nn


def apply_adapter(
    model: nn.Module,
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

    # alpha → lora_alpha 매핑
    if "alpha" in config:
        config["lora_alpha"] = config.pop("alpha")
    if "dropout" in config:
        config["lora_dropout"] = config.pop("dropout")

    if method == "lora":
        from mdp.models.adapters.lora import apply_lora
        return apply_lora(model, **config)
    elif method == "qlora":
        from mdp.models.adapters.qlora import apply_qlora
        # qlora는 model 대신 model_name_or_path를 받음
        model_name = config.pop("model_name_or_path", None)
        if model_name is None:
            raise ValueError("QLoRA에는 model_name_or_path가 필요합니다")
        return apply_qlora(model_name, **config)
    else:
        raise ValueError(f"지원하지 않는 어댑터 method: {method!r}")
