"""Liger-Kernel monkey-patch 헬퍼.

Fused Linear Cross-Entropy(FLCE) 등 memory-efficient 경로를 HF LlamaForCausalLM에
주입한다. 호출 시점은 **모델 `from_pretrained` 로딩 이전**이어야 효과가 있다.

Liger-Kernel은 optional dependency(``pip install mdp[liger]``)이며, 미설치 환경
(예: CPU CI)에서는 ImportError를 silent하게 흡수하고 기존 경로로 동작한다.

참조: spec-algorithm-hidden-states-support §원칙 7, §U2.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_liger_patches() -> bool:
    """HF LlamaForCausalLM에 Liger monkey-patch를 적용한다.

    Returns:
        True면 patch 적용, False면 skip (Liger 미설치 또는 이전 호출에서 이미 적용).

    호출 규약:
    - 반드시 ``from_pretrained`` 이전에 호출.
    - Distributed 환경에서는 각 rank subprocess의 entry point에서 한 번씩 호출해야
      모든 rank가 동일한 patched 모델을 갖는다.
    - Idempotent: 이미 적용된 상태에서 재호출해도 안전하지만, 중복 로그를 피하기
      위해 module-level 가드로 한 번만 실제 적용한다.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
    except ImportError:
        logger.debug("liger-kernel not installed, skipping Liger monkey-patch")
        return False

    try:
        apply_liger_kernel_to_llama(fused_linear_cross_entropy=True)
    except Exception as e:  # noqa: BLE001
        # apply_*는 transformers의 특정 버전에서만 동작. 호환성 문제는 graceful degradation.
        logger.warning("Liger monkey-patch failed: %s", e)
        return False

    _APPLIED = True
    logger.info("Liger kernel applied: fused_linear_cross_entropy=True")
    return True
