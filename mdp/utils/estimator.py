"""MemoryEstimator -- 학습 메모리 사용량을 추정하고 GPU/전략을 추천한다."""

from __future__ import annotations

import logging
import math
from typing import Any

from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)

# 일반적인 GPU VRAM 크기 (GB)
_GPU_VRAM_MAP: dict[str, float] = {
    "A100-80": 80.0,
    "A100-40": 40.0,
    "A100": 80.0,
    "H100": 80.0,
    "V100": 16.0,
    "L40S": 48.0,
    "RTX4090": 24.0,
    "RTX3090": 24.0,
    "T4": 16.0,
}

# 기본 단일 GPU VRAM (추정용)
_DEFAULT_GPU_VRAM_GB = 24.0

# 옵티마이저 상태 배수 (파라미터 메모리 대비)
_OPTIMIZER_STATE_MULTIPLIER: dict[str, float] = {
    "sgd": 1.0,  # momentum buffer
    "adam": 2.0,  # m + v
    "adamw": 2.0,
    "adafactor": 0.5,  # 근사적으로 낮음
}


class MemoryEstimator:
    """학습 메모리 사용량을 추정한다.

    모델 파라미터 수와 학습 설정으로부터 대략적인 GPU 메모리 요구량을 계산하고,
    필요한 GPU 수와 분산 전략을 추천한다.
    """

    def estimate(self, settings: Settings) -> dict[str, Any]:
        """메모리 사용량을 추정하고 GPU/전략을 추천한다.

        Args:
            settings: 통합 설정 객체.

        Returns:
            dict with keys:
                - ``model_mem_gb``: 모델 파라미터 메모리 (GB).
                - ``gradient_mem_gb``: 그래디언트 메모리 (GB).
                - ``optimizer_mem_gb``: 옵티마이저 상태 메모리 (GB).
                - ``activation_mem_gb``: 활성화 메모리 추정 (GB).
                - ``total_mem_gb``: 총 추정 메모리 (GB).
                - ``suggested_gpus``: 추천 GPU 수.
                - ``suggested_strategy``: 추천 분산 전략.
        """
        recipe = settings.recipe

        # 파라미터 수 추정
        param_count = self._estimate_param_count(recipe.model.class_path)
        bytes_per_param = self._bytes_per_param(recipe.training.precision)

        # 1. 모델 메모리
        model_mem = param_count * bytes_per_param

        # 2. 그래디언트 메모리 (학습 시 fp32)
        gradient_mem = param_count * 4  # gradients always fp32

        # 3. 옵티마이저 상태 메모리
        optimizer_name = self._extract_optimizer_name(recipe.optimizer)
        opt_multiplier = _OPTIMIZER_STATE_MULTIPLIER.get(optimizer_name, 2.0)
        optimizer_mem = param_count * 4 * opt_multiplier  # states in fp32

        # 4. 활성화 메모리 (배치 크기 의존, heuristic)
        batch_size = recipe.data.dataloader.batch_size
        grad_ckpt = recipe.training.gradient_checkpointing
        activation_mem = self._estimate_activation_mem(
            param_count=param_count,
            batch_size=batch_size,
            gradient_checkpointing=grad_ckpt,
        )

        # 총합
        total_mem = model_mem + gradient_mem + optimizer_mem + activation_mem

        # GB 변환
        to_gb = 1 / (1024**3)
        model_mem_gb = model_mem * to_gb
        gradient_mem_gb = gradient_mem * to_gb
        optimizer_mem_gb = optimizer_mem * to_gb
        activation_mem_gb = activation_mem * to_gb
        total_mem_gb = total_mem * to_gb

        # GPU 추천
        gpu_vram = self._get_gpu_vram(settings)
        # 안전 마진 10%
        usable_vram = gpu_vram * 0.9
        suggested_gpus = max(1, math.ceil(total_mem_gb / usable_vram))

        # 전략 추천
        suggested_strategy = self._suggest_strategy(
            total_mem_gb=total_mem_gb,
            model_mem_gb=model_mem_gb,
            gpu_vram=gpu_vram,
            suggested_gpus=suggested_gpus,
        )

        return {
            "model_mem_gb": round(model_mem_gb, 2),
            "gradient_mem_gb": round(gradient_mem_gb, 2),
            "optimizer_mem_gb": round(optimizer_mem_gb, 2),
            "activation_mem_gb": round(activation_mem_gb, 2),
            "total_mem_gb": round(total_mem_gb, 2),
            "suggested_gpus": suggested_gpus,
            "suggested_strategy": suggested_strategy,
        }

    # ── Internal ──

    @staticmethod
    def _estimate_param_count(class_path: str) -> int:
        """class_path에서 모델 크기를 추정한다.

        1. HuggingFace AutoConfig로 정확한 추정 시도
        2. 모델명의 숫자 패턴으로 추정 (예: ``llama-7b`` → 7B)
        3. 알려진 모델 사전 조회
        4. 기본값 100M
        """
        import re

        # 1. AutoConfig로 정확한 추정 시도
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(class_path)
            h = getattr(config, "hidden_size", None)
            n = getattr(config, "num_hidden_layers", None)
            v = getattr(config, "vocab_size", None)
            if h and n:
                param_est = 12 * n * h * h
                if v:
                    param_est += v * h
                return param_est
        except Exception:
            pass

        name_lower = class_path.lower()

        # 2. 패턴: 숫자 + b (billion)
        match = re.search(r"(\d+\.?\d*)b", name_lower)
        if match:
            return int(float(match.group(1)) * 1e9)

        # 패턴: 숫자 + m (million)
        match = re.search(r"(\d+\.?\d*)m", name_lower)
        if match:
            return int(float(match.group(1)) * 1e6)

        # 3. 알려진 모델 크기 (fallback)
        known_sizes: dict[str, int] = {
            "gpt2": 124_000_000,
            "bert-base": 110_000_000,
            "bert-large": 340_000_000,
            "resnet50": 25_000_000,
            "resnet18": 11_000_000,
            "vit-base": 86_000_000,
            "vit-large": 307_000_000,
        }
        for pattern, size in known_sizes.items():
            if pattern in name_lower:
                return size

        # 4. 기본값: 100M
        logger.warning(
            "모델 크기를 추정할 수 없어 기본값 100M 사용: %s", class_path
        )
        return 100_000_000

    @staticmethod
    def _bytes_per_param(precision: str) -> int:
        """정밀도에 따른 파라미터당 바이트 수."""
        precision_map = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "int8": 1,
            "int4": 1,  # 실제로는 0.5 이지만 오버헤드 포함
        }
        return precision_map.get(precision, 4)

    @staticmethod
    def _extract_optimizer_name(optimizer_config: dict[str, Any] | None) -> str:
        """옵티마이저 설정에서 이름을 추출한다."""
        if optimizer_config is None:
            return "adamw"  # safe default
        component = optimizer_config.get("_component_", "")
        name = component.rsplit(".", 1)[-1].lower()
        return name

    @staticmethod
    def _estimate_activation_mem(
        param_count: int,
        batch_size: int,
        gradient_checkpointing: bool,
    ) -> int:
        """활성화 메모리를 추정한다 (bytes).

        Heuristic: 활성화 메모리 ~ param_count * batch_size * 2 bytes
        (매우 대략적이며, 실제 아키텍처에 따라 크게 달라진다.)
        gradient checkpointing 사용 시 ~1/3로 감소.
        """
        # 기본 추정: 파라미터 수 * 2 (fp16 활성화) * sqrt(batch_size)
        # sqrt를 쓰는 이유: 활성화 메모리는 batch_size에 선형이지만
        # 레이어 수에도 비례하여 param_count에 이미 반영되어 있으므로
        base_activation = param_count * 2 * math.sqrt(batch_size)

        if gradient_checkpointing:
            base_activation *= 0.33

        return int(base_activation)

    @staticmethod
    def _get_gpu_vram(settings: Settings) -> float:
        """사용할 GPU의 VRAM(GB)을 추정한다."""
        dist_config = settings.config.compute.distributed
        if isinstance(dist_config, dict):
            accelerators = dist_config.get("accelerators", "")
            # "A100:4" → "A100"
            gpu_name = accelerators.split(":")[0] if accelerators else ""
            if gpu_name in _GPU_VRAM_MAP:
                return _GPU_VRAM_MAP[gpu_name]

        # 런타임 감지 시도
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_mem
                return mem / (1024**3)
        except (ImportError, RuntimeError):
            pass

        return _DEFAULT_GPU_VRAM_GB

    @staticmethod
    def _suggest_strategy(
        total_mem_gb: float,
        model_mem_gb: float,
        gpu_vram: float,
        suggested_gpus: int,
    ) -> str:
        """분산 전략을 추천한다.

        - 단일 GPU로 충분하면 ``"none"``
        - 모델이 단일 GPU에 들어가면 ``"ddp"``
        - 모델이 크면 ``"fsdp"``
        - 매우 크면 ``"deepspeed_zero3"``
        """
        if suggested_gpus <= 1:
            return "none"

        if model_mem_gb < gpu_vram * 0.7:
            # 모델이 단일 GPU에 여유있게 들어감 → DDP
            return "ddp"

        if model_mem_gb < gpu_vram * 2:
            # 모델이 2개 GPU에 분산 가능 → FSDP
            return "fsdp"

        # 매우 큰 모델 → DeepSpeed ZeRO-3
        return "deepspeed_zero3"
