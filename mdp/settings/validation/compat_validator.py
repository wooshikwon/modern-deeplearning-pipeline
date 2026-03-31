"""CompatValidator — Config-Recipe 호환성 검증.

GPU 수-분산 전략 정합성, serving backend 호환성,
FSDP + QLoRA 비호환 등 Config와 Recipe 간 호환성을 검증한다.
"""

from __future__ import annotations

from mdp.settings.schema import Settings
from mdp.settings.validation import ValidationResult


def _resolve_gpu_count(gpus: int | str | list[int]) -> int | None:
    """gpus 필드를 정수 GPU 수로 변환한다.

    "auto"이면 None을 반환 (판정 불가).
    """
    if isinstance(gpus, int):
        return gpus
    if isinstance(gpus, list):
        return len(gpus)
    if isinstance(gpus, str) and gpus.isdigit():
        return int(gpus)
    # "auto" 등 해석 불가한 문자열
    return None


class CompatValidator:
    """Config와 Recipe 간 호환성을 검증한다."""

    def validate(self, settings: Settings) -> ValidationResult:
        """검증 결과를 반환한다."""
        result = ValidationResult()

        self._check_gpu_distributed(settings, result)
        self._check_serving_backend(settings, result)
        self._check_fsdp_qlora(settings, result)

        return result

    # ── 개별 검증 ──

    @staticmethod
    def _check_gpu_distributed(settings: Settings, result: ValidationResult) -> None:
        """1. GPU 수와 분산 전략 정합성 검증."""
        compute = settings.config.compute
        gpu_count = _resolve_gpu_count(compute.gpus)

        if gpu_count is None:
            # "auto" 등 판정 불가 → 검증 생략
            return

        distributed = compute.distributed
        has_strategy = distributed is not None and distributed.get("strategy") is not None

        if gpu_count > 1 and not has_strategy:
            result.errors.append(
                f"GPU {gpu_count}개 환경이지만 분산 전략(strategy)이 "
                f"설정되지 않았습니다."
            )
        elif gpu_count == 1 and has_strategy:
            strategy = distributed["strategy"]  # type: ignore[index]
            result.warnings.append(
                f"GPU 1개 환경에서 분산 전략 '{strategy}'이(가) "
                f"설정되어 있습니다. 불필요할 수 있습니다."
            )

    @staticmethod
    def _check_serving_backend(settings: Settings, result: ValidationResult) -> None:
        """2. serving backend과 모델 호환성 검증."""
        serving = settings.config.serving
        if serving is None:
            return

        backend = serving.backend
        task = settings.recipe.task

        if backend == "vllm" and task not in ("text_generation", "seq2seq"):
            result.errors.append(
                f"vLLM backend은 text_generation 또는 seq2seq 태스크만 "
                f"지원합니다. 현재 태스크: '{task}'"
            )

    @staticmethod
    def _check_fsdp_qlora(settings: Settings, result: ValidationResult) -> None:
        """3. FSDP + QLoRA 비호환 검증."""
        adapter = settings.recipe.adapter
        if adapter is None:
            return

        is_qlora = adapter.method == "qlora" or (
            adapter.quantization is not None
            and adapter.quantization.get("bits") == 4
        )
        if not is_qlora:
            return

        distributed = settings.config.compute.distributed
        if distributed is None:
            return

        strategy = distributed.get("strategy", "")
        if strategy.lower() == "fsdp":
            result.errors.append(
                "FSDP와 QLoRA(4bit)는 호환되지 않습니다. "
                "대안: (1) LoRA 사용, (2) DDP 전략 사용, "
                "(3) DeepSpeed ZeRO-3 사용"
            )
