"""CompatValidator — Config-Recipe 호환성 검증.

GPU 수-분산 전략 정합성, serving backend 호환성,
FSDP + QLoRA 비호환 등 Config와 Recipe 간 호환성을 검증한다.
"""

from __future__ import annotations

from mdp.settings.schema import Settings
from mdp.settings.distributed import get_strategy_name, is_deepspeed_strategy
from mdp.settings.validation import ValidationResult, is_qlora


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
        self._check_deepspeed_boundary(settings, result)
        self._check_serving_backend(settings, result)
        self._check_fsdp_qlora(settings, result)
        self._check_moe_distributed(settings, result)

        return result

    # ── 개별 검증 ──

    @staticmethod
    def _check_deepspeed_boundary(settings: Settings, result: ValidationResult) -> None:
        """DeepSpeed is currently fail-fast until engine ownership is integrated."""
        strategy = get_strategy_name(settings)
        if not is_deepspeed_strategy(strategy):
            return
        result.errors.append(
            "DeepSpeed strategy is not supported by the current Trainer/RLTrainer "
            "runtime contract. DeepSpeed engine backward/step/checkpoint ownership "
            "requires a separate engine-contract spec. Use DDP/FSDP for now."
        )

    @staticmethod
    def _check_gpu_distributed(settings: Settings, result: ValidationResult) -> None:
        """1. GPU 수와 분산 전략 정합성 검증."""
        compute = settings.config.compute
        gpu_count = _resolve_gpu_count(compute.gpus)

        if gpu_count is None:
            # "auto" 등 판정 불가 → 검증 생략
            return

        distributed = compute.distributed
        if distributed is None:
            has_strategy = False
            strategy = None
        else:
            # runtime(create_strategy)과 동일하게 기본값 "auto" 적용
            strategy = distributed.get("strategy", "auto")
            has_strategy = strategy is not None and strategy != "none"

        if gpu_count > 1 and not has_strategy:
            result.errors.append(
                f"GPU {gpu_count}개 환경이지만 분산 전략(strategy)이 "
                f"설정되지 않았습니다."
            )
        elif gpu_count == 1 and has_strategy:
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
        distributed = settings.config.compute.distributed
        if distributed is None:
            return

        strategy = distributed.get("strategy", "auto")
        strategy_name = strategy
        if isinstance(strategy, dict):
            strategy_name = strategy.get("_component_", "")
        if not (isinstance(strategy_name, str) and strategy_name.lower().startswith("fsdp")):
            return

        # SFT 최상위 adapter
        adapter = settings.recipe.adapter
        if adapter is not None and is_qlora(adapter):
            result.errors.append(
                "FSDP와 QLoRA(4bit)는 호환되지 않습니다. "
                "대안: (1) LoRA 사용, (2) DDP 전략 사용, "
                "(3) 별도 DeepSpeed engine-contract 구현 후 ZeRO-3 사용"
            )

        # RL per-model adapter
        if settings.recipe.rl is not None:
            for name, spec in settings.recipe.rl.models.items():
                adapter = spec.get("adapter")
                if adapter is not None and is_qlora(adapter):
                    result.errors.append(
                        f"FSDP와 rl.models.{name}.adapter의 QLoRA(4bit)는 호환되지 않습니다. "
                        "대안: (1) LoRA 사용, (2) DDP 전략 사용, "
                        "(3) 별도 DeepSpeed engine-contract 구현 후 ZeRO-3 사용"
                    )

    @staticmethod
    def _check_moe_distributed(settings: Settings, result: ValidationResult) -> None:
        """4. MoE Expert Parallelism 호환성 검증."""
        distributed = settings.config.compute.distributed
        if distributed is None:
            return
        moe_config = distributed.get("moe")
        if moe_config is None or not moe_config.get("enabled", False):
            return

        ep_size = moe_config.get("ep_size", moe_config.get("expert_parallel_size"))
        if ep_size is None:
            result.errors.append(
                "MoE EP에는 moe.ep_size(Expert Parallel degree)가 필수입니다."
            )
            return

        gpu_count = _resolve_gpu_count(settings.config.compute.gpus)
        if gpu_count is not None and gpu_count % ep_size != 0:
            result.errors.append(
                f"GPU 수({gpu_count})가 moe.ep_size({ep_size})의 배수가 아닙니다. "
                f"world_size = ep_size × dp_size 관계가 성립해야 합니다."
            )
