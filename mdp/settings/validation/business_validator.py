"""BusinessValidator — 비즈니스 규칙 검증.

태스크-헤드 호환성, adapter-precision 호환성,
분산 학습 batch_size 검증 등 비즈니스 규칙을 검증한다.
"""

from __future__ import annotations

from mdp.settings.schema import Settings
from mdp.settings.validation import ValidationResult

# 태스크별 허용 Head 클래스명 매핑
TASK_HEAD_COMPAT: dict[str, list[str]] = {
    "image_classification": ["ClassificationHead"],
    "object_detection": ["DetectionHead"],
    "semantic_segmentation": ["SegmentationHead"],
    "text_classification": ["ClassificationHead"],
    "token_classification": ["TokenClassificationHead", "ClassificationHead"],
    "text_generation": ["CausalLMHead"],
    "seq2seq": ["Seq2SeqLMHead"],
    "image_generation": [],  # head 생략 권장
    "vision_language": ["DualEncoderHead", "CausalLMHead"],
}


def _extract_class_name(component_path: str) -> str:
    """_component_ 경로에서 클래스명(마지막 세그먼트)을 추출한다."""
    return component_path.rsplit(".", 1)[-1]


class BusinessValidator:
    """태스크-모델 호환성 등 비즈니스 규칙을 검증한다."""

    def validate(self, settings: Settings) -> ValidationResult:
        """검증 결과를 반환한다."""
        result = ValidationResult()

        self._check_head_task_compat(settings, result)
        self._check_adapter_precision(settings, result)
        self._check_distributed_batch(settings, result)

        return result

    # ── 개별 검증 ──

    @staticmethod
    def _check_head_task_compat(settings: Settings, result: ValidationResult) -> None:
        """1. head-task 호환성 검증."""
        recipe = settings.recipe
        task = recipe.task
        head = recipe.head

        if head is None:
            # head가 없으면 검증 대상 없음
            return

        component_path = head.get("_component_")
        if component_path is None:
            return

        head_class = _extract_class_name(component_path)
        allowed = TASK_HEAD_COMPAT.get(task)

        if allowed is not None:
            # 빈 리스트 = head 생략 권장
            if len(allowed) == 0:
                result.warnings.append(
                    f"태스크 '{task}'은(는) head 생략을 권장합니다. "
                    f"현재 head: '{head_class}'"
                )
            elif head_class not in allowed:
                result.errors.append(
                    f"태스크 '{task}'에 호환되지 않는 head '{head_class}'입니다. "
                    f"허용 head: {allowed}"
                )

        # 1b. head 교체 + adapter 사용 시 modules_to_save 경고
        adapter = recipe.adapter
        if adapter is not None and not adapter.modules_to_save:
            result.warnings.append(
                "head와 adapter를 함께 사용하지만 modules_to_save가 "
                "비어 있습니다. head 파라미터가 학습되지 않을 수 있습니다."
            )

    @staticmethod
    def _check_adapter_precision(
        settings: Settings, result: ValidationResult
    ) -> None:
        """2. adapter-precision 호환성 검증."""
        adapter = settings.recipe.adapter
        if adapter is None:
            return

        precision = settings.recipe.training.precision

        # QLoRA(4bit) + fp32 → warning
        is_qlora = adapter.method == "qlora" or (
            adapter.quantization is not None
            and adapter.quantization.get("bits") == 4
        )
        if is_qlora and precision == "fp32":
            result.warnings.append(
                "QLoRA(4bit)와 fp32 precision 조합은 비효율적입니다. "
                "bf16 사용을 권장합니다."
            )

    @staticmethod
    def _check_distributed_batch(
        settings: Settings, result: ValidationResult
    ) -> None:
        """3. 분산 학습 시 batch_size 검증."""
        distributed = settings.config.compute.distributed
        if distributed is None:
            return

        drop_last = settings.recipe.data.dataloader.drop_last
        if not drop_last:
            result.warnings.append(
                "분산 학습 시 drop_last=false이면 GPU별 배치 크기가 "
                "불균등해질 수 있습니다. drop_last=true를 권장합니다."
            )
