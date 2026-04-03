"""BusinessValidator — 비즈니스 규칙 검증.

태스크-헤드 호환성, adapter-precision 호환성,
분산 학습 batch_size 검증 등 비즈니스 규칙을 검증한다.
"""

from __future__ import annotations

from mdp.settings.schema import Settings
from mdp.settings.validation import ValidationResult, is_qlora

# 태스크별 허용 Head 클래스명 매핑
TASK_HEAD_COMPAT: dict[str, list[str]] = {
    "image_classification": ["ClassificationHead"],
    "object_detection": ["DetectionHead"],
    "semantic_segmentation": ["SegmentationHead"],
    "text_classification": ["ClassificationHead"],
    "token_classification": ["TokenClassificationHead", "ClassificationHead"],
    "text_generation": ["CausalLMHead"],
    "seq2seq": ["Seq2SeqLMHead", "CausalLMHead"],
    "image_generation": [],  # head 생략 권장
    "feature_extraction": ["ClassificationHead", "DualEncoderHead"],
}


def _extract_class_name(component_path: str) -> str:
    """_component_ 경로에서 클래스명(마지막 세그먼트)을 추출한다."""
    return component_path.rsplit(".", 1)[-1]


class BusinessValidator:
    """태스크-모델 호환성 등 비즈니스 규칙을 검증한다."""

    def validate(self, settings: Settings) -> ValidationResult:
        """검증 결과를 반환한다."""
        return self.validate_partial(settings)

    @classmethod
    def validate_partial(cls, settings: Settings, checks: list[str] | None = None) -> ValidationResult:
        """부분 검증. checks가 None이면 전체 실행."""
        result = ValidationResult()
        all_checks = {
            "head_task": cls._check_head_task_compat,
            "adapter": cls._check_adapter,
            "distributed_batch": cls._check_distributed_batch,
            "task_fields": cls._check_task_fields,
            "label_strategy_fields": cls._check_label_strategy_fields,
        }
        targets = checks or list(all_checks.keys())
        for name in targets:
            if name in all_checks:
                all_checks[name](settings, result)
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
    def _check_adapter(
        settings: Settings, result: ValidationResult
    ) -> None:
        """2. adapter 설정 검증."""
        adapter = settings.recipe.adapter
        if adapter is None:
            return

        # 2a. r 필수 (lora/qlora/prefix_tuning)
        if adapter.method in ("lora", "qlora", "prefix_tuning") and adapter.r is None:
            result.errors.append(
                f"adapter.method '{adapter.method}'에는 r(rank)이 필수입니다."
            )

        # 2b. QLoRA 전용 검증
        if is_qlora(adapter):
            # quantization.bits 필수
            if adapter.quantization is None or adapter.quantization.get("bits") is None:
                result.errors.append(
                    "QLoRA에는 adapter.quantization.bits가 필수입니다."
                )

            # torch_dtype 필수 (bf16 또는 fp16)
            torch_dtype = settings.recipe.model.torch_dtype
            if torch_dtype is None:
                result.errors.append(
                    "QLoRA에는 model.torch_dtype 지정이 필수입니다. "
                    "bfloat16 또는 float16을 사용하세요."
                )
            elif torch_dtype not in ("bfloat16", "float16"):
                result.errors.append(
                    f"QLoRA의 model.torch_dtype은 bfloat16 또는 float16이어야 합니다. "
                    f"현재: '{torch_dtype}'"
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

    @staticmethod
    def _check_task_fields(
        settings: Settings, result: ValidationResult
    ) -> None:
        """4. task-fields 정합성 검증."""
        from mdp.task_taxonomy import validate_task_fields

        recipe = settings.recipe
        data = recipe.data

        data_config = {
            "tokenizer": data.tokenizer is not None,
            "augmentation": data.augmentation is not None,
        }
        errors, warnings = validate_task_fields(recipe.task, data.fields, data_config)
        result.errors.extend(errors)
        result.warnings.extend(warnings)

        # fields에 text 역할이 없는데 tokenizer가 설정된 경우 경고
        text_roles = {"text", "target", "chosen", "rejected"}
        has_text_field = any(role in data.fields for role in text_roles)
        if not has_text_field and data.tokenizer is not None:
            result.warnings.append(
                "data.fields에 text 역할이 없어 tokenizer가 사용되지 않습니다. "
                "data.tokenizer 설정을 제거하면 불필요한 로딩을 방지할 수 있습니다."
            )

    @staticmethod
    def _check_label_strategy_fields(
        settings: Settings, result: ValidationResult
    ) -> None:
        """5. label_strategy-fields 정합성 검증."""
        strategy = settings.recipe.data.label_strategy
        fields = settings.recipe.data.fields
        declared = set(fields.keys()) if fields else set()

        REQUIRED_FIELDS: dict[str, list[str]] = {
            "preference": ["chosen", "rejected"],
            "seq2seq": ["text", "target"],
            "align": ["text", "token_labels"],
            "copy": ["text", "label"],
            "causal": ["text"],
            "none": [],
        }

        required = REQUIRED_FIELDS.get(strategy)
        if required is None:
            result.errors.append(
                f"알 수 없는 label_strategy: '{strategy}'. "
                f"허용: {list(REQUIRED_FIELDS.keys())}"
            )
            return

        for field in required:
            if field not in declared:
                result.errors.append(
                    f"label_strategy '{strategy}'에 필요한 fields.{field}가 "
                    f"선언되지 않았습니다."
                )
