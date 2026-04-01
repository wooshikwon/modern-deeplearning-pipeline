"""task_taxonomy — task 검증 프리셋.

task는 모델이 무엇을 만들어내는가(출력)로 정의한다.
파이프라인 라우팅에는 관여하지 않으며, 검증·프리셋·메트릭 전용이다.
실제 파이프라인 동작은 data.fields + head + model이 결정한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPreset:
    """task별 검증 프리셋."""

    required_fields: frozenset[str]
    required_config: frozenset[str]
    default_head: str | None
    default_metric: str | None


TASK_PRESETS: dict[str, TaskPreset] = {
    "image_classification": TaskPreset(
        required_fields=frozenset({"image", "label"}),
        required_config=frozenset({"augmentation"}),
        default_head="ClassificationHead",
        default_metric="accuracy",
    ),
    "object_detection": TaskPreset(
        required_fields=frozenset({"image", "label"}),
        required_config=frozenset({"augmentation"}),
        default_head="DetectionHead",
        default_metric="mAP",
    ),
    "semantic_segmentation": TaskPreset(
        required_fields=frozenset({"image", "label"}),
        required_config=frozenset({"augmentation"}),
        default_head="SegmentationHead",
        default_metric="mIoU",
    ),
    "text_classification": TaskPreset(
        required_fields=frozenset({"text", "label"}),
        required_config=frozenset({"tokenizer"}),
        default_head="ClassificationHead",
        default_metric="accuracy",
    ),
    "token_classification": TaskPreset(
        required_fields=frozenset({"text", "token_labels"}),
        required_config=frozenset({"tokenizer"}),
        default_head="TokenClassificationHead",
        default_metric="f1",
    ),
    "text_generation": TaskPreset(
        required_fields=frozenset({"text"}),
        required_config=frozenset({"tokenizer"}),
        default_head="CausalLMHead",
        default_metric="perplexity",
    ),
    "seq2seq": TaskPreset(
        required_fields=frozenset({"text", "target"}),
        required_config=frozenset({"tokenizer"}),
        default_head="Seq2SeqLMHead",
        default_metric="bleu",
    ),
    "image_generation": TaskPreset(
        required_fields=frozenset({"image", "text"}),
        required_config=frozenset({"augmentation", "tokenizer"}),
        default_head=None,
        default_metric="fid",
    ),
    "feature_extraction": TaskPreset(
        required_fields=frozenset(),
        required_config=frozenset(),
        default_head=None,
        default_metric=None,
    ),
}


def validate_task_fields(
    task: str, fields: dict[str, str], data_config: dict[str, bool],
) -> list[str]:
    """task 선언과 fields/config의 일관성을 검증한다.

    Returns:
        경고 메시지 리스트 (빈 리스트이면 문제 없음).
    """
    preset = TASK_PRESETS.get(task)
    if preset is None:
        return [f"알 수 없는 task: '{task}'. 지원: {list(TASK_PRESETS.keys())}"]

    warnings = []
    declared_fields = set(fields.keys()) if fields else set()

    for req in preset.required_fields:
        if req not in declared_fields:
            warnings.append(
                f"task '{task}'에 필요한 fields.{req}가 선언되지 않았습니다."
            )

    for req in preset.required_config:
        if not data_config.get(req):
            warnings.append(
                f"task '{task}'에 필요한 data.{req} 설정이 없습니다."
            )

    return warnings
