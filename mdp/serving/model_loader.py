"""모델 가중치 로딩 — inference, serve가 공유하는 공용 모듈."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any, Literal

from mdp.artifacts.layout import (
    WeightLayout,
    detect_weight_layout,
)
from mdp.artifacts.loading import (
    find_dispatch_checkpoint,
    load_weights_into_model,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_MANIFEST_FILE = "manifest.json"
AdapterPolicy = Literal[
    "preserve_recipe_adapter",
    "suppress_recipe_adapter",
    "load_peft_adapter_artifact",
]
LegacyPolicy = Literal["read_only"]


def load_checkpoint_weights(model: Any, checkpoint_dir: Path) -> Any:
    """Backward-compatible public import path for artifact weight loading."""
    from mdp.artifacts.loading import load_checkpoint_weights as _load_weights

    return _load_weights(model, checkpoint_dir)


@dataclass(frozen=True)
class ArtifactLoadPlan:
    source: Path
    artifact_kind: Literal[
        "training_checkpoint",
        "serving_artifact",
        "legacy_checkpoint",
    ]
    role: str = "policy"
    weight_format: str | None = None
    merge: bool = False
    weights_dir: Path | None = None
    adapter_policy: AdapterPolicy = "preserve_recipe_adapter"
    legacy_policy: LegacyPolicy | None = None


def resolve_artifact_load_plan(
    source: Path,
    *,
    role: str = "policy",
    merge: bool = False,
) -> ArtifactLoadPlan:
    """Classify model source and select the role-specific weight directory."""
    layout = detect_weight_layout(source)
    manifest_path = source / _CHECKPOINT_MANIFEST_FILE
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        models = manifest.get("models", {})
        if models:
            selected_name, selected = _select_manifest_model(models, role)
            record_path = Path(str(selected.get("path", ".")))
            weights_dir = source / record_path.parent
            if str(record_path.parent) == ".":
                weights_dir = source
            logger.info(
                "manifest checkpoint selected model slot=%s role=%s format=%s",
                selected_name,
                selected.get("role"),
                selected.get("format"),
            )
            weight_format = selected.get("format")
            return ArtifactLoadPlan(
                source=source,
                artifact_kind="training_checkpoint",
                role=str(selected.get("role", role)),
                weight_format=weight_format,
                merge=merge,
                weights_dir=weights_dir,
                adapter_policy=_adapter_policy_for_artifact(
                    "training_checkpoint",
                    weight_format,
                ),
            )

    if (source / "recipe.yaml").exists() and layout.kind != "missing":
        weight_format = _weight_format_from_layout(layout)
        return ArtifactLoadPlan(
            source=source,
            artifact_kind="serving_artifact",
            role=role,
            weight_format=weight_format,
            merge=merge,
            weights_dir=source,
            adapter_policy=_adapter_policy_for_artifact(
                "serving_artifact",
                weight_format,
            ),
        )

    weight_format = _weight_format_from_layout(layout)
    return ArtifactLoadPlan(
        source=source,
        artifact_kind="legacy_checkpoint",
        role=role,
        weight_format=weight_format,
        merge=merge,
        weights_dir=source,
        adapter_policy=_adapter_policy_for_artifact(
            "legacy_checkpoint",
            weight_format,
        ),
        legacy_policy="read_only",
    )


def _select_manifest_model(
    models: dict[str, Any],
    role: str,
) -> tuple[str, dict[str, Any]]:
    for name, record in models.items():
        if record.get("role") == role:
            return name, record
    if role in models:
        return role, models[role]
    available = {
        name: record.get("role")
        for name, record in models.items()
    }
    raise ValueError(
        f"manifest checkpoint에 role={role!r} 모델이 없습니다. available={available}"
    )


def _detect_weight_format(source: Path) -> str | None:
    return _weight_format_from_layout(detect_weight_layout(source))


def _weight_format_from_layout(layout: WeightLayout) -> str | None:
    if layout.kind == "custom_export":
        return "export"
    if layout.kind == "peft_adapter":
        return "peft_adapter"
    if layout.kind == "safetensors_module":
        return "safetensors"
    if layout.kind == "torch_state_dict":
        return "torch_state_dict"
    if layout.kind == "hf_pretrained_dir":
        return "hf_pretrained_dir"
    return None


def _adapter_policy_for_artifact(
    artifact_kind: Literal[
        "training_checkpoint",
        "serving_artifact",
        "legacy_checkpoint",
    ],
    weight_format: str | None,
) -> AdapterPolicy:
    if weight_format == "peft_adapter":
        return "load_peft_adapter_artifact"
    if (
        artifact_kind == "serving_artifact"
        and weight_format in {
            "export",
            "safetensors",
            "torch_state_dict",
            "hf_pretrained_dir",
        }
    ):
        return "suppress_recipe_adapter"
    return "preserve_recipe_adapter"


def _apply_artifact_adapter_policy(assembly_plan: Any, plan: ArtifactLoadPlan) -> Any:
    """Adjust recipe adapter materialization to match artifact weight ownership."""
    if plan.adapter_policy == "preserve_recipe_adapter":
        return assembly_plan
    models = tuple(
        replace(node, adapter=None) if node.adapter is not None else node
        for node in assembly_plan.models
    )
    if models == assembly_plan.models:
        return assembly_plan
    return replace(assembly_plan, models=models)


def _resolve_padding_side(model: Any) -> str:
    """모델 아키텍처에 맞는 padding_side를 반환한다.

    decoder-only (LLaMA, GPT, Mistral 등): 'left'
      → generate()는 마지막 토큰에서 이어서 생성하므로 PAD가 왼쪽에 있어야 한다.
    encoder-decoder (T5, BART 등): 'right'
      → encoder가 전체 시퀀스를 처리하므로 오른쪽 패딩이 자연스럽다.
    """
    cfg = getattr(model, "config", None)
    is_enc_dec = getattr(cfg, "is_encoder_decoder", False)
    return "right" if is_enc_dec else "left"


def _find_checkpoint_path(artifact_dir: Path) -> str | None:
    """device_map 로딩에 사용할 체크포인트 경로를 찾는다."""
    return find_dispatch_checkpoint(artifact_dir)


def _dispatch_model(
    model: Any,
    checkpoint: str,
    device_map: str,
    max_memory: dict[str, str] | None = None,
) -> Any:
    """accelerate로 모델 가중치를 여러 GPU에 분산 로딩한다."""
    try:
        from accelerate import load_checkpoint_and_dispatch
    except ImportError:
        raise ImportError(
            "device_map 사용에 accelerate가 필요합니다: pip install accelerate"
        )

    no_split_classes = getattr(model, "_no_split_modules", None)

    kwargs: dict[str, Any] = {}
    if max_memory is not None:
        kwargs["max_memory"] = {
            (int(k) if k.isdigit() else k): v for k, v in max_memory.items()
        }

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint,
        device_map=device_map,
        no_split_module_classes=no_split_classes,
        **kwargs,
    )
    logger.info("모델 분산 배치 완료 (device_map=%s): %s",
                device_map, getattr(model, "hf_device_map", "N/A"))
    return model


def reconstruct_model(
    artifact_dir: Path,
    merge: bool = False,
    role: str = "policy",
    device_map: str | None = None,
    max_memory: dict[str, str] | None = None,
    overrides: list[str] | None = None,
) -> tuple[Any, Any]:
    """artifact 디렉토리의 recipe.yaml로 모델을 재구성하고 가중치를 로드한다.

    model/ artifact와 checkpoint/ artifact 모두 recipe.yaml을 가지고 있으므로
    양쪽에서 사용 가능하다.

    Args:
        artifact_dir: artifact 또는 checkpoint 디렉토리.
        merge: True이면 adapter를 base model에 merge한다 (export/serve용).
        role: manifest checkpoint에서 로드할 모델 role. RL serving/export 기본값은 policy.
        device_map: "auto", "balanced", "sequential" 등. 지정 시 accelerate로 분산 배치.
        max_memory: GPU별 최대 메모리. {"0": "24GiB", "1": "40GiB"}.
        overrides: Recipe/Config 오버라이드 (dotted KEY=VALUE).

    Returns:
        (model, settings) 튜플.
    """
    from mdp.assembly.materializer import AssemblyMaterializer
    from mdp.assembly.planner import AssemblyPlanner
    from mdp.settings.run_plan_builder import RunPlanBuilder

    plan = resolve_artifact_load_plan(artifact_dir, role=role, merge=merge)
    weights_dir = plan.weights_dir or artifact_dir

    run_plan = RunPlanBuilder().artifact(
        str(artifact_dir),
        overrides=overrides,
        command="serve",
    )
    settings = run_plan.settings
    # RL recipe는 top-level `model` 섹션에 pretrained가 없고 `rl.models.policy`에 있다.
    # create_model()은 recipe.model만 보므로 RL recipe에서 크래시한다.
    # RL recipe이면 create_models()["policy"]를 사용한다.
    # default role은 policy — value/critic 모델은 훈련 전용이며 serving 경로에서는
    # 필요하지 않다. manifest checkpoint에서는 caller가 role을 명시할 수 있다.
    assembly_plan = AssemblyPlanner.from_run_plan(run_plan)
    assembly_plan = _apply_artifact_adapter_policy(assembly_plan, plan)
    materializer = AssemblyMaterializer(assembly_plan)
    if settings.recipe.rl is not None:
        models = materializer.materialize_models(skip_base_check=True)
        model = (
            models.get(plan.role)
            or models.get("policy")
            or next(iter(models.values()))
        )
    else:
        model = materializer.materialize_policy_model(skip_base_check=True)

    layout = detect_weight_layout(weights_dir)
    model = load_weights_into_model(
        model,
        weights_dir,
        layout=layout,
        merge=merge,
        device_map=device_map,
        max_memory=max_memory,
    )

    return model, settings
