"""모델 가중치 로딩 — inference, serve가 공유하는 공용 모듈."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any, Literal

from mdp.artifacts.layout import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_MODEL_BIN_FILE,
    ADAPTER_MODEL_SAFETENSORS_FILE,
    EXPORT_INFO_FILE,
    MODEL_PT_FILE,
    MODEL_SAFETENSORS_FILE,
    PYTORCH_MODEL_BIN_FILE,
    PYTORCH_MODEL_BIN_INDEX_FILE,
    SAFETENSORS_INDEX_FILE,
    WeightLayout,
    detect_weight_layout,
    get_adapter_name,
)

logger = logging.getLogger(__name__)

_CHECKPOINT_MANIFEST_FILE = "manifest.json"
AdapterPolicy = Literal[
    "preserve_recipe_adapter",
    "suppress_recipe_adapter",
    "load_peft_adapter_artifact",
]
LegacyPolicy = Literal["read_only"]


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


def _get_adapter_name(checkpoint_dir: Path) -> str:
    """adapter_config.json에서 adapter_name을 읽는다. 없으면 PEFT 기본값 "default"로 폴백.

    PEFT load_adapter()는 adapter_name이 필수 위치 인자이므로 반드시 명시해야 한다.
    adapter_config.json에 adapter_name 필드가 없으면 (PEFT가 저장하지 않는 경우)
    학습 시 적용된 이름인 "default"로 폴백한다.
    """
    return get_adapter_name(checkpoint_dir)


def _load_peft_adapter_artifact(model: Any, checkpoint_dir: Path) -> Any:
    target = getattr(model, "module", model)
    adapter_name = _get_adapter_name(checkpoint_dir)
    if hasattr(target, "load_adapter"):
        target.load_adapter(str(checkpoint_dir), adapter_name=adapter_name)
        logger.info(
            "LoRA adapter loaded from %s (adapter_name=%s)",
            checkpoint_dir,
            adapter_name,
        )
        return model

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "PEFT adapter artifact를 로드하려면 peft가 필요합니다: "
            "pip install peft"
        ) from exc

    loaded = PeftModel.from_pretrained(
        model,
        str(checkpoint_dir),
        adapter_name=adapter_name,
    )
    logger.info(
        "PEFT adapter artifact loaded from %s (adapter_name=%s)",
        checkpoint_dir,
        adapter_name,
    )
    return loaded


def load_checkpoint_weights(model: Any, checkpoint_dir: Path) -> Any:
    """checkpoint/ artifact에서 가중치를 로드한다. resume용 — adapter/safetensors/pt 3가지 분기."""
    import torch

    target = getattr(model, "module", model)

    adapter_config_path = checkpoint_dir / ADAPTER_CONFIG_FILE
    adapter_safetensors_path = checkpoint_dir / ADAPTER_MODEL_SAFETENSORS_FILE
    adapter_bin_path = checkpoint_dir / ADAPTER_MODEL_BIN_FILE
    safetensors_path = checkpoint_dir / MODEL_SAFETENSORS_FILE
    model_pt_path = checkpoint_dir / MODEL_PT_FILE
    pytorch_model_bin_path = checkpoint_dir / PYTORCH_MODEL_BIN_FILE

    if (
        adapter_config_path.exists()
        or adapter_safetensors_path.exists()
        or adapter_bin_path.exists()
    ):
        return _load_peft_adapter_artifact(model, checkpoint_dir)
    elif safetensors_path.exists():
        from mdp.utils.safetensors import load_module

        load_module(target, safetensors_path)
        logger.info("모델 가중치 로드: %s", safetensors_path)
        return model
    elif model_pt_path.exists():
        state_dict = torch.load(model_pt_path, map_location="cpu", weights_only=True)
        target.load_state_dict(state_dict)
        logger.info("모델 가중치 로드: %s", model_pt_path)
        return model
    elif pytorch_model_bin_path.exists():
        state_dict = torch.load(
            pytorch_model_bin_path,
            map_location="cpu",
            weights_only=True,
        )
        target.load_state_dict(state_dict, strict=False)
        logger.info("모델 가중치 로드: %s", pytorch_model_bin_path)
        return model
    else:
        logger.warning("체크포인트에 모델 가중치 파일이 없습니다: %s", checkpoint_dir)
        return model


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
    safetensors_path = artifact_dir / MODEL_SAFETENSORS_FILE
    model_pt_path = artifact_dir / MODEL_PT_FILE
    safetensors_index_path = artifact_dir / SAFETENSORS_INDEX_FILE
    pytorch_index_path = artifact_dir / PYTORCH_MODEL_BIN_INDEX_FILE
    if safetensors_path.exists():
        return str(safetensors_path)
    if model_pt_path.exists():
        return str(model_pt_path)
    if safetensors_index_path.exists():
        return str(safetensors_index_path)
    if pytorch_index_path.exists():
        return str(pytorch_index_path)
    return None


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

    # export_info.json이 있으면 BaseModel.export()가 생성한 커스텀 export artifact
    # (e.g., backbone/ + value_head.pt 분리 저장). BaseModel.load_from_export()에 위임.
    export_info_path = weights_dir / EXPORT_INFO_FILE

    # adapter_config.json이 있으면 PEFT adapter artifact
    adapter_config_path = weights_dir / ADAPTER_CONFIG_FILE
    adapter_safetensors = weights_dir / ADAPTER_MODEL_SAFETENSORS_FILE
    safetensors_path = weights_dir / MODEL_SAFETENSORS_FILE
    model_pt_path = weights_dir / MODEL_PT_FILE
    pytorch_model_bin_path = weights_dir / PYTORCH_MODEL_BIN_FILE

    if export_info_path.exists():
        target = getattr(model, "module", model)
        if hasattr(target, "load_from_export"):
            target.load_from_export(weights_dir)
        else:
            logger.warning(
                "export_info.json 있지만 load_from_export() 없음 — "
                "BaseModel 서브클래스인지 확인하세요: %s", type(target).__name__
            )
        # export는 이미 merge된 가중치이므로 merge_and_unload 불필요.
        # device_map은 추후 필요 시 추가.
    elif adapter_config_path.exists() or adapter_safetensors.exists():
        # adapter는 항상 CPU에서 먼저 로드 + merge
        model = load_checkpoint_weights(model, weights_dir)
        if merge and hasattr(model, "merge_and_unload"):
            logger.info("LoRA adapter 병합 중...")
            model = model.merge_and_unload()
        # device_map: merge 후 분산 배치는 아래에서 처리
        if device_map is not None:
            # merge된 모델의 가중치는 이미 메모리에 있으므로
            # dispatch_model로 재배치
            try:
                from accelerate import dispatch_model, infer_auto_device_map
            except ImportError:
                raise ImportError(
                    "device_map 사용에 accelerate가 필요합니다: pip install accelerate"
                )
            dm_kwargs: dict[str, Any] = {}
            if max_memory is not None:
                dm_kwargs["max_memory"] = {
                    (int(k) if k.isdigit() else k): v for k, v in max_memory.items()
                }
            no_split_classes = getattr(model, "_no_split_modules", None)
            computed_map = infer_auto_device_map(
                model,
                no_split_module_classes=no_split_classes,
                **dm_kwargs,
            )
            model = dispatch_model(model, computed_map)
            logger.info("adapter merge 후 분산 배치 완료: %s",
                        getattr(model, "hf_device_map", "N/A"))
    elif device_map is not None:
        # device_map 지정: accelerate로 가중치 분산 로딩
        checkpoint = _find_checkpoint_path(weights_dir)
        if checkpoint is None:
            raise ValueError(
                f"device_map이 지정되었지만 가중치 파일(model.safetensors/model.pt)이 없습니다: {weights_dir}"
            )
        else:
            model = _dispatch_model(model, checkpoint, device_map, max_memory)
    elif safetensors_path.exists() or model_pt_path.exists() or pytorch_model_bin_path.exists():
        model = load_checkpoint_weights(model, weights_dir)
    else:
        logger.warning("가중치 파일 없음: %s", weights_dir)

    return model, settings
