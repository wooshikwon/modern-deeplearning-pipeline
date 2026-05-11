"""Weight loading helpers driven by artifact layout descriptors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mdp.artifacts.layout import (
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


def load_weights_into_model(
    model: Any,
    weights_dir: Path,
    *,
    layout: WeightLayout | None = None,
    merge: bool = False,
    device_map: str | None = None,
    max_memory: dict[str, str] | None = None,
) -> Any:
    """Load weights described by ``WeightLayout`` into an already built model."""
    layout = layout or detect_weight_layout(weights_dir)

    if layout.kind == "custom_export":
        return _load_custom_export(model, weights_dir)
    if layout.kind == "peft_adapter":
        model = _load_peft_adapter_artifact(model, weights_dir)
        if merge and hasattr(model, "merge_and_unload"):
            logger.info("LoRA adapter 병합 중...")
            model = model.merge_and_unload()
        if device_map is not None:
            model = _dispatch_loaded_model(model, max_memory=max_memory)
        return model
    if device_map is not None:
        checkpoint = find_dispatch_checkpoint(weights_dir, layout=layout)
        if checkpoint is None:
            raise ValueError(
                "device_map이 지정되었지만 가중치 파일(model.safetensors/model.pt)이 없습니다: "
                f"{weights_dir}"
            )
        return dispatch_model_from_checkpoint(
            model,
            checkpoint,
            device_map,
            max_memory=max_memory,
        )
    if layout.kind == "safetensors_module":
        return _load_safetensors_module(model, weights_dir / MODEL_SAFETENSORS_FILE)
    if layout.kind == "torch_state_dict":
        return _load_torch_state_dict(model, weights_dir / MODEL_PT_FILE, strict=True)
    if layout.kind == "hf_pretrained_dir":
        return _load_hf_pretrained_dir(model, weights_dir, layout=layout)

    logger.warning("체크포인트에 모델 가중치 파일이 없습니다: %s", weights_dir)
    return model


def load_checkpoint_weights(model: Any, checkpoint_dir: Path) -> Any:
    """Load artifact/checkpoint weights without serving reconstruction policy."""
    return load_weights_into_model(model, checkpoint_dir)


def find_dispatch_checkpoint(
    artifact_dir: Path,
    *,
    layout: WeightLayout | None = None,
) -> str | None:
    """Return a checkpoint path accepted by accelerate dispatch helpers."""
    layout = layout or detect_weight_layout(artifact_dir)
    if layout.kind == "safetensors_module":
        return str(artifact_dir / MODEL_SAFETENSORS_FILE)
    if layout.kind == "torch_state_dict":
        return str(artifact_dir / MODEL_PT_FILE)
    if layout.kind == "hf_pretrained_dir":
        for filename in (
            MODEL_SAFETENSORS_FILE,
            PYTORCH_MODEL_BIN_FILE,
            SAFETENSORS_INDEX_FILE,
            PYTORCH_MODEL_BIN_INDEX_FILE,
        ):
            path = artifact_dir / filename
            if path.exists():
                return str(path)
    return None


def dispatch_model_from_checkpoint(
    model: Any,
    checkpoint: str,
    device_map: str,
    *,
    max_memory: dict[str, str] | None = None,
) -> Any:
    """Load and dispatch a checkpoint with accelerate."""
    try:
        from accelerate import load_checkpoint_and_dispatch
    except ImportError:
        raise ImportError(
            "device_map 사용에 accelerate가 필요합니다: pip install accelerate"
        )

    kwargs: dict[str, Any] = {}
    if max_memory is not None:
        kwargs["max_memory"] = {
            (int(k) if k.isdigit() else k): v for k, v in max_memory.items()
        }

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint,
        device_map=device_map,
        no_split_module_classes=getattr(model, "_no_split_modules", None),
        **kwargs,
    )
    logger.info(
        "모델 분산 배치 완료 (device_map=%s): %s",
        device_map,
        getattr(model, "hf_device_map", "N/A"),
    )
    return model


def _load_custom_export(model: Any, artifact_dir: Path) -> Any:
    target = getattr(model, "module", model)
    if hasattr(target, "load_from_export"):
        target.load_from_export(artifact_dir)
    else:
        logger.warning(
            "export_info.json 있지만 load_from_export() 없음 — "
            "BaseModel 서브클래스인지 확인하세요: %s",
            type(target).__name__,
        )
    return model


def _load_peft_adapter_artifact(model: Any, checkpoint_dir: Path) -> Any:
    target = getattr(model, "module", model)
    adapter_name = get_adapter_name(checkpoint_dir)
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


def _dispatch_loaded_model(
    model: Any,
    *,
    max_memory: dict[str, str] | None,
) -> Any:
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
    computed_map = infer_auto_device_map(
        model,
        no_split_module_classes=getattr(model, "_no_split_modules", None),
        **dm_kwargs,
    )
    model = dispatch_model(model, computed_map)
    logger.info(
        "adapter merge 후 분산 배치 완료: %s",
        getattr(model, "hf_device_map", "N/A"),
    )
    return model


def _load_safetensors_module(model: Any, path: Path) -> Any:
    from mdp.utils.safetensors import load_module

    load_module(getattr(model, "module", model), path)
    logger.info("모델 가중치 로드: %s", path)
    return model


def _load_torch_state_dict(model: Any, path: Path, *, strict: bool) -> Any:
    import torch

    target = getattr(model, "module", model)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    target.load_state_dict(state_dict, strict=strict)
    logger.info("모델 가중치 로드: %s", path)
    return model


def _load_hf_pretrained_dir(
    model: Any,
    artifact_dir: Path,
    *,
    layout: WeightLayout,
) -> Any:
    if layout.is_sharded:
        from transformers.modeling_utils import load_sharded_checkpoint

        target = getattr(model, "module", model)
        load_sharded_checkpoint(target, str(artifact_dir), strict=False)
        logger.info("HF sharded pretrained 가중치 로드: %s", artifact_dir)
        return model
    safetensors_path = artifact_dir / MODEL_SAFETENSORS_FILE
    if safetensors_path.exists():
        return _load_safetensors_module(model, safetensors_path)
    pytorch_bin_path = artifact_dir / PYTORCH_MODEL_BIN_FILE
    if pytorch_bin_path.exists():
        return _load_torch_state_dict(model, pytorch_bin_path, strict=False)
    logger.warning("HF pretrained directory에 로드 가능한 가중치 파일이 없습니다: %s", artifact_dir)
    return model
