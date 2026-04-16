"""모델 가중치 로딩 — inference, serve가 공유하는 공용 모듈."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_checkpoint_weights(model: Any, checkpoint_dir: Path) -> None:
    """checkpoint/ artifact에서 가중치를 로드한다. resume용 — adapter/safetensors/pt 3가지 분기."""
    import torch

    target = getattr(model, "module", model)

    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    safetensors_path = checkpoint_dir / "model.safetensors"
    model_pt_path = checkpoint_dir / "model.pt"

    if adapter_path.exists():
        if hasattr(target, "load_adapter"):
            target.load_adapter(str(checkpoint_dir))
            logger.info("LoRA adapter loaded from %s", checkpoint_dir)
        else:
            logger.warning("adapter_model.safetensors가 있지만 load_adapter 메서드 없음")
    elif safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path)
        target.load_state_dict(state_dict)
        logger.info("모델 가중치 로드: %s", safetensors_path)
    elif model_pt_path.exists():
        state_dict = torch.load(model_pt_path, map_location="cpu", weights_only=True)
        target.load_state_dict(state_dict)
        logger.info("모델 가중치 로드: %s", model_pt_path)
    else:
        logger.warning("체크포인트에 모델 가중치 파일이 없습니다: %s", checkpoint_dir)


def _find_checkpoint_path(artifact_dir: Path) -> str | None:
    """device_map 로딩에 사용할 체크포인트 경로를 찾는다."""
    safetensors_path = artifact_dir / "model.safetensors"
    model_pt_path = artifact_dir / "model.pt"
    if safetensors_path.exists():
        return str(safetensors_path)
    if model_pt_path.exists():
        return str(model_pt_path)
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
        device_map: "auto", "balanced", "sequential" 등. 지정 시 accelerate로 분산 배치.
        max_memory: GPU별 최대 메모리. {"0": "24GiB", "1": "40GiB"}.
        overrides: Recipe/Config 오버라이드 (dotted KEY=VALUE).

    Returns:
        (model, settings) 튜플.
    """
    from mdp.factory.factory import Factory
    from mdp.settings.factory import SettingsFactory

    settings = SettingsFactory().from_artifact(str(artifact_dir), overrides=overrides)
    # RL recipe는 top-level `model` 섹션에 pretrained가 없고 `rl.models.policy`에 있다.
    # create_model()은 recipe.model만 보므로 RL recipe에서 크래시한다.
    # RL recipe이면 create_models()["policy"]를 사용한다.
    factory = Factory(settings)
    if settings.recipe.rl is not None:
        models = factory.create_models(skip_base_check=True)
        model = models.get("policy") or next(iter(models.values()))
    else:
        model = factory.create_model(skip_base_check=True)

    # export_info.json이 있으면 BaseModel.export()가 생성한 커스텀 export artifact
    # (e.g., backbone/ + value_head.pt 분리 저장). BaseModel.load_from_export()에 위임.
    export_info_path = artifact_dir / "export_info.json"

    # adapter_config.json이 있으면 PEFT adapter artifact
    adapter_config_path = artifact_dir / "adapter_config.json"
    adapter_safetensors = artifact_dir / "adapter_model.safetensors"
    safetensors_path = artifact_dir / "model.safetensors"

    if export_info_path.exists():
        target = getattr(model, "module", model)
        if hasattr(target, "load_from_export"):
            target.load_from_export(artifact_dir)
        else:
            logger.warning(
                "export_info.json 있지만 load_from_export() 없음 — "
                "BaseModel 서브클래스인지 확인하세요: %s", type(target).__name__
            )
        # export는 이미 merge된 가중치이므로 merge_and_unload 불필요.
        # device_map은 추후 필요 시 추가.
    elif adapter_config_path.exists() or adapter_safetensors.exists():
        # adapter는 항상 CPU에서 먼저 로드 + merge
        load_checkpoint_weights(model, artifact_dir)
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
        checkpoint = _find_checkpoint_path(artifact_dir)
        if checkpoint is None:
            raise ValueError(
                f"device_map이 지정되었지만 가중치 파일(model.safetensors/model.pt)이 없습니다: {artifact_dir}"
            )
        else:
            model = _dispatch_model(model, checkpoint, device_map, max_memory)
    elif safetensors_path.exists():
        from safetensors.torch import load_file

        target = getattr(model, "module", model)
        state_dict = load_file(safetensors_path)
        target.load_state_dict(state_dict)
    else:
        logger.warning("가중치 파일 없음: %s", artifact_dir)

    return model, settings
