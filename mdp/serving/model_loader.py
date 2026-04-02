"""모델 가중치 로딩 — inference, serve가 공유하는 공용 모듈."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_serving_model(model_dir: Path) -> None:
    """model/ artifact에서 가중치를 로드한다. merge 완료 상태이므로 safetensors만 처리."""
    from safetensors.torch import load_file

    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors를 찾을 수 없습니다: {model_dir}")

    return load_file(safetensors_path)


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


def reconstruct_model(artifact_dir: Path, merge: bool = False) -> tuple[Any, Any]:
    """artifact 디렉토리의 recipe.yaml로 모델을 재구성하고 가중치를 로드한다.

    model/ artifact와 checkpoint/ artifact 모두 recipe.yaml을 가지고 있으므로
    양쪽에서 사용 가능하다.

    Args:
        artifact_dir: artifact 또는 checkpoint 디렉토리.
        merge: True이면 adapter를 base model에 merge한다 (export/serve용).

    Returns:
        (model, settings) 튜플.
    """
    from mdp.factory.factory import Factory
    from mdp.settings.factory import SettingsFactory

    settings = SettingsFactory().from_artifact(str(artifact_dir))
    model = Factory(settings).create_model()

    # adapter_config.json이 있으면 PEFT adapter artifact
    adapter_config_path = artifact_dir / "adapter_config.json"
    adapter_safetensors = artifact_dir / "adapter_model.safetensors"
    safetensors_path = artifact_dir / "model.safetensors"

    if adapter_config_path.exists() or adapter_safetensors.exists():
        load_checkpoint_weights(model, artifact_dir)
        if merge and hasattr(model, "merge_and_unload"):
            logger.info("LoRA adapter 병합 중...")
            model = model.merge_and_unload()
    elif safetensors_path.exists():
        from safetensors.torch import load_file

        target = getattr(model, "module", model)
        state_dict = load_file(safetensors_path)
        target.load_state_dict(state_dict)
    else:
        logger.warning("가중치 파일 없음: %s", artifact_dir)

    return model, settings
