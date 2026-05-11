"""Artifact layout descriptors shared by training and serving boundaries."""

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
from mdp.artifacts.serving import (
    ServingArtifactManager,
    ServingArtifactMode,
    ServingArtifactRecord,
)
from mdp.artifacts.loading import (
    dispatch_model_from_checkpoint,
    find_dispatch_checkpoint,
    load_checkpoint_weights,
    load_weights_into_model,
)

__all__ = [
    "ADAPTER_CONFIG_FILE",
    "ADAPTER_MODEL_BIN_FILE",
    "ADAPTER_MODEL_SAFETENSORS_FILE",
    "EXPORT_INFO_FILE",
    "MODEL_PT_FILE",
    "MODEL_SAFETENSORS_FILE",
    "PYTORCH_MODEL_BIN_FILE",
    "PYTORCH_MODEL_BIN_INDEX_FILE",
    "SAFETENSORS_INDEX_FILE",
    "WeightLayout",
    "detect_weight_layout",
    "get_adapter_name",
    "ServingArtifactManager",
    "ServingArtifactMode",
    "ServingArtifactRecord",
    "dispatch_model_from_checkpoint",
    "find_dispatch_checkpoint",
    "load_checkpoint_weights",
    "load_weights_into_model",
]
