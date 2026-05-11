"""Neutral file layout descriptors for model artifacts and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping

EXPORT_INFO_FILE = "export_info.json"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_MODEL_SAFETENSORS_FILE = "adapter_model.safetensors"
ADAPTER_MODEL_BIN_FILE = "adapter_model.bin"
MODEL_SAFETENSORS_FILE = "model.safetensors"
MODEL_PT_FILE = "model.pt"
PYTORCH_MODEL_BIN_FILE = "pytorch_model.bin"
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
PYTORCH_MODEL_BIN_INDEX_FILE = "pytorch_model.bin.index.json"
CONFIG_FILE = "config.json"

WeightLayoutKind = Literal[
    "peft_adapter",
    "safetensors_module",
    "torch_state_dict",
    "hf_pretrained_dir",
    "custom_export",
    "missing",
]


@dataclass(frozen=True)
class WeightLayout:
    kind: WeightLayoutKind
    root: Path
    files: tuple[Path, ...]
    sidecars: Mapping[str, Path]
    is_sharded: bool = False


def detect_weight_layout(root: Path) -> WeightLayout:
    """Detect files present under ``root`` without deciding load policy."""
    if (root / EXPORT_INFO_FILE).exists():
        return WeightLayout(
            kind="custom_export",
            root=root,
            files=(root / EXPORT_INFO_FILE,),
            sidecars=_detect_sidecars(root),
        )

    adapter_files = _existing(
        root,
        ADAPTER_CONFIG_FILE,
        ADAPTER_MODEL_SAFETENSORS_FILE,
        ADAPTER_MODEL_BIN_FILE,
    )
    if adapter_files:
        return WeightLayout(
            kind="peft_adapter",
            root=root,
            files=adapter_files,
            sidecars=_detect_sidecars(root),
        )

    sharded = _detect_hf_sharded_layout(root)
    if sharded is not None:
        return sharded

    if _is_hf_unsharded_dir(root):
        return WeightLayout(
            kind="hf_pretrained_dir",
            root=root,
            files=_existing(root, MODEL_SAFETENSORS_FILE, PYTORCH_MODEL_BIN_FILE),
            sidecars=_detect_sidecars(root),
        )

    if (root / MODEL_SAFETENSORS_FILE).exists():
        return WeightLayout(
            kind="safetensors_module",
            root=root,
            files=(root / MODEL_SAFETENSORS_FILE,),
            sidecars=_detect_sidecars(root),
        )

    if (root / MODEL_PT_FILE).exists():
        return WeightLayout(
            kind="torch_state_dict",
            root=root,
            files=(root / MODEL_PT_FILE,),
            sidecars=_detect_sidecars(root),
        )

    return WeightLayout(
        kind="missing",
        root=root,
        files=(),
        sidecars=_detect_sidecars(root),
    )


def get_adapter_name(artifact_dir: Path) -> str:
    """Read PEFT adapter name from ``adapter_config.json`` or return default."""
    adapter_cfg_path = artifact_dir / ADAPTER_CONFIG_FILE
    if adapter_cfg_path.exists():
        try:
            return json.loads(adapter_cfg_path.read_text()).get(
                "adapter_name",
                "default",
            )
        except Exception:
            pass
    return "default"


def _detect_hf_sharded_layout(root: Path) -> WeightLayout | None:
    for index_name in (SAFETENSORS_INDEX_FILE, PYTORCH_MODEL_BIN_INDEX_FILE):
        index_path = root / index_name
        if not index_path.exists():
            continue
        shard_files = _read_index_shard_files(root, index_path)
        return WeightLayout(
            kind="hf_pretrained_dir",
            root=root,
            files=(index_path, *shard_files),
            sidecars=_detect_sidecars(root),
            is_sharded=True,
        )
    return None


def _read_index_shard_files(root: Path, index_path: Path) -> tuple[Path, ...]:
    try:
        index = json.loads(index_path.read_text())
    except Exception:
        return ()
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    return tuple(root / name for name in shard_names if (root / name).exists())


def _is_hf_unsharded_dir(root: Path) -> bool:
    return (root / CONFIG_FILE).exists() and (
        (root / MODEL_SAFETENSORS_FILE).exists()
        or (root / PYTORCH_MODEL_BIN_FILE).exists()
    )


def _existing(root: Path, *filenames: str) -> tuple[Path, ...]:
    return tuple(path for filename in filenames if (path := root / filename).exists())


def _detect_sidecars(root: Path) -> Mapping[str, Path]:
    names = [
        "recipe.yaml",
        CONFIG_FILE,
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    return {
        name: path
        for name in names
        if (path := root / name).exists()
    }
