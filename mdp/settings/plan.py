"""Validated settings execution plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mdp.settings.schema import Settings


Command = Literal[
    "train", "rl-train", "estimate", "inference", "generate", "serve", "export"
]
Mode = Literal["sft", "rl", "estimate", "inference", "serving", "export"]
ValidationScope = Literal["training", "recipe", "estimation", "artifact"]


@dataclass(frozen=True)
class SettingsPlan:
    """Validated Settings plus runtime intent metadata."""

    command: Command
    mode: Mode
    settings: Settings
    recipe_path: Path | None
    config_path: Path | None
    artifact_dir: Path | None
    overrides: tuple[str, ...]
    callback_configs: tuple[dict[str, Any], ...]
    validation_scope: ValidationScope
    distributed_intent: bool
