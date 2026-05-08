"""Validated command execution plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mdp.settings.components import ComponentSpec
from mdp.settings.schema import Settings


Command = Literal[
    "train", "rl-train", "estimate", "inference", "generate", "serve", "export"
]
Mode = Literal["sft", "rl", "estimate", "inference", "serving", "export"]
ValidationScope = Literal["training", "recipe", "estimation", "artifact"]


@dataclass(frozen=True)
class RunSources:
    """Source files that produced the Settings for this command."""

    recipe_path: Path | None = None
    config_path: Path | None = None
    artifact_dir: Path | None = None


@dataclass(frozen=True)
class RunPlan:
    """Validated Settings plus command-level runtime intent."""

    command: Command
    mode: Mode
    settings: Settings
    sources: RunSources
    overrides: tuple[str, ...]
    callback_configs: tuple[ComponentSpec, ...]
    validation_scope: ValidationScope
    distributed_intent: bool
