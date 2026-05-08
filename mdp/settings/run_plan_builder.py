"""RunPlan builder for command-level runtime intent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mdp.settings.components import ComponentSpec
from mdp.settings.distributed import has_distributed_intent
from mdp.settings.loader import SettingsLoader
from mdp.settings.run_plan import (
    Command,
    Mode,
    RunPlan,
    RunSources,
    ValidationScope,
)
from mdp.settings.schema import Settings


class RunPlanBuilder:
    """Combine loaded Settings with command intent and runtime metadata."""

    def __init__(self, settings_loader: SettingsLoader | None = None) -> None:
        self._loader = settings_loader or SettingsLoader()

    def training(
        self,
        recipe_path: str | Path,
        config_path: str | Path,
        overrides: list[str] | None = None,
        callbacks_file: str | Path | None = None,
        command: Command = "train",
    ) -> RunPlan:
        recipe_source = Path(recipe_path)
        config_source = Path(config_path)
        settings = self._loader.load_training_settings(
            recipe_source,
            config_source,
            overrides=overrides,
        )
        return self._plan(
            command=command,
            mode=self._training_mode(settings, command),
            settings=settings,
            sources=RunSources(recipe_path=recipe_source, config_path=config_source),
            overrides=overrides,
            callbacks_file=callbacks_file,
            validation_scope="training",
        )

    def estimation(
        self,
        recipe_path: str | Path,
        overrides: list[str] | None = None,
        command: Command = "estimate",
    ) -> RunPlan:
        recipe_source = Path(recipe_path)
        settings = self._loader.load_estimation_settings(
            recipe_source,
            overrides=overrides,
        )
        return self._plan(
            command=command,
            mode="estimate",
            settings=settings,
            sources=RunSources(recipe_path=recipe_source),
            overrides=overrides,
            callbacks_file=None,
            validation_scope="estimation",
        )

    def inference(
        self,
        recipe_path: str | Path,
        config_path: str | Path,
        overrides: list[str] | None = None,
        callbacks_file: str | Path | None = None,
        command: Command = "inference",
    ) -> RunPlan:
        recipe_source = Path(recipe_path)
        config_source = Path(config_path)
        settings = self._loader.load_inference_settings(
            recipe_source,
            config_source,
            overrides=overrides,
        )
        return self._plan(
            command=command,
            mode=self._inference_mode(command),
            settings=settings,
            sources=RunSources(recipe_path=recipe_source, config_path=config_source),
            overrides=overrides,
            callbacks_file=callbacks_file,
            validation_scope="recipe",
        )

    def artifact(
        self,
        artifact_dir: str | Path,
        overrides: list[str] | None = None,
        *,
        use_config_snapshot: bool = True,
        command: Command = "serve",
        callbacks_file: str | Path | None = None,
    ) -> RunPlan:
        artifact_path = Path(artifact_dir)
        settings = self._loader.load_artifact_settings(
            artifact_path,
            overrides=overrides,
            use_config_snapshot=use_config_snapshot,
        )
        recipe_path = artifact_path / "recipe.yaml"
        config_path = (
            self._loader.find_artifact_config_snapshot(artifact_path)
            if use_config_snapshot
            else None
        )
        return self._plan(
            command=command,
            mode=self._inference_mode(command),
            settings=settings,
            sources=RunSources(
                recipe_path=recipe_path,
                config_path=config_path,
                artifact_dir=artifact_path,
            ),
            overrides=overrides,
            callbacks_file=callbacks_file,
            validation_scope="artifact",
        )

    def _plan(
        self,
        *,
        command: Command,
        mode: Mode,
        settings: Settings,
        sources: RunSources,
        overrides: list[str] | None,
        callbacks_file: str | Path | None,
        validation_scope: ValidationScope,
    ) -> RunPlan:
        return RunPlan(
            command=command,
            mode=mode,
            settings=settings,
            sources=sources,
            overrides=tuple(overrides or ()),
            callback_configs=self.load_callback_configs(callbacks_file),
            validation_scope=validation_scope,
            distributed_intent=has_distributed_intent(settings),
        )

    @staticmethod
    def _training_mode(settings: Settings, command: Command) -> Mode:
        if command == "rl-train" or settings.recipe.rl is not None:
            return "rl"
        return "sft"

    @staticmethod
    def _inference_mode(command: Command) -> Mode:
        if command == "serve":
            return "serving"
        if command == "export":
            return "export"
        return "inference"

    @staticmethod
    def load_callback_configs(
        path: str | Path | None,
    ) -> tuple[ComponentSpec, ...]:
        if path is None:
            return ()

        raw = SettingsLoader.load_yaml(Path(path), root_type="list", allow_empty=True)
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ValueError(
                f"콜백 파일은 리스트여야 합니다 (실제: {type(raw).__name__}). "
                "예: [{_component_: EarlyStopping, patience: 3}]"
            )
        return normalize_callback_configs(raw)


def normalize_callback_configs(
    configs: Sequence[ComponentSpec | Mapping[str, Any]] | None,
) -> tuple[ComponentSpec, ...]:
    """Normalize callback config inputs into typed component envelopes."""
    if configs is None:
        return ()
    normalized: list[ComponentSpec] = []
    for index, config in enumerate(configs):
        if isinstance(config, ComponentSpec):
            normalized.append(config)
            continue
        if isinstance(config, Mapping):
            if "_component_" not in config:
                raise ValueError(
                    f"콜백 항목 [{index}]에 _component_ 키가 필요합니다: {config}"
                )
            normalized.append(
                ComponentSpec.from_yaml_dict(dict(config), path=f"callbacks[{index}]")
            )
            continue
        raise TypeError(
            "callback config must be ComponentSpec or mapping "
            f"(index {index}, actual: {type(config).__name__})"
        )
    return tuple(normalized)
