"""SettingsPlanner — YAML/artifact sources to SettingsPlan."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from mdp.settings.distributed import has_distributed_intent
from mdp.settings.plan import Command, Mode, SettingsPlan, ValidationScope
from mdp.settings.schema import Config, Recipe, Settings

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


class SettingsPlanner:
    """Build validated SettingsPlan objects from recipes, configs, and artifacts."""

    def load_training(
        self,
        recipe_path: str | Path,
        config_path: str | Path,
        overrides: list[str] | None = None,
        callbacks_file: str | Path | None = None,
        command: Command = "train",
    ) -> SettingsPlan:
        recipe_source = Path(recipe_path)
        config_source = Path(config_path)
        settings = self._build_settings(
            recipe_source,
            config_source,
            overrides=overrides,
            validation_scope="training",
        )
        return self._plan(
            command=command,
            mode=self._training_mode(settings, command),
            settings=settings,
            recipe_path=recipe_source,
            config_path=config_source,
            artifact_dir=None,
            overrides=overrides,
            callbacks_file=callbacks_file,
            validation_scope="training",
        )

    def load_estimation(
        self,
        recipe_path: str | Path,
        overrides: list[str] | None = None,
        command: Command = "estimate",
    ) -> SettingsPlan:
        recipe_source = Path(recipe_path)
        settings = self._build_settings(
            recipe_source,
            None,
            overrides=overrides,
            validation_scope="estimation",
        )
        return self._plan(
            command=command,
            mode="estimate",
            settings=settings,
            recipe_path=recipe_source,
            config_path=None,
            artifact_dir=None,
            overrides=overrides,
            callbacks_file=None,
            validation_scope="estimation",
        )

    def load_inference(
        self,
        recipe_path: str | Path,
        config_path: str | Path,
        overrides: list[str] | None = None,
        callbacks_file: str | Path | None = None,
        command: Command = "inference",
    ) -> SettingsPlan:
        recipe_source = Path(recipe_path)
        config_source = Path(config_path)
        settings = self._build_settings(
            recipe_source,
            config_source,
            overrides=overrides,
            validation_scope="recipe",
        )
        return self._plan(
            command=command,
            mode=self._inference_mode(command),
            settings=settings,
            recipe_path=recipe_source,
            config_path=config_source,
            artifact_dir=None,
            overrides=overrides,
            callbacks_file=callbacks_file,
            validation_scope="recipe",
        )

    def load_artifact(
        self,
        artifact_dir: str | Path,
        overrides: list[str] | None = None,
        *,
        use_config_snapshot: bool = True,
        command: Command = "serve",
        callbacks_file: str | Path | None = None,
    ) -> SettingsPlan:
        artifact_path = Path(artifact_dir)
        recipe_path = artifact_path / "recipe.yaml"
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"artifact에 recipe.yaml이 없습니다: {artifact_dir}\n"
                "이 artifact는 recipe 내장 이전에 생성되었을 수 있습니다."
            )

        config_path: Path | None = None
        if use_config_snapshot:
            config_path = self.find_artifact_config_snapshot(artifact_path)

        settings = self._build_settings(
            recipe_path,
            config_path,
            overrides=overrides,
            validation_scope="artifact",
        )
        return self._plan(
            command=command,
            mode=self._inference_mode(command),
            settings=settings,
            recipe_path=recipe_path,
            config_path=config_path,
            artifact_dir=artifact_path,
            overrides=overrides,
            callbacks_file=callbacks_file,
            validation_scope="artifact",
        )

    def _build_settings(
        self,
        recipe_path: Path,
        config_path: Path | None,
        *,
        overrides: list[str] | None,
        validation_scope: ValidationScope,
    ) -> Settings:
        recipe_dict = self._substitute_env_vars(self._load_yaml(recipe_path))
        config_dict = (
            self._substitute_env_vars(self._load_yaml(config_path))
            if config_path is not None
            else {}
        )

        if overrides:
            recipe_dict, config_dict = self._split_and_apply_overrides(
                recipe_dict, config_dict, overrides,
            )

        recipe = Recipe(**recipe_dict)
        config = Config(**config_dict) if config_dict else Config()
        settings = Settings(recipe=recipe, config=config)
        self._validate_for_scope(settings, validation_scope)
        return settings

    def _plan(
        self,
        *,
        command: Command,
        mode: Mode,
        settings: Settings,
        recipe_path: Path | None,
        config_path: Path | None,
        artifact_dir: Path | None,
        overrides: list[str] | None,
        callbacks_file: str | Path | None,
        validation_scope: ValidationScope,
    ) -> SettingsPlan:
        callback_configs = self._load_callback_configs(callbacks_file)
        return SettingsPlan(
            command=command,
            mode=mode,
            settings=settings,
            recipe_path=recipe_path,
            config_path=config_path,
            artifact_dir=artifact_dir,
            overrides=tuple(overrides or ()),
            callback_configs=tuple(callback_configs),
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
    def find_artifact_config_snapshot(artifact_dir: Path) -> Path | None:
        """Find artifact config snapshot from manifest or standard filenames."""
        manifest_path = artifact_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                manifest = {}
            config_file = manifest.get("config_file")
            if isinstance(config_file, str) and config_file:
                candidate = artifact_dir / config_file
                if candidate.exists():
                    return candidate

        for name in ("config.yaml", "config.yml"):
            candidate = artifact_dir / name
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_callback_configs(path: str | Path | None) -> list[dict[str, Any]]:
        if path is None:
            return []

        raw = SettingsPlanner._load_yaml(Path(path))
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError(
                f"콜백 파일은 리스트여야 합니다 (실제: {type(raw).__name__}). "
                "예: [{_component_: EarlyStopping, patience: 3}]"
            )
        for i, item in enumerate(raw):
            if not isinstance(item, dict) or "_component_" not in item:
                raise ValueError(
                    f"콜백 항목 [{i}]에 _component_ 키가 필요합니다: {item}"
                )
        return [dict(item) for item in raw]

    @staticmethod
    def _split_and_apply_overrides(
        recipe_dict: dict[str, Any],
        config_dict: dict[str, Any],
        overrides: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split overrides into recipe/config namespaces and apply them."""
        from mdp.cli._override import apply_overrides

        config_top_keys = {
            "environment", "compute",
            "mlflow", "storage", "serving", "job",
        }

        recipe_ovr: list[str] = []
        config_ovr: list[str] = []
        for override in overrides:
            key = override.partition("=")[0]
            if key.startswith("config."):
                config_ovr.append(override[len("config."):])
            else:
                recipe_ovr.append(override)

        for override in recipe_ovr:
            top_key = override.partition("=")[0].split(".")[0]
            if top_key in config_top_keys and top_key not in recipe_dict:
                logger.warning(
                    "Override '%s'의 키 '%s'는 Config 필드입니다. "
                    "'config.%s'를 사용하세요.",
                    override, top_key, override,
                )

        if recipe_ovr:
            apply_overrides(recipe_dict, recipe_ovr)
        if config_ovr:
            apply_overrides(config_dict, config_ovr)
        return recipe_dict, config_dict

    @staticmethod
    def _load_yaml(path: str | Path) -> Any:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def _substitute_env_vars(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls._substitute_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._substitute_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return cls._substitute_string(obj)
        return obj

    @classmethod
    def _substitute_string(cls, s: str) -> Any:
        match = ENV_PATTERN.fullmatch(s)
        if match:
            var_name, default = match.group(1), match.group(2)
            value = os.environ.get(var_name)
            if value is None:
                if default is not None:
                    return cls._auto_cast(default)
                raise ValueError(
                    f"환경변수 '{var_name}'이(가) 설정되지 않았고 기본값도 없습니다"
                )
            return cls._auto_cast(value)

        def replacer(m: re.Match) -> str:
            var_name, default = m.group(1), m.group(2)
            value = os.environ.get(var_name)
            if value is None:
                if default is not None:
                    return default
                raise ValueError(
                    f"환경변수 '{var_name}'이(가) 설정되지 않았고 기본값도 없습니다"
                )
            return value

        return ENV_PATTERN.sub(replacer, s)

    @staticmethod
    def _auto_cast(value: str) -> int | float | bool | str:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    def _validate_for_scope(
        self,
        settings: Settings,
        validation_scope: ValidationScope,
    ) -> None:
        if validation_scope == "training":
            self._validate(settings)
        else:
            self._validate_recipe(settings)

    def _validate_recipe(self, settings: Settings) -> None:
        """Run model/adapter validation without training-only checks."""
        from mdp.settings.validation import ValidationResult
        from mdp.settings.validation.business_validator import BusinessValidator
        from mdp.settings.validation.catalog_validator import CatalogValidator

        result = ValidationResult()

        cat_result = CatalogValidator().validate(settings)
        for warning in cat_result.warnings:
            logger.warning(warning)
        result.errors.extend(cat_result.errors)

        biz_result = BusinessValidator.validate_partial(
            settings,
            checks=["head_task", "adapter", "rl_models", "component_imports"],
        )
        result.errors.extend(biz_result.errors)
        for warning in biz_result.warnings:
            logger.warning(warning)

        if result.errors:
            raise ValueError(
                "Settings 검증 실패:\n" + "\n".join(f"  - {e}" for e in result.errors)
            )

    def _validate(self, settings: Settings) -> None:
        """Run all settings validators."""
        from mdp.settings.validation.business_validator import BusinessValidator
        from mdp.settings.validation.catalog_validator import CatalogValidator
        from mdp.settings.validation.compat_validator import CompatValidator

        all_errors: list[str] = []
        for validator_cls in (CatalogValidator, BusinessValidator, CompatValidator):
            result = validator_cls().validate(settings)
            for warning in result.warnings:
                logger.warning(warning)
            all_errors.extend(result.errors)

        if all_errors:
            raise ValueError(
                "Settings 검증 실패:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )
