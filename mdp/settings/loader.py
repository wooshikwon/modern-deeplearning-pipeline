"""Settings source loader.

This module owns raw source handling only: YAML, env substitution, overrides,
artifact config snapshots, Pydantic construction, and scope validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from mdp.settings.run_plan import ValidationScope
from mdp.settings.schema import Config, Recipe, Settings

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


class _PathAwareSafeLoader(yaml.SafeLoader):
    """SafeLoader that reports duplicate mapping keys with YAML path context."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._yaml_path_stack: list[str] = ["$"]

    @property
    def _yaml_path(self) -> str:
        return self._yaml_path_stack[-1]

    @staticmethod
    def _child_path(parent: str, key: Any) -> str:
        if isinstance(key, int):
            return f"{parent}[{key}]"
        key_text = str(key)
        if key_text.isidentifier():
            return f"{parent}.{key_text}" if parent != "$" else f"$.{key_text}"
        return f"{parent}[{key_text!r}]"

    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"expected a mapping node, but found {node.id}",
                node.start_mark,
            )

        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            key_path = self._child_path(self._yaml_path, key)
            if key in mapping:
                raise ValueError(f"duplicate key {key!r} at YAML path {key_path}")
            self._yaml_path_stack.append(key_path)
            try:
                mapping[key] = self.construct_object(value_node, deep=deep)
            finally:
                self._yaml_path_stack.pop()
        return mapping

    def construct_sequence(
        self,
        node: yaml.SequenceNode,
        deep: bool = False,
    ) -> list[Any]:
        if not isinstance(node, yaml.SequenceNode):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"expected a sequence node, but found {node.id}",
                node.start_mark,
            )

        sequence: list[Any] = []
        for index, child in enumerate(node.value):
            self._yaml_path_stack.append(f"{self._yaml_path}[{index}]")
            try:
                sequence.append(self.construct_object(child, deep=deep))
            finally:
                self._yaml_path_stack.pop()
        return sequence


_PathAwareSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _PathAwareSafeLoader.construct_mapping,
)
_PathAwareSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
    _PathAwareSafeLoader.construct_sequence,
)


class SettingsLoader:
    """Load validated Settings from recipes, configs, artifacts, and overrides."""

    def load_training_settings(
        self,
        recipe_path: str | Path,
        config_path: str | Path,
        overrides: list[str] | None = None,
    ) -> Settings:
        return self._build_settings(
            Path(recipe_path),
            Path(config_path),
            overrides=overrides,
            validation_scope="training",
        )

    def load_estimation_settings(
        self,
        recipe_path: str | Path,
        overrides: list[str] | None = None,
    ) -> Settings:
        return self._build_settings(
            Path(recipe_path),
            None,
            overrides=overrides,
            validation_scope="estimation",
        )

    def load_inference_settings(
        self,
        recipe_path: str | Path,
        config_path: str | Path,
        overrides: list[str] | None = None,
    ) -> Settings:
        return self._build_settings(
            Path(recipe_path),
            Path(config_path),
            overrides=overrides,
            validation_scope="recipe",
        )

    def load_artifact_settings(
        self,
        artifact_dir: str | Path,
        overrides: list[str] | None = None,
        *,
        use_config_snapshot: bool = True,
    ) -> Settings:
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

        return self._build_settings(
            recipe_path,
            config_path,
            overrides=overrides,
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
        recipe_dict = self.substitute_env_vars(self.load_yaml(recipe_path))
        config_dict = (
            self.substitute_env_vars(self.load_yaml(config_path))
            if config_path is not None
            else {}
        )

        if overrides:
            recipe_dict, config_dict = self.split_and_apply_overrides(
                recipe_dict,
                config_dict,
                overrides,
            )

        try:
            recipe = Recipe(**recipe_dict)
        except ValidationError as exc:
            raise ValueError(
                self.format_schema_error(recipe_path, "recipe", exc)
            ) from exc
        try:
            config = Config(**config_dict) if config_dict else Config()
        except ValidationError as exc:
            config_source = config_path if config_path is not None else Path("<defaults>")
            raise ValueError(
                self.format_schema_error(config_source, "config", exc)
            ) from exc

        settings = Settings(recipe=recipe, config=config)
        self.validate_for_scope(settings, validation_scope)
        return settings

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
    def split_and_apply_overrides(
        recipe_dict: dict[str, Any],
        config_dict: dict[str, Any],
        overrides: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split overrides into recipe/config namespaces and apply them."""
        from mdp.cli._override import apply_overrides

        config_top_keys = {
            "environment",
            "compute",
            "mlflow",
            "storage",
            "serving",
            "job",
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
                    override,
                    top_key,
                    override,
                )

        if recipe_ovr:
            apply_overrides(recipe_dict, recipe_ovr)
        if config_ovr:
            apply_overrides(config_dict, config_ovr)
        return recipe_dict, config_dict

    @staticmethod
    def load_yaml(
        path: str | Path,
        *,
        root_type: str = "mapping",
        allow_empty: bool = False,
    ) -> Any:
        source = Path(path)
        text = source.read_text()
        if not text.strip():
            if allow_empty:
                return None
            raise ValueError(f"{source}: YAML path $: YAML file is empty")

        try:
            loaded = yaml.load(text, Loader=_PathAwareSafeLoader)
        except ValueError as exc:
            raise ValueError(f"{source}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"{source}: YAML path $: invalid YAML: {exc}") from exc

        if loaded is None:
            if allow_empty:
                return None
            raise ValueError(f"{source}: YAML path $: YAML file is empty")
        if root_type == "mapping" and not isinstance(loaded, dict):
            raise ValueError(
                f"{source}: YAML path $: YAML root must be a mapping "
                f"(actual: {type(loaded).__name__})"
            )
        if root_type == "list" and not isinstance(loaded, list):
            raise ValueError(
                f"{source}: YAML path $: YAML root must be a list "
                f"(actual: {type(loaded).__name__})"
            )
        return loaded

    @staticmethod
    def format_schema_error(path: Path, root: str, exc: ValidationError) -> str:
        lines = [f"{path}: schema validation failed"]
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            yaml_path = f"$.{loc}" if loc else "$"
            lines.append(f"  - YAML path {yaml_path}: {error['msg']}")
        return "\n".join(lines)

    @classmethod
    def substitute_env_vars(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls.substitute_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls.substitute_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return cls.substitute_string(obj)
        return obj

    @classmethod
    def substitute_string(cls, s: str) -> Any:
        match = ENV_PATTERN.fullmatch(s)
        if match:
            var_name, default = match.group(1), match.group(2)
            value = os.environ.get(var_name)
            if value is None:
                if default is not None:
                    return cls.auto_cast(default)
                raise ValueError(
                    f"환경변수 '{var_name}'이(가) 설정되지 않았고 기본값도 없습니다"
                )
            return cls.auto_cast(value)

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
    def auto_cast(value: str) -> int | float | bool | str:
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

    def validate_for_scope(
        self,
        settings: Settings,
        validation_scope: ValidationScope,
    ) -> None:
        if validation_scope == "training":
            self.validate(settings)
        else:
            self.validate_recipe(settings)

    def validate_recipe(self, settings: Settings) -> None:
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
            validation_scope="recipe",
        )
        result.errors.extend(biz_result.errors)
        for warning in biz_result.warnings:
            logger.warning(warning)

        if result.errors:
            raise ValueError(
                "Settings 검증 실패:\n" + "\n".join(f"  - {e}" for e in result.errors)
            )

    def validate(self, settings: Settings) -> None:
        """Run all settings validators."""
        from mdp.settings.validation.business_validator import BusinessValidator
        from mdp.settings.validation.catalog_validator import CatalogValidator
        from mdp.settings.validation.compat_validator import CompatValidator

        all_errors: list[str] = []
        for validator_cls in (CatalogValidator, BusinessValidator, CompatValidator):
            if validator_cls is BusinessValidator:
                result = validator_cls().validate(settings, validation_scope="training")
            else:
                result = validator_cls().validate(settings)
            for warning in result.warnings:
                logger.warning(warning)
            all_errors.extend(result.errors)

        if all_errors:
            raise ValueError(
                "Settings 검증 실패:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )
