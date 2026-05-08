"""SettingsFactory compatibility facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mdp.settings.planner import SettingsPlanner
from mdp.settings.schema import Settings


class SettingsFactory:
    """YAML 파일 경로를 받아 Settings 객체를 조립한다."""

    def __init__(self, planner: SettingsPlanner | None = None) -> None:
        self._planner = planner or SettingsPlanner()

    def for_training(
        self,
        recipe_path: str,
        config_path: str,
        overrides: list[str] | None = None,
    ) -> Settings:
        """학습용 Settings를 조립한다."""
        return self._planner.load_training(
            recipe_path,
            config_path,
            overrides=overrides,
        ).settings

    def for_estimation(self, recipe_path: str) -> Settings:
        """추정용 Settings를 조립한다. Recipe만 로딩하고 기본 Config를 사용한다."""
        return self._planner.load_estimation(recipe_path).settings

    def for_inference(self, recipe_path: str, config_path: str) -> Settings:
        """추론용 Settings를 조립한다. 분산 학습 전용 검증은 수행하지 않는다."""
        return self._planner.load_inference(recipe_path, config_path).settings

    def from_artifact(
        self,
        artifact_dir: str,
        overrides: list[str] | None = None,
        *,
        use_config_snapshot: bool = True,
    ) -> Settings:
        """artifact 디렉토리의 recipe/config snapshot에서 Settings를 조립한다."""
        return self._planner.load_artifact(
            artifact_dir,
            overrides=overrides,
            use_config_snapshot=use_config_snapshot,
        ).settings

    @staticmethod
    def _find_artifact_config_snapshot(artifact_dir: Path) -> Path | None:
        return SettingsPlanner.find_artifact_config_snapshot(artifact_dir)

    @staticmethod
    def _split_and_apply_overrides(
        recipe_dict: dict[str, Any],
        config_dict: dict[str, Any],
        overrides: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return SettingsPlanner._split_and_apply_overrides(
            recipe_dict,
            config_dict,
            overrides,
        )

    @staticmethod
    def _load_yaml(path: str) -> Any:
        return SettingsPlanner._load_yaml(path)

    @classmethod
    def _substitute_env_vars(cls, obj: Any) -> Any:
        return SettingsPlanner._substitute_env_vars(obj)

    @staticmethod
    def _auto_cast(value: str) -> int | float | bool | str:
        return SettingsPlanner._auto_cast(value)

    def _validate_recipe(self, settings: Settings) -> None:
        self._planner._validate_recipe(settings)

    def _validate(self, settings: Settings) -> None:
        self._planner._validate(settings)
