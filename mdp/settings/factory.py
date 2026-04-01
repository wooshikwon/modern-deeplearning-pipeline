"""SettingsFactory — YAML 파일을 Settings 객체로 조립한다."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import yaml

from mdp.settings.schema import Config, Recipe, Settings

logger = logging.getLogger(__name__)

ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


class SettingsFactory:
    """YAML 파일 경로를 받아 Settings 객체를 조립한다."""

    def for_training(self, recipe_path: str, config_path: str) -> Settings:
        """학습용 Settings를 조립한다."""
        recipe_dict = self._load_yaml(recipe_path)
        config_dict = self._load_yaml(config_path)

        recipe_dict = self._substitute_env_vars(recipe_dict)
        config_dict = self._substitute_env_vars(config_dict)

        recipe = Recipe(**recipe_dict)
        config = Config(**config_dict)
        settings = Settings(recipe=recipe, config=config)
        self._validate(settings)
        return settings

    def for_estimation(self, recipe_path: str) -> Settings:
        """추정용 Settings를 조립한다. Recipe만 로딩하고 기본 Config를 사용한다."""
        recipe_dict = self._load_yaml(recipe_path)
        recipe_dict = self._substitute_env_vars(recipe_dict)

        recipe = Recipe(**recipe_dict)
        config = Config()
        return Settings(recipe=recipe, config=config)

    def for_inference(self, recipe_path: str, config_path: str) -> Settings:
        """추론용 Settings를 조립한다. 학습 전용 검증은 수행하지 않는다."""
        recipe_dict = self._load_yaml(recipe_path)
        config_dict = self._load_yaml(config_path)

        recipe_dict = self._substitute_env_vars(recipe_dict)
        config_dict = self._substitute_env_vars(config_dict)

        recipe = Recipe(**recipe_dict)
        config = Config(**config_dict)
        return Settings(recipe=recipe, config=config)

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """YAML 파일을 딕셔너리로 로딩한다."""
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def _substitute_env_vars(cls, obj: Any) -> Any:
        """재귀적으로 ${VAR:default} 패턴을 환경변수 값으로 치환한다."""
        if isinstance(obj, dict):
            return {k: cls._substitute_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._substitute_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return cls._substitute_string(obj)
        return obj

    @classmethod
    def _substitute_string(cls, s: str) -> Any:
        """문자열 내 환경변수 패턴을 치환한다."""
        match = ENV_PATTERN.fullmatch(s)
        if match:
            # 전체 매칭: 타입 변환 적용
            var_name, default = match.group(1), match.group(2)
            value = os.environ.get(var_name)
            if value is None:
                if default is not None:
                    return cls._auto_cast(default)
                raise ValueError(
                    f"환경변수 '{var_name}'이(가) 설정되지 않았고 기본값도 없습니다"
                )
            return cls._auto_cast(value)

        # 부분 매칭: 문자열 내 치환 (타입 변환 없음)
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
        """문자열을 가능한 타입으로 변환한다."""
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

    def _validate(self, settings: Settings) -> None:
        """CatalogValidator, BusinessValidator, CompatValidator를 순서대로 실행한다.

        warnings는 로깅하고, errors가 있으면 ValueError를 발생시킨다.
        """
        from mdp.settings.validation.catalog_validator import CatalogValidator
        from mdp.settings.validation.business_validator import BusinessValidator
        from mdp.settings.validation.compat_validator import CompatValidator

        all_errors: list[str] = []
        for validator_cls in (CatalogValidator, BusinessValidator, CompatValidator):
            result = validator_cls().validate(settings)
            for w in result.warnings:
                logger.warning(w)
            all_errors.extend(result.errors)

        if all_errors:
            raise ValueError(
                "Settings 검증 실패:\n" + "\n".join(f"  - {e}" for e in all_errors)
            )
