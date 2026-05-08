"""ComponentResolver — component specs를 Python 객체로 해석한다."""

from __future__ import annotations

import importlib
from pathlib import Path
from collections.abc import Mapping
from typing import Any
import warnings

import yaml


class ComponentResolver:
    """component spec을 Python 객체로 해석한다.

    사용법::

        resolver = ComponentResolver()
        adam = resolver.resolve({
            "_component_": "AdamW",       # alias -> torch.optim.AdamW
            "lr": 3e-4,
            "weight_decay": 0.01,
        }, params=[model.parameters()])
    """

    COMPONENT_KEY = "_component_"

    def __init__(self) -> None:
        self._aliases = self._load_aliases()

    def _load_aliases(self) -> dict[str, str]:
        """aliases.yaml을 flat dict로 로드한다."""
        path = Path(__file__).parent.parent / "aliases.yaml"
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text())
        if not raw:
            return {}
        flat: dict[str, str] = {}
        for category in raw.values():
            if isinstance(category, dict):
                flat.update(category)
        return flat

    def resolve(self, config: Any, *args: Any, **extra_kwargs: Any) -> Any:
        """component spec을 Python 객체로 인스턴스화한다.

        Args:
            config: ``component``/``kwargs`` 필드를 가진 typed spec.
            *args: 클래스 생성자에 전달할 위치 인자.
            **extra_kwargs: config에 없지만 생성자에 필요한 추가 키워드 인자.

        Returns:
            인스턴스화된 Python 객체.
        """
        component, kwargs = self._component_and_kwargs(config)
        class_path = self._resolve_alias(component)

        # 중첩된 component 재귀 해석
        for key, value in kwargs.items():
            if self._is_resolvable(value):
                kwargs[key] = self.resolve(value)

        kwargs.update(extra_kwargs)

        klass = self.import_class(class_path)
        return klass(*args, **kwargs)

    def resolve_partial(self, config: Any) -> tuple[type, dict[str, Any]]:
        """인스턴스화하지 않고, (클래스, kwargs)를 반환한다."""
        component, kwargs = self._component_and_kwargs(config)
        class_path = self._resolve_alias(component)
        klass = self.import_class(class_path)
        return klass, kwargs

    def _component_and_kwargs(self, config: Any) -> tuple[str, dict[str, Any]]:
        """Return ``(component, kwargs)`` from typed specs or legacy mappings."""
        component = getattr(config, "resolved_component", None) or getattr(
            config, "component", None
        )
        if component is not None:
            if not isinstance(component, str):
                raise ValueError(f"component must be a string: {config!r}")
            return component, dict(getattr(config, "kwargs", {}))

        if isinstance(config, Mapping):
            warnings.warn(
                "ComponentResolver raw dict support is deprecated; pass "
                "ComponentSpec instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if self.COMPONENT_KEY not in config:
                raise ValueError(
                    f"딕셔너리에 '{self.COMPONENT_KEY}' 키가 없습니다: {config}"
                )
            raw_component = config[self.COMPONENT_KEY]
            if not isinstance(raw_component, str):
                raise ValueError(f"'{self.COMPONENT_KEY}' must be a string: {config}")
            kwargs = {
                k: v for k, v in dict(config).items() if k != self.COMPONENT_KEY
            }
            return raw_component, kwargs

        raise TypeError(f"component spec 또는 mapping이 필요합니다: {config!r}")

    def _is_resolvable(self, value: Any) -> bool:
        if getattr(value, "component", None) is not None:
            return True
        return isinstance(value, Mapping) and self.COMPONENT_KEY in value

    def _resolve_alias(self, name: str) -> str:
        """점(.)이 없으면 alias 조회, 있으면 그대로 반환."""
        if "." not in name:
            resolved = self._aliases.get(name)
            if resolved is None:
                raise ValueError(
                    f"'{name}'은(는) 등록된 alias가 아닙니다. "
                    f"풀 경로를 사용하거나 aliases.yaml에 추가하세요."
                )
            return resolved
        return name

    @staticmethod
    def import_class(class_path: str) -> type:
        """점(.)으로 구분된 경로에서 클래스를 임포트한다."""
        module_path, _, class_name = class_path.rpartition(".")
        if not module_path:
            raise ImportError(f"클래스 경로에 모듈이 없습니다: '{class_path}'")
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name, None)
        if klass is None:
            raise AttributeError(
                f"'{module_path}' 모듈에 '{class_name}'이(가) 없습니다"
            )
        return klass
