"""ComponentResolver — _component_ 딕셔너리를 Python 객체로 해석한다."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


class ComponentResolver:
    """_component_ 딕셔너리를 Python 객체로 해석한다.

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

    def resolve(
        self,
        config: dict[str, Any],
        *args: Any,
        **extra_kwargs: Any,
    ) -> Any:
        """_component_ 딕셔너리를 Python 객체로 인스턴스화한다.

        Args:
            config: _component_ 키를 포함하는 딕셔너리.
            *args: 클래스 생성자에 전달할 위치 인자.
            **extra_kwargs: config에 없지만 생성자에 필요한 추가 키워드 인자.

        Returns:
            인스턴스화된 Python 객체.
        """
        if self.COMPONENT_KEY not in config:
            raise ValueError(
                f"딕셔너리에 '{self.COMPONENT_KEY}' 키가 없습니다: {config}"
            )

        class_path = self._resolve_alias(config[self.COMPONENT_KEY])

        kwargs = {k: v for k, v in config.items() if k != self.COMPONENT_KEY}

        # 중첩된 _component_ 재귀 해석
        for key, value in kwargs.items():
            if isinstance(value, dict) and self.COMPONENT_KEY in value:
                kwargs[key] = self.resolve(value)

        kwargs.update(extra_kwargs)

        klass = self.import_class(class_path)
        return klass(*args, **kwargs)

    def resolve_partial(
        self, config: dict[str, Any]
    ) -> tuple[type, dict[str, Any]]:
        """인스턴스화하지 않고, (클래스, kwargs)를 반환한다."""
        if self.COMPONENT_KEY not in config:
            raise ValueError(
                f"딕셔너리에 '{self.COMPONENT_KEY}' 키가 없습니다: {config}"
            )

        class_path = self._resolve_alias(config[self.COMPONENT_KEY])
        kwargs = {k: v for k, v in config.items() if k != self.COMPONENT_KEY}
        klass = self.import_class(class_path)
        return klass, kwargs

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
