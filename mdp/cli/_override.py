"""CLI override 유틸리티 — --override KEY=VALUE 파싱 및 적용."""

from __future__ import annotations

import json
from typing import Any


def apply_overrides(
    target: dict[str, Any], overrides: list[str],
) -> dict[str, Any]:
    """dotted key=value 오버라이드를 deep-merge한다.

    Examples::

        apply_overrides(recipe_dict, ["training.epochs=0.1"])
        apply_overrides(config_dict, ["compute.gpus=4"])
    """
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"올바른 형식: KEY=VALUE. 입력: '{item}'")
        keys = key.split(".")
        parsed = parse_value(value)
        _deep_set(target, keys, parsed)
    return target


def parse_value(value: str) -> Any:
    """자동 타입 추론. null → bool → int → float → JSON → str."""
    if value.lower() in ("null", "none"):
        return None
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
    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _deep_set(d: dict, keys: list[str], value: Any) -> None:
    """중첩 딕셔너리에 dotted path로 값을 설정한다."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
