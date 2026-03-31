"""CatalogValidator — Model Catalog 기반 검증.

catalog YAML에서 모델 메타데이터를 로딩하고,
recipe의 pretrained URI와 task가 catalog과 호환되는지 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mdp.settings.schema import Settings

_DEFAULT_CATALOG_DIR = Path(__file__).resolve().parents[2] / "models" / "catalog"


class CatalogValidator:
    """모델이 catalog에 존재하는지, 호환 태스크인지 검증한다."""

    def __init__(self, catalog_dir: Path | None = None) -> None:
        self._catalog_dir = catalog_dir or _DEFAULT_CATALOG_DIR
        self._catalog: dict[str, dict[str, Any]] = {}
        self._load_catalog()

    # ── 내부 메서드 ──

    def _load_catalog(self) -> None:
        """catalog 디렉토리의 모든 YAML을 로딩하여 내부 dict에 캐시한다."""
        if not self._catalog_dir.is_dir():
            return
        for yaml_path in self._catalog_dir.rglob("*.yaml"):
            with open(yaml_path) as f:
                entry = yaml.safe_load(f)
            if entry and "name" in entry:
                self._catalog[entry["name"]] = entry

    def _find_by_pretrained(self, pretrained: str) -> dict[str, Any] | None:
        """모든 catalog 항목의 pretrained_sources를 순회하여 매칭한다."""
        for entry in self._catalog.values():
            sources = entry.get("pretrained_sources", [])
            if isinstance(sources, list):
                if pretrained in sources:
                    return entry
            elif isinstance(sources, dict):
                if pretrained in sources.values():
                    return entry
        return None

    # ── 공개 API ──

    def validate(self, settings: Settings) -> list[str]:
        """경고 메시지 목록을 반환한다. 빈 리스트면 통과."""
        warnings: list[str] = []

        pretrained = settings.recipe.model.pretrained
        if pretrained is None:
            return warnings

        entry = self._find_by_pretrained(pretrained)
        if entry is None:
            warnings.append(
                f"모델 '{pretrained}'이(가) catalog에 없습니다. "
                f"catalog 검증을 건너뜁니다."
            )
            return warnings

        task = settings.recipe.task
        supported = entry.get("supported_tasks", [])
        if task not in supported:
            warnings.append(
                f"태스크 '{task}'은(는) 모델 '{entry['name']}'의 "
                f"지원 태스크 목록에 없습니다. "
                f"지원 태스크: {supported}"
            )

        return warnings

    def get_defaults(self, pretrained: str) -> dict[str, Any] | None:
        """pretrained URI로 catalog 항목을 찾아 전체 메타데이터를 반환한다."""
        return self._find_by_pretrained(pretrained)
