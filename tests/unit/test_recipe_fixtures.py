"""Recipe fixture 전체 conformance 테스트.

S7 리팩토링 범위: tests/fixtures/recipes/ 아래 모든 Recipe YAML이
component-unified schema(DataSpec dict wrappers, _component_ dict 패턴)에
부합하고, 선언된 모든 _component_ 식별자가 ComponentResolver로 해석
가능한지를 파일 단위로 검증한다.

파일 단위 검증은 기존 test_settings.test_yaml_parsing의 parametrize
리스트가 빠뜨린 fixture를 놓치지 않도록 하기 위한 것이다. 새 fixture가
추가되면 이 테스트는 자동으로 그것을 감시 범위에 포함한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
import yaml

from mdp.settings.factory import SettingsFactory
from mdp.settings.resolver import ComponentResolver
from mdp.settings.schema import Recipe

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "recipes"


def _all_recipe_paths() -> list[Path]:
    """fixture 디렉토리의 모든 recipe YAML 경로를 반환."""
    paths = sorted(FIXTURES.glob("*.yaml"))
    assert paths, f"No recipe fixtures found at {FIXTURES}"
    return paths


# ---------------------------------------------------------------------------
# (1) 전체 fixture가 for_estimation 경로로 파싱 + 부분검증을 통과해야 한다.
#     (config 없이도 Recipe 단독으로 유효해야 한다)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_parses_via_factory(recipe_path: Path) -> None:
    """모든 recipe fixture가 SettingsFactory.for_estimation으로 파싱된다."""
    factory = SettingsFactory()
    settings = factory.for_estimation(str(recipe_path))
    assert settings.recipe.name
    assert settings.recipe.task


# ---------------------------------------------------------------------------
# (2) DataSpec이 component-unified schema를 따른다 — 구 source:/label_strategy:
#     필드가 남아 있으면 회귀로 간주한다.
# ---------------------------------------------------------------------------

_LEGACY_DATA_KEYS = {
    "source",
    "label_strategy",
    "fields",
    "format",
    "split",
    "subset",
    "streaming",
    "data_files",
    "tokenizer",
    "augmentation",
    "val_split",
}


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_data_spec_uses_new_schema(recipe_path: Path) -> None:
    """data 블록이 dataset/collator 래퍼를 쓰고, 구 고정 키가 top-level에 없다."""
    raw = yaml.safe_load(recipe_path.read_text())
    data = raw.get("data", {})

    # top-level 구 키가 남아 있으면 안 된다 — 전부 dataset/collator 내부로 이동했어야 한다.
    leaked = set(data.keys()) & _LEGACY_DATA_KEYS
    assert not leaked, (
        f"{recipe_path.name}: data 블록에 구 스키마 키가 남아 있다: {sorted(leaked)}"
    )

    # dataset/collator는 _component_ dict이어야 한다.
    assert "dataset" in data, f"{recipe_path.name}: data.dataset 누락"
    assert "collator" in data, f"{recipe_path.name}: data.collator 누락"
    assert isinstance(data["dataset"], dict) and "_component_" in data["dataset"], (
        f"{recipe_path.name}: data.dataset이 _component_ dict이 아니다"
    )
    assert isinstance(data["collator"], dict) and "_component_" in data["collator"], (
        f"{recipe_path.name}: data.collator가 _component_ dict이 아니다"
    )


# ---------------------------------------------------------------------------
# (3) model/adapter/optimizer/... 같은 component dict가 모두 _component_ 키를
#     가지고 있고, 그 값이 ComponentResolver로 해석 가능해야 한다.
# ---------------------------------------------------------------------------


def _iter_component_dicts(node: Any) -> Iterator[dict[str, Any]]:
    """중첩 구조를 훑어 _component_ 키를 가진 모든 dict를 yield한다."""
    if isinstance(node, dict):
        if "_component_" in node:
            yield node
        for v in node.values():
            yield from _iter_component_dicts(v)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_component_dicts(item)


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_components_resolvable(recipe_path: Path) -> None:
    """fixture 내 모든 _component_ 값이 class path로 해석 가능하다.

    실제 클래스 임포트는 하지 않고 (optional deps/네트워크를 피하려고),
    alias → full path 해석까지만 검증한다. 잘못된 alias 이름(오타, 제거된
    alias 참조)은 여기서 잡힌다.
    """
    raw = yaml.safe_load(recipe_path.read_text())
    resolver = ComponentResolver()

    errors: list[str] = []
    for cfg in _iter_component_dicts(raw):
        name = cfg["_component_"]
        if not isinstance(name, str):
            errors.append(f"_component_ 값이 문자열이 아님: {name!r}")
            continue
        try:
            resolved = resolver._resolve_alias(name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        # full path는 점 구분자를 반드시 포함한다.
        assert "." in resolved, (
            f"{recipe_path.name}: '{name}'이 점 없는 경로로 해석되었다: {resolved}"
        )

    assert not errors, f"{recipe_path.name}: {errors}"


# ---------------------------------------------------------------------------
# (4) RL 모델 dict 형태: rl.models는 dict-of-dict이고, 각 역할은 _component_
#     키를 갖는다. 새 RL fixture가 RLModelSpec 제거(§3.4)의 타깃 형태를
#     충실히 따르는지 보장한다.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_rl_fixture_models_shape(recipe_path: Path) -> None:
    """rl 블록이 있으면 models가 {role: {_component_, ...}} 형태여야 한다."""
    raw = yaml.safe_load(recipe_path.read_text())
    rl = raw.get("rl")
    if rl is None:
        pytest.skip("RL 미지정")
    models = rl.get("models")
    assert isinstance(models, dict) and models, (
        f"{recipe_path.name}: rl.models는 비어있지 않은 dict이어야 한다"
    )
    for role, spec in models.items():
        assert isinstance(spec, dict), (
            f"{recipe_path.name}: rl.models.{role}이 dict이 아니다"
        )
        assert "_component_" in spec, (
            f"{recipe_path.name}: rl.models.{role}에 _component_ 키가 없다"
        )


# ---------------------------------------------------------------------------
# (6) Pydantic Recipe 모델로 직접 로딩해도 필드 구조가 일치한다. 기존
#     test_yaml_parsing이 config 페어링까지 요구하는 것과 달리, 이 테스트는
#     Recipe 스키마만 단독으로 검증한다.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_recipe_pydantic_load(recipe_path: Path) -> None:
    """Recipe(**yaml)이 크래시 없이 로드된다."""
    raw = yaml.safe_load(recipe_path.read_text())
    recipe = Recipe(**raw)
    # DataSpec이 dict 필드를 실제로 노출하는지 확인 (S1 변경의 회귀 방지)
    assert isinstance(recipe.data.dataset, dict)
    assert isinstance(recipe.data.collator, dict)
    # ModelSpec 제거 확인 — model이 dict로 노출되어야 한다 (S2 변경의 회귀 방지)
    assert isinstance(recipe.model, dict)
