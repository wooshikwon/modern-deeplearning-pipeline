"""기존 5+ recipe fixture 회귀 테스트 — sampler 미지정 100% 동작 보존.

spec-length-bucketed-sampler.md U6의 핵심 회귀 검증:

`data.sampler` 필드를 추가했어도 기존 fixture들은 sampler를 선언하지 않으므로
``create_dataloaders()``의 분기는 정확히 **이전 동작**으로 떨어져야 한다:

- ``distributed=False`` + ``sampler_config=None`` → ``RandomSampler`` 자동 부착
  (PyTorch ``DataLoader(shuffle=True)``의 표준 동작)
- ``distributed=True`` + ``sampler_config=None`` → ``DistributedSampler`` 자동 부착

본 테스트는 모든 fixture를 parametrize 대상으로 잡고 각 fixture에 대해 위 두
분기가 모두 정확히 작동함을 격리 검증한다. 실제 HF datasets 다운로드는 회피하기
위해 dataset/collator는 fake 클래스로 대체하되, dataloader_config(``batch_size``,
``num_workers``, ``drop_last`` 등)는 fixture 값을 그대로 사용한다 — 즉 실 fixture가
선언한 dl_kwargs가 sampler 미지정 분기와 호환됨을 함께 보장한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch.utils.data as tdata
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mdp.data.dataloader import create_dataloaders
from mdp.settings.schema import Recipe

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "recipes"


def _all_recipe_paths() -> list[Path]:
    paths = sorted(FIXTURES.glob("*.yaml"))
    assert paths, f"No recipe fixtures found at {FIXTURES}"
    return paths


# ──────────────────────────────────────────────────────────────────────
# Fake dataset/collator (모듈 레벨 — _component_ import path로 참조)
# ──────────────────────────────────────────────────────────────────────


class _FakeListDataset:
    """단순 list-of-dict dataset.

    각 sample은 ``{"x": idx}``를 반환한다. ``__len__``과 ``__getitem__``만 제공하면
    DataLoader / DistributedSampler / RandomSampler 모두 정상 동작한다.
    """

    def __init__(self, n: int = 32) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"x": idx}


class _FakeListDatasetSpec:
    """``_component_``로 인스턴스화될 wrapper. n을 그대로 위임한다."""

    def __new__(cls, n: int = 32) -> _FakeListDataset:  # type: ignore[misc]
        return _FakeListDataset(n)


class _IdentityCollator:
    """features를 그대로 반환하는 collator — 조립 경로만 검증한다."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return features


def _component_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _fake_dataset_config(n: int = 32) -> dict[str, Any]:
    return {"_component_": _component_path(_FakeListDatasetSpec), "n": n}


def _fake_collator_config() -> dict[str, Any]:
    return {"_component_": _component_path(_IdentityCollator)}


# ──────────────────────────────────────────────────────────────────────
# Fixture 로딩 헬퍼
# ──────────────────────────────────────────────────────────────────────


def _load_recipe_raw(recipe_path: Path) -> dict[str, Any]:
    return yaml.safe_load(recipe_path.read_text())


def _extract_dataloader_config(raw: dict[str, Any]) -> dict[str, Any]:
    """fixture의 ``data.dataloader`` block을 그대로 반환 (없으면 빈 dict).

    DataLoaderSpec의 default value들은 schema 단계에서 채워지지만, 회귀 테스트
    에서는 fixture가 명시한 키만 사용해도 충분하다. ``num_workers=0``로 강제하여
    persistent_workers / prefetch_factor의 부적절성을 회피한다.
    """
    dl = dict(raw.get("data", {}).get("dataloader", {}))
    # 테스트 격리: subprocess worker는 사용하지 않는다 (CI/로컬 일관성)
    dl["num_workers"] = 0
    dl.pop("persistent_workers", None)
    dl.pop("prefetch_factor", None)
    dl.pop("pin_memory", None)
    return dl


# ──────────────────────────────────────────────────────────────────────
# (1) 모든 fixture가 ``data.sampler`` 필드를 선언하지 않는다
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_has_no_sampler_field(recipe_path: Path) -> None:
    """기존 fixture YAML에 ``data.sampler`` 키가 등장하지 않는다.

    spec-length-bucketed-sampler.md 원칙 4 (기존 동작 100% 보존)의 정적 보증.
    sampler 필드 추가는 본 spec이 도입하는 신규 기능이므로 기존 fixture는 미선언
    상태여야 한다.
    """
    raw = _load_recipe_raw(recipe_path)
    data = raw.get("data", {})
    assert "sampler" not in data, (
        f"{recipe_path.name}: 기존 fixture는 data.sampler 필드를 선언하면 안 된다 "
        f"(본 spec이 신규 도입). 현재값: {data.get('sampler')!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# (2) Recipe schema가 sampler=None을 허용 + 기본값이 None이다
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_recipe_loads_with_sampler_none(recipe_path: Path) -> None:
    """fixture를 Recipe Pydantic으로 로드해도 ``data.sampler``가 None이다.

    DataSpec.sampler가 optional 필드로 추가되었어도 미선언 fixture에서 None default가
    유지되는지 확인. 만약 mandatory 필드로 잘못 추가되었거나 default가 dict/empty로
    바뀌었으면 본 테스트가 즉시 잡는다.
    """
    raw = _load_recipe_raw(recipe_path)
    recipe = Recipe(**raw)
    assert recipe.data.sampler is None, (
        f"{recipe_path.name}: data.sampler default가 None이 아니다: "
        f"{recipe.data.sampler!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# (3) sampler 미지정 + distributed=False → RandomSampler 자동 부착
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_no_sampler_non_distributed_uses_random_sampler(
    recipe_path: Path,
) -> None:
    """fixture의 dl_kwargs로 ``create_dataloaders``를 호출하면, distributed=False
    환경에서 ``RandomSampler``(``shuffle=True`` 분기)가 자동 부착된다.

    이는 본 spec 이전 동작 — ``sampler_config=None`` + ``distributed=False`` →
    ``DataLoader(shuffle=True)`` 경로가 그대로 보존됨을 검증한다. PyTorch가
    ``shuffle=True``일 때 ``RandomSampler``를 생성하므로 sampler attribute로
    분기 결정을 확인할 수 있다.
    """
    raw = _load_recipe_raw(recipe_path)
    dl_config = _extract_dataloader_config(raw)

    loaders = create_dataloaders(
        dataset_config=_fake_dataset_config(n=32),
        collator_config=_fake_collator_config(),
        dataloader_config=dl_config,
        sampler_config=None,
        distributed=False,
    )

    train_loader = loaders["train"]
    assert isinstance(train_loader, DataLoader)
    # shuffle=True 분기는 RandomSampler를 자동 부착한다 (PyTorch 표준)
    assert isinstance(train_loader.sampler, tdata.RandomSampler), (
        f"{recipe_path.name}: distributed=False + sampler 미지정 → "
        f"RandomSampler 기대, 실제: {type(train_loader.sampler).__name__}"
    )
    # batch_sampler 경로가 아니어야 한다 — DataLoader.batch_size가 살아있어야 함
    assert train_loader.batch_size is not None, (
        f"{recipe_path.name}: batch_sampler 경로로 잘못 진입했다"
    )


# ──────────────────────────────────────────────────────────────────────
# (4) sampler 미지정 + distributed=True → DistributedSampler 자동 부착
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_no_sampler_distributed_uses_distributed_sampler(
    recipe_path: Path,
) -> None:
    """fixture의 dl_kwargs로 ``create_dataloaders``를 호출하면, distributed=True
    환경에서 ``DistributedSampler``가 자동 부착된다.

    ``DistributedSampler.__init__``은 ``torch.distributed`` 초기화를 요구하므로
    단위 테스트에서는 ``num_replicas``/``rank``를 mock 함수로 제공한다 —
    PyTorch DistributedSampler는 두 인자가 None일 때만 ``get_world_size()``/
    ``get_rank()``를 호출하므로, 미초기화 환경에서도 explicit 인자로 인스턴스화
    가능하다. 본 테스트는 ``DistributedSampler`` 생성자를 monkey-patch하여
    explicit 인자를 자동 주입한다.
    """
    raw = _load_recipe_raw(recipe_path)
    dl_config = _extract_dataloader_config(raw)

    real_init = DistributedSampler.__init__

    def _patched_init(
        self: DistributedSampler,
        dataset: Any,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        # distributed가 미초기화된 단위 테스트 환경에서도 결정적 인자 주입
        return real_init(
            self,
            dataset=dataset,
            num_replicas=num_replicas if num_replicas is not None else 1,
            rank=rank if rank is not None else 0,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    with mock.patch.object(DistributedSampler, "__init__", _patched_init):
        loaders = create_dataloaders(
            dataset_config=_fake_dataset_config(n=32),
            collator_config=_fake_collator_config(),
            dataloader_config=dl_config,
            sampler_config=None,
            distributed=True,
        )

    train_loader = loaders["train"]
    assert isinstance(train_loader, DataLoader)
    assert isinstance(train_loader.sampler, DistributedSampler), (
        f"{recipe_path.name}: distributed=True + sampler 미지정 → "
        f"DistributedSampler 기대, 실제: {type(train_loader.sampler).__name__}"
    )
    # batch_sampler 경로가 아니어야 한다
    assert train_loader.batch_size is not None, (
        f"{recipe_path.name}: batch_sampler 경로로 잘못 진입했다"
    )


# ──────────────────────────────────────────────────────────────────────
# (5) DataLoader가 1개 이상의 batch를 yield 한다 (smoke 검증)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: p.name)
def test_fixture_no_sampler_yields_batches(recipe_path: Path) -> None:
    """fixture로 만든 DataLoader가 적어도 1개의 batch를 정상 yield한다.

    드물지만 dl_kwargs 호환성 문제(예: drop_last=True + 너무 작은 dataset)로
    iteration이 0건이 되는 회귀를 잡는다.
    """
    raw = _load_recipe_raw(recipe_path)
    dl_config = _extract_dataloader_config(raw)
    # batch_size가 dataset(n=32)보다 큰 fixture(예: vit-lora-cifar10: 64)를 위해
    # n을 충분히 크게 잡고, drop_last=True여도 최소 1 batch가 되도록 보장한다.
    batch_size = int(dl_config.get("batch_size", 1))
    n = max(64, batch_size * 2)

    loaders = create_dataloaders(
        dataset_config=_fake_dataset_config(n=n),
        collator_config=_fake_collator_config(),
        dataloader_config=dl_config,
        sampler_config=None,
        distributed=False,
    )

    train_loader = loaders["train"]
    batches = list(train_loader)
    assert len(batches) >= 1, (
        f"{recipe_path.name}: dl_kwargs={dl_config}로 0개 batch가 yield됨"
    )
    # 첫 batch가 batch_size 또는 잔여(=n%batch_size if drop_last=False)여야 한다
    first = batches[0]
    assert isinstance(first, list) and len(first) > 0
