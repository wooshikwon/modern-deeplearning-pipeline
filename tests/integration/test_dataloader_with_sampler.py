"""``create_dataloaders``의 sampler_config 통합 + AssemblyMaterializer end-to-end 통합 테스트.

spec-length-bucketed-sampler.md U5의 통합 테스트 4건:

1. ``test_dataloader_with_length_grouped_sampler``
   — sampler_config 주입 시 batch 내 길이가 정렬됨 (bucket 효과)
2. ``test_dataloader_without_sampler_uses_default_shuffle``
   — sampler_config 미지정 시 기존 동작(``shuffle=True``) 100% 보존
3. ``test_dataloader_with_sampler_overrides_dl_kwargs``
   — sampler 주입 시 ``batch_size``/``shuffle``/``drop_last``가 자동 무시되어
     DataLoader가 ValueError를 일으키지 않음
4. ``test_materializer_creates_dataloader_with_sampler_from_recipe``
   — AssemblyMaterializer.create_dataloaders() 가 ``data.sampler``를 그대로 전달하여
     end-to-end 경로가 작동

설계 결정:

- 실 dataset(``HuggingFaceDataset``)/collator(``CausalLMCollator``)는 토크나이저와
  HF datasets 다운로드를 동반하므로 단위 비용이 크다. 본 통합 테스트는 fake
  dataset/collator 클래스를 모듈 레벨에 정의하고, ``_component_`` 경로로 그대로
  resolve되도록 하여 ``create_dataloaders`` 조립 경로 자체만 격리 검증한다.
- AssemblyMaterializer 경로 테스트는 ``Settings``를 실제 Pydantic 인스턴스로 빌드한다 —
  ``DataSpec.sampler`` 필드 추가가 schema 단계에서 정상 통과함을 함께 보장.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from mdp.data.dataloader import create_dataloaders
from mdp.data.samplers import LengthGroupedBatchSampler
from mdp.assembly.materializer import AssemblyMaterializer
from mdp.assembly.planner import AssemblyPlanner
from mdp.settings.run_plan import RunPlan, RunSources
from mdp.settings.schema import (
    Config,
    DataloaderSpec,
    DataSpec,
    MetadataSpec,
    Recipe,
    Settings,
    TrainingSpec,
)


# ──────────────────────────────────────────────────────────────────────
# Fake dataset/collator (모듈 레벨 — _component_ import path로 참조)
# ──────────────────────────────────────────────────────────────────────


class _FakeLengthedDataset:
    """``Lengthed`` protocol을 만족하는 단순 dataset.

    각 sample은 ``{"x": idx, "length": L}`` 형태로 반환되며,
    ``__getlength__(idx)``가 미리 주입된 길이를 노출한다. 실제 토큰화 비용 없이
    sampler의 길이 수집·정렬 경로만 통과시킨다.
    """

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = list(lengths)

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"x": idx, "length": self._lengths[idx]}

    def __getlength__(self, idx: int) -> int:
        return self._lengths[idx]


class _FakeLengthedDatasetSpec:
    """Recipe.data.dataset에서 ``_component_``로 인스턴스화될 wrapper.

    Recipe schema가 dict-of-Any를 받으므로 lengths를 그대로 위임한다.
    """

    def __new__(cls, lengths: list[int]) -> _FakeLengthedDataset:  # type: ignore[misc]
        return _FakeLengthedDataset(lengths)


class _IdentityCollator:
    """features를 그대로 반환하는 collator. dataloader 조립 경로만 격리한다."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return features


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────


def _component_path(cls: type) -> str:
    """클래스의 import path를 정확히 반환한다.

    pytest의 import 모드에 따라 module 이름이 ``tests.integration.*``이 아닌
    ``test_*``로 등록될 수 있으므로 ``cls.__module__``을 기준으로 path를 구성한다.
    이를 통해 ComponentResolver가 새로 import해도 같은 클래스 객체를 가져온다.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def _make_dataset_config(lengths: list[int]) -> dict[str, Any]:
    return {
        "_component_": _component_path(_FakeLengthedDatasetSpec),
        "lengths": lengths,
    }


def _make_collator_config() -> dict[str, Any]:
    return {"_component_": _component_path(_IdentityCollator)}


def _make_sampler_config(
    bucket_size: int | None = None,
    shuffle_buckets: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "_component_": _component_path(LengthGroupedBatchSampler),
        "shuffle_buckets": shuffle_buckets,
        "seed": seed,
        "drop_last": drop_last,
    }
    if bucket_size is not None:
        cfg["bucket_size"] = bucket_size
    return cfg


# ──────────────────────────────────────────────────────────────────────
# 통합 테스트 1) sampler 주입 시 batch 길이 정렬
# ──────────────────────────────────────────────────────────────────────


def test_dataloader_with_length_grouped_sampler() -> None:
    """``data.sampler``로 ``LengthGroupedBatchSampler`` 주입 시 batch 내 길이 정렬.

    bucket 안에서 sample을 길이 오름차순으로 정렬한 뒤 batch로 자르므로,
    각 batch 안의 길이는 비감소(monotonically non-decreasing)여야 한다.
    """
    # 길이 분포: 100 sample, [10..109]를 무작위로 섞은 형태 ─ 결정적 시드 사용
    import random

    rng = random.Random(42)
    lengths = list(range(10, 110))
    rng.shuffle(lengths)

    loaders = create_dataloaders(
        dataset_config=_make_dataset_config(lengths),
        collator_config=_make_collator_config(),
        dataloader_config={
            "batch_size": 4,
            "num_workers": 0,
            "drop_last": False,
        },
        sampler_config=_make_sampler_config(
            bucket_size=16, shuffle_buckets=False, seed=0
        ),
        distributed=False,
    )

    train_loader = loaders["train"]
    assert isinstance(train_loader, DataLoader)
    # batch_sampler 경로로 주입되었는지 확인 — DataLoader가 batch_size 인자를 갖지 않음
    assert train_loader.batch_size is None

    batches: list[list[dict[str, int]]] = list(train_loader)
    assert len(batches) > 0

    for batch in batches:
        batch_lengths = [item["length"] for item in batch]
        # bucket 내부 정렬이므로 batch는 비감소 길이
        assert batch_lengths == sorted(batch_lengths), (
            f"batch lengths not sorted: {batch_lengths}"
        )


# ──────────────────────────────────────────────────────────────────────
# 통합 테스트 2) sampler 미지정 시 기존 shuffle=True 동작 보존
# ──────────────────────────────────────────────────────────────────────


def test_dataloader_without_sampler_uses_default_shuffle() -> None:
    """``sampler_config=None`` (기존 fixture와 동일) → ``shuffle=True`` 경로 보존.

    distributed=False + sampler 미지정 → DataLoader.batch_size가 살아있고
    sampler는 None이며 shuffle은 True (RandomSampler 자동 부착).
    """
    import torch.utils.data as tdata

    lengths = list(range(10, 50))
    loaders = create_dataloaders(
        dataset_config=_make_dataset_config(lengths),
        collator_config=_make_collator_config(),
        dataloader_config={
            "batch_size": 4,
            "num_workers": 0,
            "drop_last": False,
        },
        sampler_config=None,
        distributed=False,
    )

    train_loader = loaders["train"]
    assert isinstance(train_loader, DataLoader)
    # batch_sampler 경로가 아님 — DataLoader는 자체 batch_sampler를 만든다
    assert train_loader.batch_size == 4
    # shuffle=True 경로는 RandomSampler를 자동 부착한다 (PyTorch 표준)
    assert isinstance(train_loader.sampler, tdata.RandomSampler)

    # 실제 iteration도 정상
    batches = list(train_loader)
    total = sum(len(b) for b in batches)
    assert total == len(lengths)


# ──────────────────────────────────────────────────────────────────────
# 통합 테스트 3) sampler 주입 시 dl_kwargs 자동 정리 (DataLoader ValueError 회피)
# ──────────────────────────────────────────────────────────────────────


def test_dataloader_with_sampler_overrides_dl_kwargs() -> None:
    """sampler 주입 시 ``batch_size``/``shuffle``/``drop_last``가 자동으로 dl_kwargs에서 제거.

    PyTorch DataLoader 계약상 ``batch_sampler`` 사용 시 위 3개는 ValueError를
    일으킨다. ``create_dataloaders``가 이 정리를 자동 수행하는지 검증한다.
    Recipe의 ``data.dataloader``에 명시된 batch_size 등은 sampler 책임으로
    위임되므로 사용자에게 추가 부담 없이 작동한다.
    """
    lengths = list(range(20, 70))
    # dl_kwargs에 일부러 모두 명시 — 자동 제거가 일어나지 않으면 ValueError
    dataloader_config = {
        "batch_size": 8,
        "shuffle": True,            # batch_sampler와 상호 배타
        "drop_last": True,          # batch_sampler와 상호 배타
        "num_workers": 0,
    }

    # ValueError가 발생하지 않아야 한다
    loaders = create_dataloaders(
        dataset_config=_make_dataset_config(lengths),
        collator_config=_make_collator_config(),
        dataloader_config=dataloader_config,
        sampler_config=_make_sampler_config(
            bucket_size=16, shuffle_buckets=False, seed=0
        ),
        distributed=False,
    )

    train_loader = loaders["train"]
    assert isinstance(train_loader, DataLoader)
    # batch_sampler 경로 확정
    assert train_loader.batch_size is None
    # batch_size 8이 sampler에 주입되어 batch당 8개씩 (drop_last=False가 sampler 기본)
    batches = list(train_loader)
    assert len(batches) > 0
    # 첫 batch가 정확히 8개 (마지막 batch는 잔여로 작을 수 있음)
    assert len(batches[0]) == 8


# ──────────────────────────────────────────────────────────────────────
# 통합 테스트 4) AssemblyMaterializer.create_dataloaders() end-to-end
# ──────────────────────────────────────────────────────────────────────


def _build_settings_with_sampler(lengths: list[int]) -> Settings:
    """Recipe + Config로 ``Settings``를 빌드한다 (DataSpec.sampler 활성)."""
    recipe = Recipe(
        name="length-bucketed-test",
        task="causal_lm",
        data=DataSpec(
            dataset=_make_dataset_config(lengths),
            collator=_make_collator_config(),
            sampler=_make_sampler_config(
                bucket_size=16, shuffle_buckets=False, seed=0
            ),
            dataloader=DataloaderSpec(
                batch_size=4, num_workers=0, drop_last=False,
                pin_memory=False, persistent_workers=False, prefetch_factor=None,
            ),
        ),
        training=TrainingSpec(epochs=1.0),
        metadata=MetadataSpec(author="test", description="length-bucketed test"),
    )
    config = Config()
    return Settings(recipe=recipe, config=config)


def _materializer(settings: Settings) -> AssemblyMaterializer:
    run_plan = RunPlan(
        command="train",
        mode="sft",
        settings=settings,
        sources=RunSources(),
        overrides=(),
        callback_configs=(),
        validation_scope="training",
        distributed_intent=False,
    )
    return AssemblyMaterializer(AssemblyPlanner.from_run_plan(run_plan))


def test_materializer_creates_dataloader_with_sampler_from_recipe() -> None:
    """AssemblyMaterializer.create_dataloaders() 가 Recipe.data.sampler를 그대로 전달한다.

    Settings → AssemblyMaterializer → create_dataloaders → DataLoader(batch_sampler=...) 경로
    전체가 작동함을 end-to-end로 검증. 본 테스트가 통과하면 사용자가 Recipe의
    ``data.sampler`` 섹션을 추가하는 것만으로 length-bucketed sampler가
    파이프라인에 진입한다.
    """
    lengths = list(range(10, 60))
    settings = _build_settings_with_sampler(lengths)

    # DataSpec.sampler 필드가 schema 단계에서 정상 보관됨
    assert settings.recipe.data.sampler is not None
    assert settings.recipe.data.sampler.component

    materializer = _materializer(settings)
    loaders = materializer.materialize_dataloaders()

    train_loader = loaders["train"]
    assert isinstance(train_loader, DataLoader)
    # batch_sampler 경로 확정
    assert train_loader.batch_size is None

    # bucket 정렬 검증 — batch 내 길이는 비감소
    batches = list(train_loader)
    assert len(batches) > 0
    for batch in batches:
        batch_lengths = [item["length"] for item in batch]
        assert batch_lengths == sorted(batch_lengths)


# ──────────────────────────────────────────────────────────────────────
# 보조: DataSpec.sampler optional 확인 (스키마 회귀)
# ──────────────────────────────────────────────────────────────────────


def test_dataspec_sampler_field_is_optional() -> None:
    """``DataSpec``에 ``sampler`` 필드가 추가되어도 미지정이 기본 (None)이다.

    기존 5+ recipe fixture가 sampler를 가지지 않으므로 None default가 필수.
    U6 회귀 테스트의 사전 검증.
    """
    spec = DataSpec(
        dataset={"_component_": "X"},
        collator={"_component_": "Y"},
    )
    assert spec.sampler is None
