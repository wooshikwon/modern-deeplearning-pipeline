"""ComponentResolver.resolve()의 위치 인자 메커니즘 단위 테스트.

spec-length-bucketed-sampler.md 결정 7에 따른 Unit U4의 검증 대상:

- `ComponentResolver.resolve(config, *args, **extra_kwargs)`는 이미 위치 인자
  메커니즘을 갖추고 있다 (`mdp/settings/resolver.py:44`).
- 그러나 현재 코드베이스에서 이 메커니즘은 실제로 사용되지 않는다 — Optimizer는
  `resolve_partial` 우회 패턴을 쓴다.
- Sampler에서 처음 사용될 예정이므로, **이 패턴이 실제로 보장된다는 사실을
  단위 테스트로 박는다**. resolver 코드 자체는 수정하지 않는다.

테스트 항목:
1. `test_resolve_with_positional_dataset_argument` — 위치 인자 1개 + config kwargs
2. `test_resolve_with_positional_and_extra_kwargs` — 위치 인자 + extra_kwargs
3. `test_resolve_with_unresolved_component_in_kwargs` — nested `_component_`와
   위치 인자가 충돌하지 않음
"""

from __future__ import annotations

from typing import Any

import pytest

from mdp.settings.resolver import ComponentResolver


# ──────────────────────────────────────────────────────────────────────
# 테스트용 Fake 클래스 (모듈 레벨 — _component_ import path로 참조 가능)
# ──────────────────────────────────────────────────────────────────────


class FakeSampler:
    """LengthGroupedBatchSampler를 모방하는 테스트용 Fake.

    실제 sampler와 동일한 시그니처 패턴: `__init__(self, dataset, batch_size, ...)`.
    dataset은 위치 인자, batch_size는 키워드 인자, bucket_size는 config에서.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        bucket_size: int = 128,
    ) -> None:
        self._dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size


class FakeNestedComponent:
    """nested `_component_` 해석이 위치 인자와 충돌하지 않는지 검증용.

    `__init__(self, dataset, sub_component, ...)` 형태. sub_component는
    config에서 nested `_component_`로 들어와 resolve된 인스턴스가 주입된다.
    """

    def __init__(
        self,
        dataset: Any,
        sub_component: Any,
        scale: float = 1.0,
    ) -> None:
        self._dataset = dataset
        self.sub_component = sub_component
        self.scale = scale


class FakeSubComponent:
    """nested resolution 대상."""

    def __init__(self, name: str = "default") -> None:
        self.name = name


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def resolver() -> ComponentResolver:
    return ComponentResolver()


# ──────────────────────────────────────────────────────────────────────
# 위치 인자 메커니즘 테스트
# ──────────────────────────────────────────────────────────────────────


def _component_path(cls: type) -> str:
    """클래스의 import path를 정확히 반환한다.

    pytest 실행 환경에서 테스트 모듈은 rootdir 기반 import 모드 때문에
    `tests.unit.test_*`이 아닌 `test_*`로 등록될 수 있다. 이 경우 문자열
    상수로 `_component_` 경로를 박으면 ComponentResolver.import_class()가
    별개의 module 객체를 새로 import하여 같은 정의의 클래스인데도 isinstance
    검사가 실패한다. `cls.__module__`을 사용하면 pytest가 올린 모듈 이름과
    동일한 경로가 보장되므로 동일 객체가 반환된다.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


class TestResolverPositionalArgs:
    """ComponentResolver.resolve()의 *args 전파 메커니즘을 검증한다.

    sampler 의존성 주입 패턴(`resolver.resolve(sampler_config, dataset,
    batch_size=batch_size)`)이 안정적으로 동작함을 보장한다.
    """

    def test_resolve_with_positional_dataset_argument(
        self, resolver: ComponentResolver
    ) -> None:
        """위치 인자(dataset) + extra_kwargs(batch_size) + config kwargs(bucket_size)
        조합이 정확하게 생성자에 전달된다.

        이는 `create_dataloaders`가 sampler를 만들 때 사용할 핵심 패턴이다::

            resolver.resolve(sampler_config, dataset, batch_size=batch_size)
        """
        config = {
            "_component_": _component_path(FakeSampler),
            "bucket_size": 16,
        }
        fake_dataset = object()  # placeholder — identity로 비교

        sampler = resolver.resolve(config, fake_dataset, batch_size=4)

        assert isinstance(sampler, FakeSampler)
        assert sampler._dataset is fake_dataset
        assert sampler.batch_size == 4
        assert sampler.bucket_size == 16

    def test_resolve_with_positional_and_extra_kwargs(
        self, resolver: ComponentResolver
    ) -> None:
        """위치 인자 1개 + extra_kwargs 2개 조합. 모두 정상 주입.

        config에 정의되지 않은 `batch_size`와 `bucket_size`를 호출 시점에
        extra_kwargs로 전달해도 정상 동작한다 — config의 `bucket_size`가
        없어도 caller가 주입할 수 있는 유연성을 검증한다.
        """
        config = {
            "_component_": _component_path(FakeSampler),
        }
        fake_dataset = object()

        sampler = resolver.resolve(
            config, fake_dataset, batch_size=8, bucket_size=32
        )

        assert isinstance(sampler, FakeSampler)
        assert sampler._dataset is fake_dataset
        assert sampler.batch_size == 8
        assert sampler.bucket_size == 32

    def test_resolve_with_unresolved_component_in_kwargs(
        self, resolver: ComponentResolver
    ) -> None:
        """config의 nested `_component_`가 위치 인자와 충돌 없이 함께 처리된다.

        resolver.resolve()는 config의 dict 값에 대해 재귀적으로 `_component_`를
        해석한다. 이때 호출자가 별도로 위치 인자를 전달해도 nested resolution이
        정상 동작해야 한다 — 두 메커니즘은 직교한다.
        """
        config = {
            "_component_": _component_path(FakeNestedComponent),
            "sub_component": {
                "_component_": _component_path(FakeSubComponent),
                "name": "nested",
            },
            "scale": 2.5,
        }
        fake_dataset = object()

        instance = resolver.resolve(config, fake_dataset)

        assert isinstance(instance, FakeNestedComponent)
        assert instance._dataset is fake_dataset
        assert instance.scale == 2.5
        # nested _component_가 재귀적으로 resolve되어 인스턴스로 주입됨
        assert isinstance(instance.sub_component, FakeSubComponent)
        assert instance.sub_component.name == "nested"
