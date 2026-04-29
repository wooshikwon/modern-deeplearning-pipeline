"""HuggingFaceDataset의 ``Lengthed`` Protocol 구현 단위 테스트.

본 테스트는 sampler 구현 없이 dataset 측의 길이 노출 계약만 검증한다.
실제 HF datasets 로딩 비용을 피하기 위해 ``__new__``로 인스턴스를 만들고
``_ds`` 속성에 fake sequence를 직접 주입한다.
"""

from __future__ import annotations

import pytest

from mdp.data.datasets import HuggingFaceDataset
from mdp.data.samplers import Lengthed


def _make_tokenized_dataset(samples: list[dict[str, list[int]]]) -> HuggingFaceDataset:
    """tokenizer가 적용된 HuggingFaceDataset과 동등한 fake 인스턴스 생성.

    실제 ``__init__``은 ``datasets.load_dataset``을 호출하므로 단위 테스트에서
    피한다. ``_ds`` 속성에 list-of-dict를 박아 ``__getitem__``/``__getlength__``의
    동작 경로만 검증한다.
    """
    ds = HuggingFaceDataset.__new__(HuggingFaceDataset)
    ds._ds = samples
    return ds


def _make_untokenized_dataset(samples: list[dict[str, str]]) -> HuggingFaceDataset:
    """tokenizer 미적용 dataset (input_ids 부재) fake 인스턴스 생성."""
    ds = HuggingFaceDataset.__new__(HuggingFaceDataset)
    ds._ds = samples
    return ds


def test_huggingface_dataset_implements_lengthed():
    """HuggingFaceDataset 인스턴스가 ``Lengthed`` Protocol을 만족한다.

    @runtime_checkable Protocol이므로 isinstance 검사가 의미를 가진다 —
    sampler가 dataset의 길이 노출 가능 여부를 명시적으로 확인할 수 있다.
    """
    ds = _make_tokenized_dataset([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}])
    assert isinstance(ds, Lengthed)


def test_huggingface_dataset_getlength_returns_input_ids_len():
    """각 sample에 대해 ``__getlength__(idx) == len(input_ids)`` 가 성립한다."""
    samples = [
        {"input_ids": [10, 20, 30], "labels": [10, 20, 30]},
        {"input_ids": list(range(100)), "labels": list(range(100))},
        {"input_ids": [42], "labels": [42]},
    ]
    ds = _make_tokenized_dataset(samples)

    assert ds.__getlength__(0) == 3
    assert ds.__getlength__(1) == 100
    assert ds.__getlength__(2) == 1

    # 모든 인덱스에 대해 input_ids 길이와 일치하는지 정칙 검증
    for i, sample in enumerate(samples):
        assert ds.__getlength__(i) == len(sample["input_ids"])


def test_huggingface_dataset_getlength_raises_without_tokenizer():
    """tokenizer 미적용 dataset(input_ids 없음)에서 호출 시 KeyError.

    이는 의도된 동작 — length-bucketed sampler가 부적절한 dataset에 대해
    명시적·즉시 실패하도록 함으로써 silent fallback을 차단한다.
    """
    ds = _make_untokenized_dataset([{"text": "hello world", "label": 0}])

    with pytest.raises((KeyError, ValueError)):
        ds.__getlength__(0)
