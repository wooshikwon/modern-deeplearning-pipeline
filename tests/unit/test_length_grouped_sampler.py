"""``LengthGroupedBatchSampler`` (single-GPU) 단위 테스트.

spec-length-bucketed-sampler.md U2 9건:

1. ``test_same_length_samples_grouped_in_one_batch``
2. ``test_variable_length_samples_grouped_within_bucket``
3. ``test_shuffle_buckets_changes_batch_order_with_seed``
4. ``test_lengthed_protocol_used_when_available``
5. ``test_length_fn_fallback_when_protocol_missing``
6. ``test_drop_last_true_drops_partial_final_batch``
7. ``test_bucket_size_larger_than_dataset_falls_back_to_single_bucket``
8. ``test_set_epoch_changes_seed_deterministically``
9. ``test_default_bucket_size_is_batch_size_times_8``
"""

from __future__ import annotations

import pytest

from mdp.data.samplers import LengthGroupedBatchSampler, Lengthed


# ──────────────────────────────────────────────────────────────────────
# Fake datasets
# ──────────────────────────────────────────────────────────────────────


class _LengthedDataset:
    """``Lengthed`` protocol을 구현한 단순 fake dataset.

    각 sample은 ``{"x": int}``로 단순 보관되며, ``__getlength__(idx)``는
    초기화 시 주입된 ``lengths[idx]``를 반환한다. 실제 토큰화 비용 없이
    sampler의 길이 수집 경로를 단위 테스트하기 위해 사용한다.
    """

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = list(lengths)

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"x": idx, "length": self._lengths[idx]}

    def __getlength__(self, idx: int) -> int:
        return self._lengths[idx]


class _PlainDataset:
    """``Lengthed`` protocol을 구현하지 않은 일반 dataset.

    sampler가 ``length_fn`` fallback 경로로 진입하는지 검증할 때 사용한다.
    """

    def __init__(self, samples: list[dict]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        return self._samples[idx]


# ──────────────────────────────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────────────────────────────


def _flatten(batches: list[list[int]]) -> list[int]:
    return [idx for batch in batches for idx in batch]


# ──────────────────────────────────────────────────────────────────────
# 1) 균등 길이 dataset
# ──────────────────────────────────────────────────────────────────────


def test_same_length_samples_grouped_in_one_batch() -> None:
    """모든 sample이 같은 길이면 batch 내 길이 분포가 trivially 동일하다.

    이는 sampler가 정렬 단계에서 길이를 깨뜨리지 않음을 확인하는 sanity check.
    """
    lengths = [50] * 32
    ds = _LengthedDataset(lengths)
    sampler = LengthGroupedBatchSampler(
        ds,
        batch_size=4,
        bucket_size=8,
        shuffle_buckets=True,
        seed=0,
    )

    batches = list(iter(sampler))

    # 모든 batch에서 길이 동일
    for batch in batches:
        batch_lens = [lengths[i] for i in batch]
        assert batch_lens == [50] * len(batch)
    # 전체 sample 손실 없음 (drop_last=False 기본)
    assert sorted(_flatten(batches)) == list(range(32))


# ──────────────────────────────────────────────────────────────────────
# 2) 가변 길이 dataset의 bucket 내부 정렬
# ──────────────────────────────────────────────────────────────────────


def test_variable_length_samples_grouped_within_bucket() -> None:
    """길이 분포 [10..109] 100 sample, batch_size=4, bucket_size=16.

    bucketed sampling의 **상대적 효과**를 검증한다. 무작위 shuffle 후 16개
    bucket을 잘라 정렬했을 때, batch 내부의 길이 spread가 random shuffle
    baseline보다 통계적으로 유의하게 작아야 한다.

    엄밀 보장: 정렬된 bucket을 batch_size=4로 자르므로 한 batch의 길이 spread는
    그 bucket의 길이 분포에서 인접한 4개의 spread 합이다. bucket 길이 분포의
    range를 R이라 하면 batch spread 평균 ≤ R * (4-1)/(16-1) = R*3/15 = R/5.
    실측치가 정렬 효과 없는 random batch의 spread (≈ R 전체)보다 충분히 작아야
    한다 — 본 테스트는 평균 batch spread가 random baseline의 절반 미만임을
    검증한다 (실제로는 1/5 수준).
    """
    lengths = [10 + i for i in range(100)]
    ds = _LengthedDataset(lengths)
    sampler = LengthGroupedBatchSampler(
        ds,
        batch_size=4,
        bucket_size=16,
        shuffle_buckets=False,
        seed=42,
    )
    batches = list(iter(sampler))

    # bucketed: batch 내 길이 차이 평균
    bucketed_spreads = [
        max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches
    ]
    avg_bucketed_spread = sum(bucketed_spreads) / len(bucketed_spreads)

    # random baseline: 4개 sample을 무작위로 골랐을 때 길이 spread의 기대값.
    # uniform [10, 109]에서 4개 추출 시 max-min 기대값 ≈ (4-1)/(4+1) * range = 60.
    # bucketed 평균은 분포 전체 range(99)의 1/5 수준 ≈ 20 부근이어야 한다.
    random_baseline_spread = 60

    assert avg_bucketed_spread < random_baseline_spread / 2, (
        f"bucketed avg spread {avg_bucketed_spread:.1f} should be much less than "
        f"random baseline {random_baseline_spread} — sorting effect missing"
    )

    # 정렬 자체는 정확히 동작 — 각 batch 내부는 오름차순
    for batch in batches:
        batch_lens = [lengths[i] for i in batch]
        assert batch_lens == sorted(batch_lens)

    # sample 손실 없음
    assert sorted(_flatten(batches)) == list(range(100))


# ──────────────────────────────────────────────────────────────────────
# 3) shuffle_buckets — epoch별 batch 순서 변화 + 결정성
# ──────────────────────────────────────────────────────────────────────


def test_shuffle_buckets_changes_batch_order_with_seed() -> None:
    """``shuffle_buckets=True``: epoch 변경 시 batch 순서가 달라진다.

    - 같은 (seed, epoch) → 동일 결과 (결정성)
    - 다른 epoch → batch 순서 변동 (단, 구성 sample set은 다를 수도 같을 수도)
    """
    lengths = [10 + (i % 50) for i in range(64)]
    ds = _LengthedDataset(lengths)

    def _build() -> LengthGroupedBatchSampler:
        return LengthGroupedBatchSampler(
            ds,
            batch_size=4,
            bucket_size=16,
            shuffle_buckets=True,
            seed=7,
        )

    s1 = _build()
    s1.set_epoch(0)
    epoch0_run1 = list(iter(s1))

    s2 = _build()
    s2.set_epoch(0)
    epoch0_run2 = list(iter(s2))

    s3 = _build()
    s3.set_epoch(1)
    epoch1_run = list(iter(s3))

    # 결정성: 같은 (seed, epoch)면 batch 순서·구성 동일
    assert epoch0_run1 == epoch0_run2

    # epoch 변경: 적어도 한 batch라도 달라야 한다 (확률적이지만 64 sample / seed=7
    # 조합으로는 항상 다른 순열 — 실패 시 seed 조정으로 재현 가능)
    assert epoch0_run1 != epoch1_run


# ──────────────────────────────────────────────────────────────────────
# 4) Lengthed protocol 우선
# ──────────────────────────────────────────────────────────────────────


def test_lengthed_protocol_used_when_available() -> None:
    """``Lengthed`` dataset이면 ``length_fn`` 미지정으로도 정상 작동.

    내부 ``_lengths`` 필드가 ``__getlength__`` 결과와 일치함을 확인.
    """
    lengths = [5, 7, 3, 9, 1, 8, 2, 6]
    ds = _LengthedDataset(lengths)
    assert isinstance(ds, Lengthed)  # protocol 만족 사전 검증

    sampler = LengthGroupedBatchSampler(ds, batch_size=2, bucket_size=4, seed=0)

    assert sampler._lengths == lengths
    # 동작 자체도 정상 (sample 손실 없음)
    batches = list(iter(sampler))
    assert sorted(_flatten(batches)) == list(range(8))


# ──────────────────────────────────────────────────────────────────────
# 5) length_fn fallback + 둘 다 없으면 ValueError
# ──────────────────────────────────────────────────────────────────────


def test_length_fn_fallback_when_protocol_missing() -> None:
    """일반 dataset + ``length_fn`` 명시 → fallback 경로로 정상 작동.

    추가로 둘 다 없으면 ``__init__``에서 ValueError로 즉시 실패.
    """
    samples = [{"input_ids": list(range(L))} for L in [5, 7, 3, 9, 1, 8, 2, 6]]
    ds = _PlainDataset(samples)
    assert not isinstance(ds, Lengthed)  # protocol 미구현 사전 검증

    sampler = LengthGroupedBatchSampler(
        ds,
        batch_size=2,
        bucket_size=4,
        length_fn=lambda s: len(s["input_ids"]),
        seed=0,
    )
    assert sampler._lengths == [5, 7, 3, 9, 1, 8, 2, 6]

    # length_fn도 없으면 즉시 실패
    with pytest.raises(ValueError, match="Lengthed dataset.*length_fn"):
        LengthGroupedBatchSampler(ds, batch_size=2, bucket_size=4, seed=0)


# ──────────────────────────────────────────────────────────────────────
# 6) drop_last 동작
# ──────────────────────────────────────────────────────────────────────


def test_drop_last_true_drops_partial_final_batch() -> None:
    """``drop_last=True``: 마지막 incomplete batch가 누락된다. False는 그대로 yield.

    18 sample / batch_size=4 / bucket_size=8 →
    - bucket 0 (8 sample): batch_size=4로 정확히 잘림 → 2 batch
    - bucket 1 (8 sample): 2 batch
    - bucket 2 (2 sample): drop_last=True면 0 batch, False면 1 incomplete batch
    """
    lengths = list(range(18))
    ds = _LengthedDataset(lengths)

    s_keep = LengthGroupedBatchSampler(
        ds, batch_size=4, bucket_size=8, drop_last=False, shuffle_buckets=False, seed=0
    )
    batches_keep = list(iter(s_keep))
    assert len(batches_keep) == 5  # 2 + 2 + 1
    assert len(s_keep) == 5
    # 마지막 batch는 size 2
    incomplete = [b for b in batches_keep if len(b) < 4]
    assert len(incomplete) == 1 and len(incomplete[0]) == 2

    s_drop = LengthGroupedBatchSampler(
        ds, batch_size=4, bucket_size=8, drop_last=True, shuffle_buckets=False, seed=0
    )
    batches_drop = list(iter(s_drop))
    assert len(batches_drop) == 4  # 2 + 2 + 0
    assert len(s_drop) == 4
    # 모든 batch가 size 4
    for b in batches_drop:
        assert len(b) == 4


# ──────────────────────────────────────────────────────────────────────
# 7) bucket_size > len(dataset) → 단일 bucket으로 클램프 (EC3)
# ──────────────────────────────────────────────────────────────────────


def test_bucket_size_larger_than_dataset_falls_back_to_single_bucket() -> None:
    """``bucket_size`` > dataset 크기면 ``min(bucket_size, len)``으로 클램프.

    전체 dataset이 단일 bucket으로 처리되어 batch 내부 길이 차이가 minimal.
    경고 없이 정상 동작.
    """
    lengths = list(range(10))  # 10 sample
    ds = _LengthedDataset(lengths)

    sampler = LengthGroupedBatchSampler(
        ds, batch_size=2, bucket_size=1024, shuffle_buckets=False, seed=0
    )
    # bucket_size가 dataset 크기로 클램프됨
    assert sampler.bucket_size == 10

    batches = list(iter(sampler))
    # 단일 bucket에서 길이 정렬 후 batch 분할 → 전체가 정렬된 순서
    flat = _flatten(batches)
    flat_lens = [lengths[i] for i in flat]
    assert flat_lens == sorted(flat_lens)
    assert len(batches) == 5
    assert len(sampler) == 5


# ──────────────────────────────────────────────────────────────────────
# 8) set_epoch — seed 변경의 결정성
# ──────────────────────────────────────────────────────────────────────


def test_set_epoch_changes_seed_deterministically() -> None:
    """``set_epoch(0)``과 ``set_epoch(1)`` 결과가 다르고, 같은 epoch는 같다.

    epoch 자체는 instance state이므로 같은 sampler에서 set_epoch만 바꿔도
    재호출 시 결정적 결과를 내야 한다.
    """
    lengths = [10 + (i % 30) for i in range(48)]
    ds = _LengthedDataset(lengths)
    sampler = LengthGroupedBatchSampler(
        ds, batch_size=4, bucket_size=8, shuffle_buckets=True, seed=11
    )

    sampler.set_epoch(0)
    e0_a = list(iter(sampler))
    sampler.set_epoch(0)
    e0_b = list(iter(sampler))
    sampler.set_epoch(1)
    e1 = list(iter(sampler))

    assert e0_a == e0_b  # 같은 epoch → 결정적 동일
    assert e0_a != e1    # 다른 epoch → 결과 변동


# ──────────────────────────────────────────────────────────────────────
# 9) bucket_size 기본값
# ──────────────────────────────────────────────────────────────────────


def test_default_bucket_size_is_batch_size_times_8() -> None:
    """``bucket_size=None``이면 ``batch_size * 8``로 자동 결정 (spec 결정 5).

    단 dataset 크기보다 크면 EC3에 따라 클램프됨 — 본 테스트는 dataset이
    충분히 커서 클램프되지 않는 케이스를 검증.
    """
    lengths = list(range(500))  # 500 sample > batch_size*8 = 32
    ds = _LengthedDataset(lengths)

    sampler = LengthGroupedBatchSampler(ds, batch_size=4, bucket_size=None, seed=0)

    assert sampler.bucket_size == 32  # 4 * 8

    # 작은 batch_size로 다시 확인
    s2 = LengthGroupedBatchSampler(ds, batch_size=8, bucket_size=None, seed=0)
    assert s2.bucket_size == 64  # 8 * 8


