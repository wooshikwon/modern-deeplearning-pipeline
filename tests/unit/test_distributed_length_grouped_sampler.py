"""``DistributedLengthGroupedBatchSampler`` (DDP) 단위 테스트.

spec-length-bucketed-sampler.md U3 5건. 단일 프로세스에서 ``num_replicas``/``rank``를
명시 인자로 주입하여 megabatch 패턴을 검증한다 — 실제 DDP process group 없이
단위 테스트가 가능하도록 base sampler가 두 인자를 받도록 설계되어 있다.

1. ``test_disjoint_indices_across_ranks``
2. ``test_same_seed_epoch_deterministic_across_ranks``
3. ``test_set_epoch_changes_partition``
4. ``test_megabatch_alignment``
5. ``test_raises_when_distributed_not_initialized``
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mdp.data.samplers import DistributedLengthGroupedBatchSampler


# ──────────────────────────────────────────────────────────────────────
# Fake dataset
# ──────────────────────────────────────────────────────────────────────


class _LengthedDataset:
    """``Lengthed`` protocol을 구현한 단순 fake dataset.

    ``length_grouped_sampler`` 테스트와 동일한 형태 — 실 토큰화 비용 없이
    sampler의 megabatch 분할·정렬 경로를 단위 테스트하기 위함.
    """

    def __init__(self, lengths: list[int]) -> None:
        self._lengths = list(lengths)

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"x": idx, "length": self._lengths[idx]}

    def __getlength__(self, idx: int) -> int:
        return self._lengths[idx]


def _flatten(batches: list[list[int]]) -> list[int]:
    return [idx for batch in batches for idx in batch]


def _build(
    ds: _LengthedDataset,
    *,
    rank: int,
    num_replicas: int = 4,
    batch_size: int = 4,
    bucket_size: int | None = None,
    shuffle_buckets: bool = False,
    seed: int = 0,
    drop_last: bool = False,
) -> DistributedLengthGroupedBatchSampler:
    return DistributedLengthGroupedBatchSampler(
        ds,
        batch_size=batch_size,
        bucket_size=bucket_size,
        shuffle_buckets=shuffle_buckets,
        seed=seed,
        drop_last=drop_last,
        num_replicas=num_replicas,
        rank=rank,
    )


# ──────────────────────────────────────────────────────────────────────
# 1) 같은 epoch에서 rank 합집합이 전체 dataset (padding 포함)
# ──────────────────────────────────────────────────────────────────────


def test_disjoint_indices_across_ranks() -> None:
    """같은 (seed, epoch)에서 모든 rank의 indices를 모으면 전체 dataset과 일치.

    - 32 sample / num_replicas=4 / batch_size=4 → megabatch_size=16, megabatch 2개.
    - rank 0~3이 각 megabatch에서 자기 chunk(batch_size=4)를 받으므로,
      rank 별 batch 2개씩, 전체 합쳐서 32 indices.
    - sample 손실 없음 (drop_last=False, n이 megabatch_size로 나눠떨어짐).
    """
    n = 32
    lengths = [10 + (i % 30) for i in range(n)]
    ds = _LengthedDataset(lengths)

    all_indices: list[int] = []
    per_rank_batches: list[list[list[int]]] = []
    for r in range(4):
        s = _build(ds, rank=r, num_replicas=4, batch_size=4, seed=0)
        batches = list(iter(s))
        per_rank_batches.append(batches)
        all_indices.extend(_flatten(batches))

    # 전체 dataset 1회 정확히 cover
    assert sorted(all_indices) == list(range(n)), (
        "rank 합집합이 전체 dataset과 달라요 — disjoint partition 보장 실패"
    )

    # rank 간 disjoint (한 sample이 두 rank에 동시에 가지 않음)
    seen: set[int] = set()
    for r, batches in enumerate(per_rank_batches):
        flat = _flatten(batches)
        rank_set = set(flat)
        assert len(rank_set) == len(flat), f"rank {r} 내부 중복 발견"
        assert rank_set.isdisjoint(seen), f"rank {r}가 다른 rank와 겹침"
        seen.update(rank_set)

    # n이 megabatch_size로 나눠떨어지므로 각 rank가 정확히 8 sample 받음
    for r, batches in enumerate(per_rank_batches):
        assert len(_flatten(batches)) == 8, f"rank {r}이 받은 sample 수가 8이 아님"

    # padding 포함 케이스 — n이 megabatch_size로 나눠떨어지지 않을 때
    # 30 sample / num_replicas=4 / batch_size=4 → megabatch 2개 (32),
    # padding 2개. 합집합은 30..31 indices가 한 번 더 등장.
    n_pad = 30
    ds_pad = _LengthedDataset([10 + (i % 30) for i in range(n_pad)])
    all_pad: list[int] = []
    for r in range(4):
        s = _build(ds_pad, rank=r, num_replicas=4, batch_size=4, seed=0)
        all_pad.extend(_flatten(list(iter(s))))

    # padding으로 32 sample이 분배 — 모든 sample이 적어도 1회 등장
    assert set(all_pad) == set(range(n_pad)), (
        "padding 후 합집합이 전체 dataset을 cover하지 못함"
    )
    assert len(all_pad) == 32, (
        f"padding 후 총 분배 sample 수가 32이어야 하는데 {len(all_pad)}"
    )


# ──────────────────────────────────────────────────────────────────────
# 2) 같은 (seed, epoch)에서 rank 간 megabatch partition 일치
# ──────────────────────────────────────────────────────────────────────


def test_same_seed_epoch_deterministic_across_ranks() -> None:
    """같은 (seed, epoch)에서 rank 0과 rank 1이 같은 megabatch partition을 사용.

    검증 방법: rank 0과 rank 1의 동일 step batch가 같은 megabatch에서 추출됨을
    보이기 위해, 두 batch의 합집합이 정렬된 megabatch의 처음 ``2 * batch_size``
    indices와 일치하는지 확인. 추가로 rank 간 disjoint도 함께 확인.
    """
    n = 64
    lengths = [(i * 7) % 50 + 5 for i in range(n)]  # 다양한 길이 분포
    ds = _LengthedDataset(lengths)

    s0 = _build(ds, rank=0, num_replicas=4, batch_size=4, seed=42)
    s1 = _build(ds, rank=1, num_replicas=4, batch_size=4, seed=42)

    batches_0 = list(iter(s0))
    batches_1 = list(iter(s1))

    assert len(batches_0) == len(batches_1), (
        "같은 epoch에서 rank 간 batch 수가 다름 — partition 일치 실패"
    )

    # step별: rank 0의 batch와 rank 1의 batch는 같은 megabatch 내의
    # 인접한 두 chunk여야 한다 — 길이 정렬된 megabatch에서 rank 0이
    # 더 짧은(0..batch_size), rank 1이 그 다음(batch_size..2*batch_size)을 가짐.
    for step, (b0, b1) in enumerate(zip(batches_0, batches_1)):
        # rank 0의 길이는 모두 rank 1의 길이 max 이하 (정렬된 megabatch의 인접 chunk)
        max_len_0 = max(lengths[i] for i in b0)
        min_len_1 = min(lengths[i] for i in b1)
        assert max_len_0 <= min_len_1, (
            f"step {step}: rank 0 max_len({max_len_0}) > rank 1 min_len({min_len_1}) "
            "— 같은 megabatch에서 길이 정렬된 인접 chunk가 아님"
        )
        # disjoint
        assert set(b0).isdisjoint(set(b1)), f"step {step}: rank 0/1이 sample 공유"


# ──────────────────────────────────────────────────────────────────────
# 3) set_epoch — epoch 변경 시 같은 rank 결과 변동
# ──────────────────────────────────────────────────────────────────────


def test_set_epoch_changes_partition() -> None:
    """같은 rank에서 epoch=0과 epoch=1이 다른 indices를 본다.

    seed는 동일, epoch만 바꿈 → ``seed + epoch``가 달라져 셔플 결과 변경 →
    megabatch 구성 달라짐 → 같은 rank가 보는 indices도 달라짐.
    """
    n = 64
    lengths = [(i * 13) % 40 + 5 for i in range(n)]
    ds = _LengthedDataset(lengths)

    s = _build(ds, rank=0, num_replicas=4, batch_size=4, seed=7)

    s.set_epoch(0)
    e0_a = list(iter(s))
    s.set_epoch(0)
    e0_b = list(iter(s))
    s.set_epoch(1)
    e1 = list(iter(s))

    # 결정성: 같은 epoch는 같은 결과
    assert e0_a == e0_b, "같은 (seed, epoch)에서 결과가 달라짐 — 결정성 실패"
    # epoch 변경: 적어도 한 batch라도 달라야 함
    assert e0_a != e1, "epoch 변경에도 결과가 동일 — set_epoch가 반영 안 됨"


# ──────────────────────────────────────────────────────────────────────
# 4) 같은 step에서 rank 간 길이 분포 정렬
# ──────────────────────────────────────────────────────────────────────


def test_megabatch_alignment() -> None:
    """같은 step에서 모든 rank의 batch가 같은 megabatch 안에 머무른다.

    spec 결정 4의 핵심 — DDP all_reduce stragler 제거의 근거.

    megabatch 패턴은 같은 step의 batch_max를 **megabatch_max**로 제한한다 (전체
    dataset_max가 아님). megabatch는 매 step 새로 무작위 sampling되므로, step별
    megabatch_max가 dataset 전체 길이 분포의 평균 부근으로 흩어진다 → step
    latency가 평균에 수렴한다.

    검증 방법:
    1. **인접 chunk 보장**: 같은 step의 rank들의 batch는 같은 megabatch를 길이
       정렬한 후의 인접한 chunk여야 한다 (rank R.max ≤ rank R+1.min).
    2. **step별 max가 dataset_max에 항상 수렴하지 않음**: step별 (모든 rank의
       batch_max 중 max)가 dataset 전체 max보다 충분히 작은 step이 다수 존재해야
       한다 — megabatch 분할의 효과.
    """
    import statistics

    n = 128
    # 길이 분포: [10, 110) — range 100
    lengths = [10 + (i * 17) % 100 for i in range(n)]
    ds = _LengthedDataset(lengths)
    dataset_max = max(lengths)

    num_replicas = 4
    batch_size = 4
    samplers = [
        _build(ds, rank=r, num_replicas=num_replicas, batch_size=batch_size, seed=0)
        for r in range(num_replicas)
    ]
    runs = [list(iter(s)) for s in samplers]
    n_steps = len(runs[0])

    # (1) 인접 chunk 보장 — 모든 step에서 rank R.max ≤ rank R+1.min
    for step in range(n_steps):
        for r in range(num_replicas - 1):
            max_r = max(lengths[i] for i in runs[r][step])
            min_r1 = min(lengths[i] for i in runs[r + 1][step])
            assert max_r <= min_r1, (
                f"step {step}: rank {r}.max({max_r}) > rank {r+1}.min({min_r1}) "
                "— 정렬된 megabatch의 인접 chunk 순서 위배"
            )

    # (2) step별 effective_max (= 모든 rank의 batch_max 중 max = megabatch_max)가
    #     dataset_max보다 작은 step이 다수 존재. random partition이라면 한 step에서
    #     dataset_max가 어느 rank엔 거의 항상 포함되어 effective_max ≈ dataset_max.
    #     megabatch 패턴이면 dataset_max가 들어간 megabatch 1개에서만 effective_max
    #     = dataset_max, 나머지 step은 작아진다.
    step_effective_max = [
        max(max(lengths[i] for i in runs[r][step]) for r in range(num_replicas))
        for step in range(n_steps)
    ]
    n_below_dataset_max = sum(1 for em in step_effective_max if em < dataset_max)
    # 8 step 중 적어도 절반 이상의 step에서 effective_max < dataset_max여야 한다
    assert n_below_dataset_max >= n_steps // 2, (
        f"step_effective_max < dataset_max 인 step이 {n_below_dataset_max}/{n_steps}로 "
        "기대치 (≥ 절반) 미달 — megabatch 분할이 step latency를 분산시키지 못함"
    )

    # (3) step_effective_max의 평균이 dataset_max보다 충분히 작아야 한다 —
    #     평균 step latency가 dataset_max에 갇히지 않음.
    avg_effective_max = statistics.mean(step_effective_max)
    assert avg_effective_max < dataset_max * 0.95, (
        f"avg step_effective_max {avg_effective_max:.2f} 가 dataset_max "
        f"{dataset_max} 의 95%를 초과 — megabatch 분할 효과 부재"
    )


# ──────────────────────────────────────────────────────────────────────
# 5) distributed 미초기화 + num_replicas/rank 미명시 → ValueError
# ──────────────────────────────────────────────────────────────────────


def test_raises_when_distributed_not_initialized() -> None:
    """distributed 미초기화 환경에서 num_replicas/rank를 명시하지 않으면 ValueError.

    실제 process group 초기화 비용을 피하기 위해 ``torch.distributed.is_initialized``
    를 mock하여 미초기화 상태를 시뮬레이션한다.
    """
    ds = _LengthedDataset([10] * 16)

    with patch("torch.distributed.is_initialized", return_value=False):
        # num_replicas/rank 둘 다 미명시
        with pytest.raises(ValueError, match="num_replicas/rank.*process group"):
            DistributedLengthGroupedBatchSampler(
                ds,
                batch_size=2,
                bucket_size=8,
                seed=0,
            )

        # rank만 미명시
        with pytest.raises(ValueError, match="num_replicas/rank.*process group"):
            DistributedLengthGroupedBatchSampler(
                ds,
                batch_size=2,
                bucket_size=8,
                seed=0,
                num_replicas=2,
            )

        # num_replicas만 미명시
        with pytest.raises(ValueError, match="num_replicas/rank.*process group"):
            DistributedLengthGroupedBatchSampler(
                ds,
                batch_size=2,
                bucket_size=8,
                seed=0,
                rank=0,
            )

    # 대조: 둘 다 명시되면 distributed 미초기화여도 정상 작동
    with patch("torch.distributed.is_initialized", return_value=False):
        s = DistributedLengthGroupedBatchSampler(
            ds,
            batch_size=2,
            bucket_size=8,
            seed=0,
            num_replicas=2,
            rank=0,
        )
        batches = list(iter(s))
        assert len(batches) == len(s)
