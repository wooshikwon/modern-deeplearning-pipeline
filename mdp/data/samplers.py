"""Sampler `_component_`의 표준 계약과 구현.

본 모듈은 length-bucketed sampling을 위한 인터페이스 계약(`Lengthed` Protocol)과
공통 base 클래스(`BaseLengthSampler`), single-GPU 환경에서 동작하는
`LengthGroupedBatchSampler`, multi-rank 환경의
`DistributedLengthGroupedBatchSampler`를 제공한다.

두 sampler 모두 ``DataLoader(batch_sampler=...)`` 슬롯에 직접 주입할 수 있으며,
recipe의 ``data.sampler`` 섹션에서 ``_component_`` 패턴으로 선택된다.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

import torch
from torch.utils.data import BatchSampler

logger = logging.getLogger(__name__)


@runtime_checkable
class Lengthed(Protocol):
    """Sample 길이를 효율적으로 노출하는 dataset의 구조적 계약.

    `__getlength__(idx)`가 i번째 sample의 토큰/시퀀스 길이를 int로 반환한다.
    이미 토큰화된 dataset(예: HuggingFaceDataset with tokenizer set)은 trivially 구현 가능 —
    내부 저장소에서 토큰 시퀀스 길이만 읽으면 된다.

    Length-bucketed sampler는 이 protocol을 1순위로 사용하고,
    미구현 dataset에 대해서는 사용자가 sampler config에 length_fn을 명시해야 한다.
    """

    def __getlength__(self, idx: int) -> int: ...


class BaseLengthSampler(BatchSampler):
    """Length-bucketed sampling의 공통 base.

    - dataset에서 sample별 길이를 수집한다 (1순위: ``__getlength__``, fallback: ``length_fn``).
    - 첫 epoch 길이 수집 결과를 ``__init__``에서 1회 캐싱한다 (cold path는 1회만).
    - 자식 클래스(`LengthGroupedBatchSampler`,
      `DistributedLengthGroupedBatchSampler`)가 ``__iter__``를 구현해
      bucket → 정렬 → batch yield 흐름을 정의한다.

    Note — 부모 ``BatchSampler.__init__``을 호출하지 않는 이유:
        본 클래스는 ``BatchSampler``를 ``isinstance`` 호환과 PyTorch ``DataLoader``의
        ``batch_sampler=`` slot 호환을 위해서만 상속한다. PyTorch ``BatchSampler``는
        실질적으로 ``(sampler, batch_size, drop_last)`` 보관용 구조이지만,
        ``__init__``에 ``sampler``를 명시 전달해야 하고 그렇지 않으면 ``SequentialSampler``
        같은 기본 inner sampler가 부착되어 본 클래스의 bucket-shuffle 알고리즘과
        독립적인 index 흐름이 만들어진다. 따라서 ``super().__init__()``을 호출하지 않고
        ``batch_size``/``drop_last`` 등 필요한 속성만 본 클래스가 직접 둔다. DataLoader는
        batch_sampler를 ``Iterable[list[int]]``로만 소비하므로 동작 계약상 안전하다.

    Args:
        dataset: 길이 수집 대상 dataset. ``Lengthed`` protocol을 구현하면 1순위로
            사용되고, 그렇지 않으면 ``length_fn``으로 fallback. 둘 다 없으면 ValueError.
        batch_size: 한 batch 당 sample 수.
        bucket_size: bucket 한 단위 크기. None이면 ``batch_size * 8``로 자동 결정
            (HF Transformers ``LengthGroupedSampler`` 권장의 보수 버전, spec 결정 5).
        shuffle_buckets: True면 epoch 내부에서 batch 순서를 한 번 더 셔플한다.
        length_fn: ``Lengthed`` protocol 미구현 dataset에 대한 fallback. sample 하나를
            받아 길이(int)를 반환하는 callable. protocol이 우선이며, length_fn이
            적용된 경우에만 cold path warning을 한 번 남긴다.
        seed: 결정적 셔플의 base seed. 실제 generator seed는 ``seed + epoch``.
        drop_last: True면 batch_size 미만의 마지막 batch를 drop한다.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        bucket_size: int | None = None,
        shuffle_buckets: bool = True,
        length_fn: Callable[[Any], int] | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._dataset = dataset
        self.batch_size = batch_size
        self.shuffle_buckets = shuffle_buckets
        self._length_fn = length_fn
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # 기본값: batch_size * 8 (spec 결정 5)
        requested_bucket_size = (
            bucket_size if bucket_size is not None else batch_size * 8
        )
        if requested_bucket_size <= 0:
            raise ValueError(
                f"bucket_size must be positive, got {requested_bucket_size}"
            )

        # ── 길이 수집 (1순위: __getlength__, fallback: length_fn) ──
        self._lengths: list[int] = self._collect_lengths()

        # EC3: bucket_size가 dataset보다 크면 단일 bucket으로 클램프 (경고 불필요)
        self.bucket_size = min(requested_bucket_size, max(len(self._lengths), 1))

        # 부모 BatchSampler.__init__은 호출하지 않는다 — 기본 SequentialSampler 부착이
        # 본 클래스의 bucket-shuffle 로직을 가리지 않도록 sampler 속성을 직접 둔다.
        # PyTorch BatchSampler는 단지 (sampler, batch_size, drop_last) 보관용이라
        # 본 클래스는 BatchSampler를 isinstance 호환 차원에서만 상속한다.

    def _collect_lengths(self) -> list[int]:
        """전체 sample의 길이 list를 수집해 반환한다.

        우선 순위:
        1. ``Lengthed`` protocol (``__getlength__``) 구현 → 인덱스 순회.
        2. ``length_fn`` 명시 → ``dataset[i]`` 샘플 1회 순회.
        3. 둘 다 없음 → ValueError.

        ``length_fn`` 경로는 cold path (sample마다 토크나이즈/디코드 비용 가능)이므로
        한 번만 warning 로그를 남긴다. 결과는 호출자(``__init__``)가 캐싱한다.
        """
        try:
            n = len(self._dataset)
        except TypeError as exc:  # streaming dataset 등 __len__ 미지원
            raise TypeError(
                "BaseLengthSampler requires a dataset with __len__; "
                "streaming datasets are not supported."
            ) from exc

        if isinstance(self._dataset, Lengthed):
            return [int(self._dataset.__getlength__(i)) for i in range(n)]

        if self._length_fn is not None:
            logger.warning(
                "Length collection via length_fn fallback — this is O(N) per dataset. "
                "Consider implementing __getlength__ on your dataset class for faster "
                "initialization."
            )
            return [int(self._length_fn(self._dataset[i])) for i in range(n)]

        raise ValueError(
            "BaseLengthSampler requires either a Lengthed dataset (implementing "
            "__getlength__) or an explicit length_fn callable. Neither was provided."
        )

    def set_epoch(self, epoch: int) -> None:
        """현재 epoch을 설정한다 (DistributedSampler와 같은 인터페이스).

        다음 ``__iter__`` 호출이 ``seed + epoch`` 기반 결정적 셔플을 수행하도록 한다.
        """
        self.epoch = int(epoch)


class LengthGroupedBatchSampler(BaseLengthSampler):
    """Single-GPU 환경에서 길이 grouped sampling을 수행하는 BatchSampler.

    알고리즘 (spec U2):

    1. 전체 indices를 무작위로 셔플 (``seed + epoch`` 결정적 generator)
    2. 셔플된 indices를 ``bucket_size`` 단위로 chunk
    3. 각 bucket 내부에서 길이 기준 오름차순으로 정렬 (Python ``sorted`` — stable)
    4. 정렬된 bucket을 ``batch_size`` 단위로 잘라 batch indices list 생성
       - ``drop_last=True``이면 batch_size 미만의 마지막 batch는 drop
    5. ``shuffle_buckets=True``면 batch들의 순서를 같은 generator로 한 번 더 셔플
       (epoch 내부 변동성 확보)
    """

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self._lengths)
        if n == 0:
            return

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # (1) 전체 indices 무작위 셔플 — torch.randperm(generator=g)는 결정적
        perm = torch.randperm(n, generator=g).tolist()

        # (2)~(4) bucket → 정렬 → batch
        batches: list[list[int]] = []
        for start in range(0, n, self.bucket_size):
            bucket = perm[start : start + self.bucket_size]
            # 길이 기준 오름차순 정렬 (stable). 동률은 perm 순서 보존
            bucket.sort(key=lambda idx: self._lengths[idx])

            for b_start in range(0, len(bucket), self.batch_size):
                batch = bucket[b_start : b_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        # (5) shuffle_buckets — batch들의 순서를 한 번 더 셔플 (epoch 내부)
        if self.shuffle_buckets and len(batches) > 1:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Yield되는 batch 총 개수.

        bucket마다 PyTorch ``BatchSampler``의 표준 공식을 적용하고 합산한다.

        - ``drop_last=True``: bucket 당 ``bucket_len // batch_size``
        - ``drop_last=False``: bucket 당 ``ceil(bucket_len / batch_size)``
        """
        n = len(self._lengths)
        if n == 0:
            return 0

        total = 0
        for start in range(0, n, self.bucket_size):
            bucket_len = min(self.bucket_size, n - start)
            if self.drop_last:
                total += bucket_len // self.batch_size
            else:
                total += (bucket_len + self.batch_size - 1) // self.batch_size
        return total


class DistributedLengthGroupedBatchSampler(BaseLengthSampler):
    """Multi-rank 환경에서 megabatch 패턴으로 길이 grouped sampling을 수행한다.

    spec 결정 4의 알고리즘:

    1. 모든 rank가 같은 ``seed + epoch``로 전체 indices를 결정적으로 셔플한다.
       동기화 메시지 없이 모든 rank가 동일한 셔플 결과를 갖는다.
    2. ``len(dataset)``이 ``num_replicas * batch_size``로 나눠떨어지지 않으면
       DistributedSampler 표준 패턴을 따라 셔플된 indices의 앞부분을 반복하여
       megabatch 경계까지 padding한다 — ``__len__``이 모든 rank에서 정확히
       같은 batch 수를 보장한다.
    3. 패딩된 indices를 ``num_replicas * batch_size`` 단위 megabatch로 분할.
    4. 각 megabatch 내부에서 길이 기준 오름차순 정렬.
    5. 정렬된 megabatch를 ``batch_size`` 단위 ``num_replicas`` 조각으로 자르고,
       rank R이 R번째 조각만 자신의 batch로 가져간다.
    6. ``shuffle_buckets=True``면 megabatch 순서를 한 번 더 결정적으로 셔플
       (모든 rank에서 동일 순서) — step 진행에 따른 길이 분포 편향 방지.

    위 알고리즘의 핵심은 **같은 step에서 모든 rank가 같은 megabatch를 공유**한다는
    점이다. rank 간 batch는 megabatch 내 길이 정렬의 인접 chunk이므로, batch_max의
    rank 간 격차가 megabatch 내부 길이 spread로 제한된다 → DDP ``all_reduce``의
    stragler가 사라진다.

    Args:
        dataset: 길이 수집 대상 dataset (``BaseLengthSampler`` 참조).
        batch_size: rank 1대당 한 step의 sample 수.
        bucket_size: **본 클래스에서는 사용되지 않는다 (silent ignore)**. 분할 단위는
            megabatch (``num_replicas * batch_size``)로 고정이며, ``BaseLengthSampler``의
            인터페이스 일관성과 hyperparameter 검증(positive int)을 위해서만 받는다.
            recipe에 명시하면 ``__init__``에서 일회성 warning이 출력된다 — single-GPU
            ``LengthGroupedBatchSampler``의 ``bucket_size``와 의미가 다르다는 점에 유의.
        shuffle_buckets: True면 megabatch 순서를 결정적으로 셔플.
        length_fn: ``BaseLengthSampler`` 참조.
        seed: 결정적 셔플의 base seed. 모든 rank가 같은 값을 공유해야 한다.
        drop_last: True면 padding 없이 마지막 미완성 megabatch를 drop. False(기본)면
            padding으로 모든 sample을 처리한다 (DistributedSampler 표준 동작).
        num_replicas: 전체 rank 수. None이면 ``torch.distributed.get_world_size()``
            로 조회. distributed 미초기화 + None이면 ValueError.
        rank: 현재 rank id. None이면 ``torch.distributed.get_rank()``로 조회.
            distributed 미초기화 + None이면 ValueError.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        bucket_size: int | None = None,
        shuffle_buckets: bool = True,
        length_fn: Callable[[Any], int] | None = None,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        # num_replicas / rank 결정 — 미명시면 torch.distributed 조회
        resolved_replicas, resolved_rank = self._resolve_distributed_meta(
            num_replicas, rank
        )
        self.num_replicas = resolved_replicas
        self.rank = resolved_rank

        # bucket_size는 본 클래스에서 사용되지 않는다 (megabatch가 분할 단위).
        # 사용자가 명시적으로 전달했다면 single-GPU sampler와 의미가 다름을 알린다.
        # rank 0에서만 출력하여 multi-rank 로그 중복을 피한다.
        if bucket_size is not None and resolved_rank == 0:
            logger.warning(
                "DistributedLengthGroupedBatchSampler: bucket_size=%s is ignored. "
                "The split unit is the megabatch (num_replicas * batch_size = %d * %d); "
                "bucket_size only takes effect in the single-GPU "
                "LengthGroupedBatchSampler.",
                bucket_size,
                resolved_replicas,
                batch_size,
            )

        # base 클래스가 길이 수집·캐싱·hyperparameter 검증·EC3 클램프를 처리한다
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            bucket_size=bucket_size,
            shuffle_buckets=shuffle_buckets,
            length_fn=length_fn,
            seed=seed,
            drop_last=drop_last,
        )

        self._megabatch_size = self.num_replicas * self.batch_size

    @staticmethod
    def _resolve_distributed_meta(
        num_replicas: int | None,
        rank: int | None,
    ) -> tuple[int, int]:
        """``num_replicas``/``rank``를 결정한다.

        명시값 우선, 미명시 시 ``torch.distributed``에서 조회. distributed가
        초기화되지 않았고 둘 중 하나라도 미명시면 ValueError를 즉시 던진다 —
        spec의 정합성 표 참조.
        """
        if num_replicas is None or rank is None:
            if not (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            ):
                raise ValueError(
                    "DistributedLengthGroupedBatchSampler requires either explicit "
                    "num_replicas/rank arguments or an initialized "
                    "torch.distributed process group. Neither was provided."
                )
            if num_replicas is None:
                num_replicas = torch.distributed.get_world_size()
            if rank is None:
                rank = torch.distributed.get_rank()

        # Type narrowing: above branches guarantee both are int at this point.
        assert num_replicas is not None and rank is not None

        if num_replicas <= 0:
            raise ValueError(
                f"num_replicas must be positive, got {num_replicas}"
            )
        if not (0 <= rank < num_replicas):
            raise ValueError(
                f"rank must be in [0, {num_replicas}), got {rank}"
            )
        return int(num_replicas), int(rank)

    def _padded_indices(self, perm: list[int]) -> list[int]:
        """megabatch 경계까지 indices를 padding하거나 잘라낸다.

        - ``drop_last=False`` (기본): DistributedSampler 표준 패턴 — 부족분을
          셔플된 indices의 앞쪽에서 반복해서 채운다. 모든 sample이 적어도 한 번
          포함됨.
        - ``drop_last=True``: 마지막 미완성 megabatch를 잘라낸다.
        """
        n = len(perm)
        mb = self._megabatch_size
        if n == 0:
            return []

        if self.drop_last:
            usable = (n // mb) * mb
            return perm[:usable]

        # 패딩 — DistributedSampler 표준 (indices[: padding_size] 반복)
        remainder = n % mb
        if remainder == 0:
            return perm
        padding_size = mb - remainder
        # 부족분이 perm 길이보다 클 수 있는 corner case (작은 dataset + 큰 mb).
        # repeat는 cycle이 필요 — 여러 번 이어붙인다.
        if padding_size <= n:
            return perm + perm[:padding_size]
        # padding_size > n: perm을 여러 번 이어붙여 채운다.
        repeats = (padding_size + n - 1) // n
        extended = perm + (perm * repeats)[:padding_size]
        return extended

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self._lengths)
        if n == 0:
            return

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # (1) 전체 indices 결정적 셔플 — 모든 rank가 동일 결과
        perm = torch.randperm(n, generator=g).tolist()

        # (2) megabatch 경계까지 padding (또는 drop)
        padded = self._padded_indices(perm)
        if not padded:
            return

        # (3)~(5) megabatch 분할 → 길이 정렬 → rank별 chunk
        mb = self._megabatch_size
        bs = self.batch_size
        rank = self.rank
        rank_start = rank * bs
        rank_end = rank_start + bs

        rank_batches: list[list[int]] = []
        for mb_start in range(0, len(padded), mb):
            megabatch = padded[mb_start : mb_start + mb]
            # 길이 오름차순 정렬 (stable). 동률은 megabatch 내 원순서 보존
            megabatch.sort(key=lambda idx: self._lengths[idx])
            # rank의 chunk만 추출 (megabatch는 rank 0..num_replicas-1로 균등 분할)
            rank_batches.append(megabatch[rank_start:rank_end])

        # (6) shuffle_buckets — megabatch 순서 결정적 셔플 (모든 rank 동일 순서)
        if self.shuffle_buckets and len(rank_batches) > 1:
            order = torch.randperm(len(rank_batches), generator=g).tolist()
            rank_batches = [rank_batches[i] for i in order]

        for batch in rank_batches:
            yield batch

    def __len__(self) -> int:
        """현재 rank가 받는 batch 수.

        megabatch 1개 = rank 1개당 batch 1개. 따라서 batch 수는 megabatch 수와 같다.

        - ``drop_last=False``: ``ceil(n / megabatch_size)``
        - ``drop_last=True``: ``n // megabatch_size``
        """
        n = len(self._lengths)
        if n == 0:
            return 0
        mb = self._megabatch_size
        if self.drop_last:
            return n // mb
        return (n + mb - 1) // mb
