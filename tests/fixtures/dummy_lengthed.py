"""Synthetic fixture for sanity script — in-memory lengthed dataset + padding collator.

본 모듈은 ``scripts/measure_padding_ratio.py``의 동작을 외부 데이터·토크나이저
다운로드 없이 즉시 검증할 수 있도록 합성 dataset과 padding collator를 제공한다.

사용처:
    tests/fixtures/recipes_sanity/dummy-lengthed-causal.yaml의 ``data.dataset``
    및 ``data.collator``의 ``_component_`` 타깃.

설계:
    - ``SyntheticLengthedDataset``는 ``__getlength__`` Protocol을 구현하여
      ``LengthGroupedBatchSampler``의 1순위 길이 노출 경로(cold path 회피)를
      그대로 검증한다.
    - ``PaddingAttentionMaskCollator``는 batch 내 max 길이로 zero padding하고
      ``attention_mask``를 생성한다 — 사용자가 일반적인 causal LM training에서
      쓰는 collator의 최소 골격.
"""

from __future__ import annotations

import random
from typing import Any

import torch


class SyntheticLengthedDataset:
    """미리 결정된 길이 분포를 갖는 합성 dataset.

    각 sample은 임의 길이의 ``input_ids`` (모두 1로 채워진 dummy token)와
    동일 길이의 ``labels``를 반환한다. ``__getlength__`` Protocol을 구현하므로
    LengthGroupedBatchSampler가 cold path(``length_fn`` fallback) 없이
    바로 길이 list를 수집한다.

    Args:
        num_samples: dataset 크기
        min_length: sample 최소 길이 (inclusive)
        max_length: sample 최대 길이 (inclusive)
        seed: 길이 분포의 결정적 seed
    """

    def __init__(
        self,
        num_samples: int = 256,
        min_length: int = 100,
        max_length: int = 500,
        seed: int = 42,
    ) -> None:
        if min_length <= 0 or max_length < min_length:
            raise ValueError(
                f"잘못된 길이 범위: min={min_length}, max={max_length}"
            )
        rng = random.Random(seed)
        self._lengths = [rng.randint(min_length, max_length) for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self._lengths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        L = self._lengths[idx]
        ids = [1] * L
        return {"input_ids": ids, "labels": list(ids)}

    def __getlength__(self, idx: int) -> int:
        return self._lengths[idx]


class PaddingAttentionMaskCollator:
    """batch 내 max 길이로 zero padding 후 ``attention_mask``를 생성한다.

    출력은 stacked tensor로 ``{"input_ids": (B, S), "attention_mask": (B, S),
    "labels": (B, S)}``. ``measure_padding_ratio.py``가 ``attention_mask.sum(-1)``
    로 sample-별 유효 토큰 수를 추출한다.
    """

    def __init__(self, pad_to: str = "max_in_batch", pad_token_id: int = 0) -> None:
        if pad_to != "max_in_batch":
            raise ValueError(
                "PaddingAttentionMaskCollator는 pad_to='max_in_batch'만 지원한다"
            )
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("collator: features가 비어있다")
        max_len = max(len(f["input_ids"]) for f in features)
        bsz = len(features)

        input_ids = torch.full(
            (bsz, max_len), fill_value=self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        labels = torch.full((bsz, max_len), fill_value=-100, dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"]
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(f["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
