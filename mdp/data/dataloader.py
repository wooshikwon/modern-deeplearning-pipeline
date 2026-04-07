"""create_dataloaders — _component_ 기반 Dataset + Collator 조립.

Dataset과 Collator를 ComponentResolver로 인스턴스화하고
DataLoader로 감싸는 얇은 조립자.
"""

from __future__ import annotations

import logging
from typing import Any

from torch.utils.data import DataLoader

from mdp.settings.resolver import ComponentResolver

logger = logging.getLogger(__name__)


def create_dataloaders(
    dataset_config: dict[str, Any],
    collator_config: dict[str, Any],
    dataloader_config: dict[str, Any] | None = None,
    val_dataset_config: dict[str, Any] | None = None,
    distributed: bool = False,
) -> dict[str, DataLoader]:
    """_component_ 설정으로 DataLoader 딕셔너리를 생성한다.

    1. ``dataset_config``를 resolve하여 Dataset 인스턴스 생성
    2. ``collator_config``를 resolve하여 collate_fn 생성
    3. DataLoader 조립 (batch_size, num_workers, sampler 등)
    4. ``val_dataset_config``가 있으면 val DataLoader도 생성

    Args:
        dataset_config: ``_component_`` 키를 포함하는 Dataset 설정.
        collator_config: ``_component_`` 키를 포함하는 Collator 설정.
        dataloader_config: DataLoader kwargs (batch_size, num_workers 등).
        val_dataset_config: Validation Dataset 설정. None이면 val 비활성.
        distributed: True이면 DistributedSampler 사용.

    Returns:
        ``{"train": DataLoader}`` 또는 ``{"train": DataLoader, "val": DataLoader}``.
    """
    resolver = ComponentResolver()

    # ── Dataset 인스턴스화 ──
    train_ds = resolver.resolve(dataset_config)

    # ── Collator 인스턴스화 ──
    collate_fn = resolver.resolve(collator_config)

    # ── DataLoader kwargs ──
    dl_kwargs = dict(dataloader_config or {})
    train_drop_last = dl_kwargs.pop("drop_last", True)
    # DataloaderSpec의 persistent_workers/prefetch_factor은 num_workers=0일 때 문제
    if dl_kwargs.get("num_workers", 0) == 0:
        dl_kwargs.pop("persistent_workers", None)
        dl_kwargs.pop("prefetch_factor", None)

    # ── Train DataLoader ──
    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)

    result: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
            drop_last=train_drop_last,
            **dl_kwargs,
        ),
    }

    # ── Val DataLoader ──
    if val_dataset_config is not None:
        val_ds = resolver.resolve(val_dataset_config)
        val_sampler = None
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            val_sampler = DistributedSampler(val_ds, shuffle=False)

        result["val"] = DataLoader(
            val_ds,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            drop_last=False,
            **dl_kwargs,
        )
    else:
        logger.info("val_dataset 미지정: Validation 비활성화.")

    return result
