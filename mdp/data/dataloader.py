"""create_dataloaders — component spec 기반 Dataset + Collator + Sampler 조립.

Dataset/Collator/Sampler를 ComponentResolver로 인스턴스화하고
DataLoader로 감싸는 얇은 조립자.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from torch.utils.data import DataLoader

from mdp.settings.resolver import ComponentResolver

logger = logging.getLogger(__name__)


class ComponentSpecLike(Protocol):
    """Minimal component spec surface consumed by DataLoader assembly."""

    component: str | None
    resolved_component: str | None


def create_dataloaders(
    dataset_config: ComponentSpecLike,
    collator_config: ComponentSpecLike,
    dataloader_config: dict[str, Any] | None = None,
    val_dataset_config: ComponentSpecLike | None = None,
    sampler_config: ComponentSpecLike | None = None,
    distributed: bool = False,
) -> dict[str, DataLoader]:
    """component spec 설정으로 DataLoader 딕셔너리를 생성한다.

    1. ``dataset_config``를 resolve하여 Dataset 인스턴스 생성
    2. ``collator_config``를 resolve하여 collate_fn 생성
    3. ``sampler_config``가 있으면 ``DataLoader(batch_sampler=...)`` 경로,
       없으면 distributed flag에 따라 기존 동작 (DistributedSampler 자동 부착 또는
       ``shuffle=True``) 유지
    4. ``val_dataset_config``가 있으면 val DataLoader도 생성 (val에는 sampler 미적용)

    Args:
        dataset_config: Dataset component spec.
        collator_config: Collator component spec.
        dataloader_config: DataLoader kwargs (batch_size, num_workers 등).
        val_dataset_config: Validation Dataset 설정. None이면 val 비활성.
        sampler_config: (Batch)Sampler component spec.
            지정 시 ``DataLoader(batch_sampler=...)``로 train에 주입되며,
            ``batch_size``/``shuffle``/``drop_last``는 sampler 책임으로 위임된다
            (DataLoader 표준 계약상 ``batch_sampler`` 사용 시 이들은 ValueError를
            일으키므로 dl_kwargs에서 자동 제거). val에는 적용하지 않는다 —
            length grouping은 train에서만 의미가 있다.
        distributed: True이면 sampler 미지정 시 DistributedSampler 자동 부착.
            sampler 지정 시에는 distributed 분기를 sampler가 책임진다.

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
    # train_drop_last는 sampler 미지정 분기(else)에서만 소비된다. sampler 주입
    # 분기에서는 batch_size/shuffle/drop_last를 set comprehension으로 다시 한 번
    # 제거하므로 drop_last를 미리 pop해 둬도 무관하다. 다만 명시 의도가 둘로
    # 나뉘는 redundancy가 약간 있음을 인지한다.
    train_drop_last = dl_kwargs.pop("drop_last", True)
    # DataloaderSpec의 persistent_workers/prefetch_factor은 num_workers=0일 때 문제
    if dl_kwargs.get("num_workers", 0) == 0:
        dl_kwargs.pop("persistent_workers", None)
        dl_kwargs.pop("prefetch_factor", None)

    # ── Train DataLoader ──
    if sampler_config is not None:
        # 사용자 주입 sampler 경로 — distributed 분기는 sampler가 책임진다
        # (LengthGroupedBatchSampler vs DistributedLengthGroupedBatchSampler 선택).
        # 사용자가 single-GPU sampler를 distributed env에서 사용 시 학습은 동작하지만
        # rank 간 partition이 부재하므로 경고 로그 (spec 정합성 표).
        if distributed:
            sampler_class_name = str(
                sampler_config.resolved_component
                or sampler_config.component
                or "<unknown>"
            )
            if "Distributed" not in sampler_class_name:
                logger.warning(
                    "sampler_config '%s' is not a distributed sampler but "
                    "distributed=True. Ranks will not be partitioned by the sampler. "
                    "Consider DistributedLengthGroupedBatchSampler for multi-rank runs.",
                    sampler_class_name,
                )

        batch_size = dl_kwargs.get("batch_size", 1)
        train_batch_sampler = resolver.resolve(
            sampler_config,
            train_ds,
            batch_size=batch_size,
        )
        # DataLoader(batch_sampler=...)는 batch_size/shuffle/drop_last와 상호 배타
        train_dl_kwargs = {
            k: v
            for k, v in dl_kwargs.items()
            if k not in {"batch_size", "shuffle", "drop_last"}
        }
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            **train_dl_kwargs,
        )
    else:
        # 기존 동작 100% 보존 (spec 원칙 4)
        train_sampler = None
        if distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_ds, shuffle=True)

        train_loader = DataLoader(
            train_ds,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
            drop_last=train_drop_last,
            **dl_kwargs,
        )

    result: dict[str, DataLoader] = {"train": train_loader}

    # ── Val DataLoader (sampler 미적용) ──
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
