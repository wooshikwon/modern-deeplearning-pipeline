"""create_dataloaders — Dataset + DataLoader 파이프라인 조립."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from mdp.data.tokenizer import (
    LABEL_CAUSAL,
    LABEL_COPY,
    LABEL_SEQ2SEQ,
    build_tokenizer,
    derive_label_strategy,
)
from mdp.data.transforms import build_transforms
from mdp.settings.resolver import ComponentResolver


def _select_collator(
    label_strategy: str,
    tokenizer_config: dict[str, Any] | None,
) -> Any | None:
    """label_strategy에 따라 적절한 collator를 선택한다."""
    if tokenizer_config is None:
        return None

    from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["pretrained"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if label_strategy == LABEL_CAUSAL:
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif label_strategy == LABEL_SEQ2SEQ:
        from transformers import DataCollatorForSeq2Seq

        return DataCollatorForSeq2Seq(tokenizer=tokenizer)
    elif label_strategy == LABEL_COPY:
        return DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        return DataCollatorWithPadding(tokenizer=tokenizer)


def create_dataloaders(
    dataset_config: dict[str, Any],
    aug_config: dict[str, Any] | None,
    tokenizer_config: dict[str, Any] | None,
    loader_config: dict[str, Any],
    fields: dict[str, str] | None = None,
    distributed: bool = False,
) -> dict[str, DataLoader]:
    """Dataset과 DataLoader를 조립하여 ``{"train": ..., "val": ...}`` 딕셔너리를 반환한다.

    Args:
        dataset_config: ``_component_`` 패턴을 포함하는 데이터셋 설정.
        aug_config: augmentation 설정 (``train``/``val`` 키 포함 가능).
        tokenizer_config: tokenizer 설정.
        loader_config: DataLoader 설정 (batch_size, num_workers 등).
        fields: ``{role: column_name}`` 매핑. label 전략 파생에 사용.
        distributed: ``True``이면 ``DistributedSampler`` 사용.

    Returns:
        ``{"train": DataLoader, "val": DataLoader}`` 딕셔너리.
        val split이 없으면 ``"val"`` 키가 빠진다.
    """
    resolver = ComponentResolver()

    # label strategy 파생
    label_strategy = derive_label_strategy(fields)

    # augmentation
    train_transform = None
    val_transform = None
    if aug_config is not None:
        train_transform = build_transforms(aug_config.get("train", aug_config))
        val_transform = build_transforms(aug_config.get("val"))

    # tokenizer
    tokenize_fn = build_tokenizer(tokenizer_config, label_strategy=label_strategy)

    # Dataset 클래스 + kwargs 추출
    dataset_cls, dataset_kwargs = resolver.resolve_partial(dataset_config)

    # collate_fn 해석
    collate_fn = None
    loader_kwargs = dict(loader_config)
    collate_cfg = loader_kwargs.pop("collate_fn", None)
    if collate_cfg is not None:
        collate_fn = resolver.resolve(collate_cfg)
    else:
        collate_fn = _select_collator(label_strategy, tokenizer_config)

    # ── DataLoader 공통 설정 ──
    def _make_loader(
        dataset: Any,
        *,
        shuffle: bool = True,
        sampler: Any = None,
        drop_last: bool = False,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=drop_last,
            **loader_kwargs,
        )

    # ── train split ──
    train_kwargs = dict(dataset_kwargs)
    train_kwargs.setdefault("transform", train_transform)
    if tokenize_fn is not None:
        train_kwargs.setdefault("tokenize_fn", tokenize_fn)

    train_dataset = dataset_cls(**train_kwargs)

    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, shuffle=True)

    result: dict[str, DataLoader] = {
        "train": _make_loader(
            train_dataset, shuffle=True, sampler=train_sampler, drop_last=True
        ),
    }

    # ── val split ──
    val_dataset = None
    val_split = dataset_kwargs.get("split")
    if val_split is not None:
        val_kwargs = dict(dataset_kwargs)
        # split 이름 치환: train → validation (HuggingFace 규약)
        if val_split == "train":
            val_kwargs["split"] = "validation"
        else:
            val_kwargs["split"] = "test"
        val_kwargs["transform"] = val_transform
        if tokenize_fn is not None:
            val_kwargs["tokenize_fn"] = tokenize_fn

        try:
            val_dataset = dataset_cls(**val_kwargs)
        except Exception:
            # val split이 존재하지 않으면 무시
            val_dataset = None

    # split 파라미터가 없는 데이터셋 (ImageFolder, CSV): random_split fallback
    if val_dataset is None and train_dataset is not None:
        from torch.utils.data import random_split

        total = len(train_dataset)
        val_size = int(total * 0.2)
        train_size = total - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        # train DataLoader를 재생성 (split된 데이터셋으로)
        result["train"] = _make_loader(
            train_dataset, shuffle=True, sampler=train_sampler, drop_last=True
        )

    if val_dataset is not None:
        val_sampler = None
        if distributed:
            from torch.utils.data.distributed import DistributedSampler

            val_sampler = DistributedSampler(val_dataset, shuffle=False)

        result["val"] = _make_loader(
            val_dataset, shuffle=False, sampler=val_sampler, drop_last=False
        )

    return result
