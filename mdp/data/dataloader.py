"""create_dataloaders — source 기반 Dataset + DataLoader 파이프라인 조립.

datasets.load_dataset()로 통합 로딩하고, fields 매핑으로 컬럼을
역할명(text, label, target 등)에 맞춘 뒤, task-derived 전처리를 적용한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from mdp.data.loader import load_data
from mdp.data.tokenizer import (
    LABEL_ALIGN,
    LABEL_CAUSAL,
    LABEL_COPY,
    LABEL_PREFERENCE,
    LABEL_SEQ2SEQ,
    build_tokenizer,
    derive_label_strategy,
)
from mdp.data.transforms import build_transforms

logger = logging.getLogger(__name__)


def _select_collator(
    label_strategy: str,
    tokenizer_config: dict[str, Any] | None,
    tokenizer: Any | None = None,
) -> Any | None:
    """label_strategy에 따라 적절한 collator를 선택한다."""
    if label_strategy == LABEL_PREFERENCE and tokenizer_config is None:
        raise ValueError(
            "preference 학습(DPO/GRPO)에는 data.tokenizer 설정이 필수입니다. "
            "recipe의 data.tokenizer.pretrained을 지정하세요."
        )
    if tokenizer_config is None:
        return None

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["pretrained"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if label_strategy == LABEL_PREFERENCE:
        from mdp.data.collators import PreferenceCollator

        max_length = tokenizer_config.get("max_length", 2048)
        return PreferenceCollator(tokenizer=tokenizer, max_length=max_length)

    if label_strategy == LABEL_CAUSAL:
        from transformers import DataCollatorForLanguageModeling

        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    elif label_strategy == LABEL_SEQ2SEQ:
        from transformers import DataCollatorForSeq2Seq

        return DataCollatorForSeq2Seq(tokenizer=tokenizer)
    elif label_strategy == LABEL_COPY:
        from transformers import DataCollatorWithPadding

        return DataCollatorWithPadding(tokenizer=tokenizer)
    elif label_strategy == LABEL_ALIGN:
        from transformers import DataCollatorWithPadding

        return DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        # LABEL_NONE (이미지 등): 기본 torch collate 사용
        return None


# ── Source loading ──


def _detect_format(source: str, fmt: str) -> str:
    """source 경로와 format 힌트로 datasets 로딩 포맷을 결정한다."""
    if fmt != "auto":
        return fmt
    ext = Path(source).suffix.lower()
    return {
        ".csv": "csv", ".tsv": "csv",
        ".json": "json", ".jsonl": "json",
        ".parquet": "parquet",
    }.get(ext, "json")


def _load_source(
    source: str,
    split: str | dict[str, Any],
    *,
    subset: str | None = None,
    streaming: bool = False,
    data_files: str | dict[str, str] | None = None,
    fmt: str = "auto",
) -> Any:
    """source 문자열에서 HuggingFace Dataset을 로드한다.

    - 로컬 디렉토리 → imagefolder
    - 로컬 파일 (.csv, .json, .jsonl, .parquet) → 해당 포맷
    - 그 외 → HuggingFace Hub 이름
    """
    from datasets import load_dataset

    split_str = split if isinstance(split, str) else None
    path = Path(source)

    if path.exists():
        if path.is_dir():
            resolved_fmt = fmt if fmt != "auto" else "imagefolder"
            return load_dataset(
                resolved_fmt, data_dir=source, split=split_str,
                streaming=streaming,
            )
        else:
            resolved_fmt = _detect_format(source, fmt)
            return load_dataset(
                resolved_fmt,
                data_files=data_files or source,
                split=split_str,
                streaming=streaming,
            )
    else:
        # HuggingFace Hub
        kwargs: dict[str, Any] = {}
        if subset is not None:
            kwargs["name"] = subset
        if data_files is not None:
            kwargs["data_files"] = data_files
        return load_dataset(source, split=split_str, streaming=streaming, **kwargs)


def _rename_columns(ds: Any, fields: dict[str, str] | None) -> Any:
    """fields의 {role: column_name} 매핑으로 컬럼을 역할명으로 리네임한다.

    role과 column_name이 같으면 리네임하지 않는다.
    """
    if not fields:
        return ds
    rename_map = {col: role for role, col in fields.items() if col != role}
    if rename_map and hasattr(ds, "rename_columns"):
        ds = ds.rename_columns(rename_map)
    return ds


def _infer_val_split(train_split: str) -> str:
    """train split 이름에서 val split 이름을 추론한다."""
    return "validation" if train_split == "train" else "test"


# ── Public API ──


def create_dataloaders(
    source: str,
    fields: dict[str, str] | None = None,
    *,
    split: str | dict[str, Any] = "train",
    subset: str | None = None,
    streaming: bool = False,
    data_files: str | dict[str, str] | None = None,
    fmt: str = "auto",
    aug_config: dict[str, Any] | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    loader_config: dict[str, Any] | None = None,
    distributed: bool = False,
    val_split: str | None = "auto",
) -> dict[str, DataLoader]:
    """source에서 데이터를 로드하고 DataLoader 딕셔너리를 반환한다.

    1. ``datasets.load_dataset()``로 source 로딩 (HF Hub / 로컬 파일 / 디렉토리)
    2. fields ``{role: column_name}`` 매핑으로 컬럼 리네이밍
    3. fields roles로부터 label_strategy 파생 → 전처리 자동 결정
    4. ``load_data``로 transform/tokenization 적용
    5. DataLoader 조립

    Args:
        source: HF Hub 이름 또는 로컬 경로.
        fields: ``{role: column_name}`` 매핑. role이 전처리를 결정.
        split: 학습 split 이름. 기본 ``"train"``.
        subset: HF 데이터셋의 config/subset 이름.
        streaming: 스트리밍 모드 여부.
        data_files: 로컬 파일 지정 (HF load_dataset data_files).
        fmt: 로컬 파일 포맷. ``"auto"``이면 확장자로 추론.
        aug_config: augmentation 설정 (``train``/``val`` 키 포함 가능).
        tokenizer_config: tokenizer 설정.
        loader_config: DataLoader 설정 (batch_size, num_workers 등).
        distributed: ``True``이면 ``DistributedSampler`` 사용.
        val_split: val split 제어. ``"auto"``이면 train split에서 추론,
            ``None``이면 비활성화, 문자열이면 해당 split 직접 사용.

    Returns:
        ``{"train": DataLoader, "val": DataLoader}`` 딕셔너리.
    """
    loader_config = loader_config or {}

    # ── label strategy ──
    label_strategy = derive_label_strategy(fields)

    # ── transforms / tokenizer ──
    train_transform = None
    val_transform = None
    if aug_config is not None:
        train_transform = build_transforms(aug_config.get("train", aug_config))
        val_transform = build_transforms(aug_config.get("val"))

    # ── shared tokenizer instance ──
    tokenizer_instance = None
    if tokenizer_config is not None:
        from transformers import AutoTokenizer

        tokenizer_instance = AutoTokenizer.from_pretrained(
            tokenizer_config["pretrained"],
        )
        if tokenizer_instance.pad_token is None:
            tokenizer_instance.pad_token = tokenizer_instance.eos_token

    tokenize_fn = build_tokenizer(
        tokenizer_config, label_strategy=label_strategy, tokenizer=tokenizer_instance,
    )

    # ── collate_fn ──
    collate_fn = _select_collator(
        label_strategy, tokenizer_config, tokenizer=tokenizer_instance,
    )

    # ── DataLoader kwargs (drop_last를 분리하여 이중 전달 방지) ──
    dl_kwargs = dict(loader_config)
    dl_kwargs.pop("collate_fn", None)
    train_drop_last = dl_kwargs.pop("drop_last", True)

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
            **dl_kwargs,
        )

    # ── train dataset ──
    load_kwargs = dict(
        subset=subset, streaming=streaming, data_files=data_files, fmt=fmt,
    )
    train_ds = _load_source(source, split=split, **load_kwargs)
    train_ds = _rename_columns(train_ds, fields)
    train_ds = load_data(
        train_ds, transform=train_transform, tokenize_fn=tokenize_fn,
        streaming=streaming,
    )

    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds, shuffle=True)

    result: dict[str, DataLoader] = {
        "train": _make_loader(
            train_ds, shuffle=True, sampler=train_sampler, drop_last=train_drop_last,
        ),
    }

    # ── val dataset ──
    val_ds = None

    if val_split is None:
        # Explicit disable — skip val loading entirely
        logger.info("val_split=None: Validation이 비활성화되었습니다.")
    else:
        if val_split == "auto":
            # Current behavior: infer from train split
            split_str = split if isinstance(split, str) else "train"
            resolved_val_split = _infer_val_split(split_str)
        else:
            # User-specified split name
            resolved_val_split = val_split

        try:
            val_ds = _load_source(source, split=resolved_val_split, **load_kwargs)
            val_ds = _rename_columns(val_ds, fields)
            val_ds = load_data(
                val_ds, transform=val_transform, tokenize_fn=tokenize_fn,
                streaming=streaming,
            )
        except (ValueError, FileNotFoundError, KeyError):
            val_ds = None
            logger.warning(
                "Val split '%s'을 찾을 수 없습니다. Validation 없이 학습합니다. "
                "Early stopping과 best checkpoint 선택이 비활성화됩니다.",
                resolved_val_split,
            )

    if val_ds is not None:
        val_sampler = None
        if distributed:
            from torch.utils.data.distributed import DistributedSampler

            val_sampler = DistributedSampler(val_ds, shuffle=False)

        result["val"] = _make_loader(
            val_ds, shuffle=False, sampler=val_sampler, drop_last=False,
        )

    return result
