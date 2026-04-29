"""내장 Dataset 클래스 — _component_ 패턴으로 주입 가능한 Dataset 래퍼."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HuggingFaceDataset:
    """HuggingFace datasets 기반 Dataset.

    _component_ 패턴으로 Recipe에서 주입된다::

        dataset:
          _component_: HuggingFaceDataset
          source: wikitext
          subset: wikitext-103-v1
          split: train
          fields: {text: text}
          tokenizer: gpt2
          max_length: 1024

    내부적으로 ``datasets.load_dataset()``를 호출하고,
    fields 리네이밍 + 토큰화 + torch format 변환을 수행한다.
    """

    def __init__(
        self,
        source: str,
        split: str = "train",
        subset: str | None = None,
        fields: dict[str, str] | None = None,
        tokenizer: str | None = None,
        max_length: int = 2048,
        padding: bool | str = False,
        truncation: bool = True,
        streaming: bool = False,
        format: str = "auto",
        data_files: str | dict[str, str] | None = None,
    ) -> None:
        from datasets import load_dataset

        # ── source 로딩 ──
        path = Path(source)
        if path.exists():
            if path.is_dir():
                resolved_fmt = format if format != "auto" else "imagefolder"
                ds = load_dataset(resolved_fmt, data_dir=source, split=split, streaming=streaming)
            else:
                resolved_fmt = self._detect_format(source, format)
                ds = load_dataset(resolved_fmt, data_files=data_files or source, split=split, streaming=streaming)
        else:
            kwargs: dict[str, Any] = {}
            if subset is not None:
                kwargs["name"] = subset
            if data_files is not None:
                kwargs["data_files"] = data_files
            ds = load_dataset(source, split=split, streaming=streaming, **kwargs)

        # ── fields 리네이밍 ──
        if fields:
            rename_map = {col: role for role, col in fields.items() if col != role}
            if rename_map and hasattr(ds, "rename_columns"):
                ds = ds.rename_columns(rename_map)

        # ── 토큰화 (language) ──
        if tokenizer is not None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            raw_columns = list(fields.values()) if fields else None

            def _tokenize_fn(examples):
                text_key = "text"
                if text_key not in examples:
                    for k in examples:
                        if isinstance(examples[k], list) and len(examples[k]) > 0 and isinstance(examples[k][0], str):
                            text_key = k
                            break
                texts = examples.get(text_key, [])
                enc = tok(
                    texts,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                )
                # causal LM: labels = input_ids
                enc["labels"] = [list(ids) for ids in enc["input_ids"]]
                return enc

            remove_cols = []
            if raw_columns and hasattr(ds, "column_names"):
                remove_cols = [c for c in (ds.column_names or []) if c in (set(raw_columns) | set(fields.keys()) if fields else set())]
            ds = ds.map(_tokenize_fn, batched=True, remove_columns=remove_cols or None)

            if not streaming:
                ds.set_format("torch")

        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._ds[idx]

    def __getlength__(self, idx: int) -> int:
        """i번째 sample의 토큰 길이를 반환한다 (Lengthed protocol).

        tokenizer가 적용되어 ``input_ids``가 존재할 때만 의미가 있다. 미적용
        dataset에 대해 호출하면 dict-style 접근에서 ``KeyError``가 발생한다 —
        이것이 의도된 동작으로, length-bucketed sampler가 부적절한 dataset에
        대해 명시적 실패를 내도록 한다.

        streaming dataset의 경우 ``self._ds[idx]`` 자체가 random access를 지원하지
        않아 자연스럽게 실패한다 (length-bucketed + streaming 조합은 spec 비범위).
        """
        sample = self._ds[idx]
        return len(sample["input_ids"])

    @staticmethod
    def _detect_format(source: str, fmt: str) -> str:
        if fmt != "auto":
            return fmt
        ext = Path(source).suffix.lower()
        return {
            ".csv": "csv", ".tsv": "csv",
            ".json": "json", ".jsonl": "json",
            ".parquet": "parquet",
        }.get(ext, "json")


class ImageClassificationDataset:
    """이미지 분류 Dataset.

    _component_ 패턴으로 Recipe에서 주입된다::

        dataset:
          _component_: ImageClassificationDataset
          source: cifar10
          split: train
          fields: {image: image, label: label}
          augmentation:
            - type: RandomResizedCrop
              params: {size: 224}
            - type: RandomHorizontalFlip

    내부적으로 ``datasets.load_dataset()`` + ``torchvision.transforms`` Compose를 적용한다.
    """

    def __init__(
        self,
        source: str,
        split: str = "train",
        fields: dict[str, str] | None = None,
        augmentation: list[dict[str, Any]] | None = None,
        subset: str | None = None,
        streaming: bool = False,
    ) -> None:
        from datasets import load_dataset

        path = Path(source)
        if path.exists() and path.is_dir():
            ds = load_dataset("imagefolder", data_dir=source, split=split, streaming=streaming)
        else:
            kwargs: dict[str, Any] = {}
            if subset is not None:
                kwargs["name"] = subset
            ds = load_dataset(source, split=split, streaming=streaming, **kwargs)

        # fields 리네이밍
        if fields:
            rename_map = {col: role for role, col in fields.items() if col != role}
            if rename_map and hasattr(ds, "rename_columns"):
                ds = ds.rename_columns(rename_map)

        # augmentation (torchvision transforms)
        transform = None
        if augmentation:
            transform = self._build_transform(augmentation)

        if transform is not None and not streaming:
            ds.set_transform(
                lambda examples: {
                    k: ([transform(img) for img in v] if k in ("image", "pixel_values") else v)
                    for k, v in examples.items()
                }
            )

        self._ds = ds

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._ds[idx]

    @staticmethod
    def _build_transform(steps: list[dict[str, Any]]) -> Any:
        """augmentation steps DSL을 torchvision.transforms.v2.Compose로 변환."""
        from mdp.data.transforms import build_transforms

        return build_transforms({"steps": steps})
