"""Collator 래퍼 — _component_ 패턴으로 주입 가능한 Collator 모음.

각 Collator는 ``__init__(tokenizer, max_length, ...)`` + ``__call__(batch) -> dict`` 인터페이스를 따른다.
내부적으로 HuggingFace의 DataCollator를 래핑하거나, 커스텀 collation 로직을 구현한다.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── CausalLM ──


class CausalLMCollator:
    """Causal Language Modeling용 collator.

    ``DataCollatorForLanguageModeling(mlm=False)``를 래핑한다.
    """

    def __init__(self, tokenizer: str, max_length: int = 2048, **kwargs: Any) -> None:
        from transformers import AutoTokenizer, DataCollatorForLanguageModeling

        tok = AutoTokenizer.from_pretrained(tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self._inner = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return self._inner(features)


# ── Seq2Seq ──


class Seq2SeqCollator:
    """Seq2Seq(encoder-decoder) 모델용 collator."""

    def __init__(self, tokenizer: str, max_length: int = 2048, **kwargs: Any) -> None:
        from transformers import AutoTokenizer, DataCollatorForSeq2Seq

        tok = AutoTokenizer.from_pretrained(tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self._inner = DataCollatorForSeq2Seq(tokenizer=tok)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return self._inner(features)


# ── Classification (text) ──


class ClassificationCollator:
    """텍스트 분류용 collator — DataCollatorWithPadding 래핑."""

    def __init__(self, tokenizer: str, **kwargs: Any) -> None:
        from transformers import AutoTokenizer, DataCollatorWithPadding

        tok = AutoTokenizer.from_pretrained(tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self._inner = DataCollatorWithPadding(tokenizer=tok)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return self._inner(features)


# ── Token Classification ──


class TokenClassificationCollator:
    """토큰 분류용 collator — DataCollatorForTokenClassification 래핑."""

    def __init__(self, tokenizer: str, **kwargs: Any) -> None:
        from transformers import AutoTokenizer, DataCollatorForTokenClassification

        tok = AutoTokenizer.from_pretrained(tokenizer)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok
        self._inner = DataCollatorForTokenClassification(tokenizer=tok)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return self._inner(features)


# ── Vision ──


class VisionCollator:
    """Vision(이미지) 데이터용 collator — torch default_collate 사용."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        from torch.utils.data.dataloader import default_collate
        return default_collate(features)


# ── Preference (기존) ──


class PreferenceCollator:
    """Pairwise preference 데이터용 collator.

    각 샘플의 chosen/rejected 텍스트를 독립적으로 tokenize + padding한다.
    prompt가 있으면 prompt 부분의 labels를 -100으로 마스킹한다.

    입력 (배치 내 각 샘플):
        {"prompt": str, "chosen": str, "rejected": str}
        prompt는 생략 가능.

    출력:
        {
            "chosen_input_ids": Tensor[batch, max_len_chosen],
            "chosen_attention_mask": Tensor[batch, max_len_chosen],
            "chosen_labels": Tensor[batch, max_len_chosen],
            "rejected_input_ids": Tensor[batch, max_len_rejected],
            "rejected_attention_mask": Tensor[batch, max_len_rejected],
            "rejected_labels": Tensor[batch, max_len_rejected],
        }
    """

    def __init__(self, tokenizer: str | Any, max_length: int = 2048, **kwargs: Any) -> None:
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        chosen_batch = []
        rejected_batch = []

        for sample in features:
            prompt = sample.get("prompt", "")
            chosen_text = prompt + sample["chosen"] if prompt else sample["chosen"]
            rejected_text = prompt + sample["rejected"] if prompt else sample["rejected"]

            # prompt 길이 측정 (loss masking용)
            if prompt:
                full_prefix = self.tokenizer(
                    prompt, add_special_tokens=True,
                )["input_ids"]
                # Use length with special tokens — this matches what the full
                # text tokenization will produce for the prompt portion
                prompt_len = len(full_prefix)
            else:
                prompt_len = 0

            # chosen tokenize
            chosen_enc = self.tokenizer(
                chosen_text,
                max_length=self.max_length,
                truncation=True,
            )
            chosen_labels = list(chosen_enc["input_ids"])
            for i in range(min(prompt_len, len(chosen_labels))):
                chosen_labels[i] = -100

            chosen_batch.append({
                "input_ids": chosen_enc["input_ids"],
                "attention_mask": chosen_enc["attention_mask"],
                "labels": chosen_labels,
            })

            # rejected tokenize
            rejected_enc = self.tokenizer(
                rejected_text,
                max_length=self.max_length,
                truncation=True,
            )
            rejected_labels = list(rejected_enc["input_ids"])
            for i in range(min(prompt_len, len(rejected_labels))):
                rejected_labels[i] = -100

            rejected_batch.append({
                "input_ids": rejected_enc["input_ids"],
                "attention_mask": rejected_enc["attention_mask"],
                "labels": rejected_labels,
            })

        # padding (chosen과 rejected 독립)
        # labels를 분리하여 tokenizer.pad에 넘기지 않음 (nest 문제 방지)
        chosen_labels_list = [b.pop("labels") for b in chosen_batch]
        rejected_labels_list = [b.pop("labels") for b in rejected_batch]

        chosen_padded = self.tokenizer.pad(chosen_batch, padding=True, return_tensors="pt")
        rejected_padded = self.tokenizer.pad(rejected_batch, padding=True, return_tensors="pt")

        # labels를 직접 padding (-100으로 채움)
        chosen_max = chosen_padded["input_ids"].shape[1]
        rejected_max = rejected_padded["input_ids"].shape[1]

        chosen_labels = torch.full((len(chosen_labels_list), chosen_max), -100, dtype=torch.long)
        for i, labels in enumerate(chosen_labels_list):
            chosen_labels[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)

        rejected_labels = torch.full((len(rejected_labels_list), rejected_max), -100, dtype=torch.long)
        for i, labels in enumerate(rejected_labels_list):
            rejected_labels[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)

        return {
            "chosen_input_ids": chosen_padded["input_ids"],
            "chosen_attention_mask": chosen_padded["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_padded["input_ids"],
            "rejected_attention_mask": rejected_padded["attention_mask"],
            "rejected_labels": rejected_labels,
        }
