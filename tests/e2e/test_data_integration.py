"""데이터 파이프라인 통합 테스트: tokenizer → labels 생성 → collator 연쇄.

4 tests:
- test_seq2seq_missing_target_raises: target 컬럼 없으면 KeyError
- test_causal_label_strategy_produces_labels: causal → labels = input_ids 복사
- test_copy_label_strategy_preserves_labels: copy → 원본 labels 유지
- test_streaming_multimodal_raises: streaming + multimodal 조합 ValueError
"""

from __future__ import annotations

import pytest

from mdp.data.tokenizer import (
    LABEL_CAUSAL,
    LABEL_COPY,
    LABEL_SEQ2SEQ,
    build_tokenizer,
)


def test_seq2seq_missing_target_raises() -> None:
    """seq2seq label strategy에서 target 컬럼 없으면 KeyError."""
    config = {"pretrained": "gpt2", "max_length": 32}
    tokenize_fn = build_tokenizer(config, label_strategy=LABEL_SEQ2SEQ)
    assert tokenize_fn is not None

    examples = {"text": ["hello world", "test sentence"]}
    with pytest.raises(KeyError, match="target"):
        tokenize_fn(examples)


def test_causal_label_strategy_produces_labels() -> None:
    """causal strategy에서 labels가 input_ids의 복사본으로 생성되는지."""
    config = {"pretrained": "gpt2", "max_length": 32, "padding": "max_length"}
    tokenize_fn = build_tokenizer(config, label_strategy=LABEL_CAUSAL)

    examples = {"text": ["hello world"]}
    result = tokenize_fn(examples)

    assert "labels" in result
    assert "input_ids" in result
    # labels는 input_ids의 복사 (같은 값, 다른 객체)
    assert result["labels"] == result["input_ids"]
    assert result["labels"] is not result["input_ids"]


def test_copy_label_strategy_preserves_labels() -> None:
    """copy strategy에서 examples의 label을 그대로 유지하는지."""
    config = {"pretrained": "gpt2", "max_length": 32}
    tokenize_fn = build_tokenizer(config, label_strategy=LABEL_COPY)

    examples = {"text": ["hello world", "test"], "label": [0, 1]}
    result = tokenize_fn(examples)

    assert "labels" in result
    assert result["labels"] == [0, 1]


def test_streaming_multimodal_raises() -> None:
    """streaming=True + multimodal(vision+language) 조합은 ValueError를 발생시킨다."""
    from unittest.mock import MagicMock

    from mdp.data.loader import load_data

    fake_ds = MagicMock()
    dummy_transform = lambda x: x  # noqa: E731
    dummy_tokenize = lambda x: x  # noqa: E731

    with pytest.raises(ValueError, match="streaming=True.*multimodal"):
        load_data(
            fake_ds,
            transform=dummy_transform,
            tokenize_fn=dummy_tokenize,
            streaming=True,
        )
