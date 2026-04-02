"""Preference collator 통합 테스트.

4 tests:
- test_preference_label_strategy: chosen+rejected → LABEL_PREFERENCE
- test_preference_collator_output_structure: collator가 올바른 dict 키를 반환
- test_preference_collator_prompt_masking: prompt 부분 labels = -100
- test_preference_collator_independent_padding: chosen/rejected 독립 padding
"""

from __future__ import annotations

import torch

from mdp.data.collators import PreferenceCollator
from mdp.data.tokenizer import LABEL_PREFERENCE, derive_label_strategy


def test_preference_label_strategy() -> None:
    """chosen + rejected 역할이 있으면 LABEL_PREFERENCE."""
    assert derive_label_strategy({"chosen": "c", "rejected": "r"}) == LABEL_PREFERENCE
    assert derive_label_strategy({"prompt": "p", "chosen": "c", "rejected": "r"}) == LABEL_PREFERENCE
    # chosen만 있으면 preference가 아님
    assert derive_label_strategy({"chosen": "c"}) != LABEL_PREFERENCE


def test_preference_collator_output_structure() -> None:
    """PreferenceCollator가 6개 키를 가진 dict를 반환하는지."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    collator = PreferenceCollator(tokenizer=tokenizer, max_length=128)

    batch = [
        {"prompt": "Question: ", "chosen": "Good answer", "rejected": "Bad answer"},
        {"prompt": "Question: ", "chosen": "Another good", "rejected": "Another bad"},
    ]
    result = collator(batch)

    expected_keys = {
        "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
        "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
    }
    assert set(result.keys()) == expected_keys
    assert result["chosen_input_ids"].shape[0] == 2  # batch size
    assert result["rejected_input_ids"].shape[0] == 2


def test_preference_collator_prompt_masking() -> None:
    """prompt 부분의 labels가 -100으로 마스킹되는지."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    collator = PreferenceCollator(tokenizer=tokenizer, max_length=128)

    batch = [
        {"prompt": "This is the prompt. ", "chosen": "Good response.", "rejected": "Bad response."},
    ]
    result = collator(batch)

    chosen_labels = result["chosen_labels"][0]
    # 앞부분(prompt)이 -100이고, 뒷부분(response)은 -100이 아닌 토큰이 존재
    assert (chosen_labels[:3] == -100).all(), "Prompt tokens should be masked"
    non_masked = chosen_labels[chosen_labels != -100]
    assert len(non_masked) > 0, "Response tokens should not all be masked"


def test_preference_collator_independent_padding() -> None:
    """chosen과 rejected가 독립적으로 padding되는지."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    collator = PreferenceCollator(tokenizer=tokenizer, max_length=128)

    batch = [
        {"chosen": "Short.", "rejected": "This is a much longer rejected response with many tokens."},
        {"chosen": "Also short.", "rejected": "Another long rejected text here."},
    ]
    result = collator(batch)

    # chosen과 rejected의 seq_len이 다를 수 있음
    chosen_len = result["chosen_input_ids"].shape[1]
    rejected_len = result["rejected_input_ids"].shape[1]
    # rejected가 더 긴 텍스트이므로 rejected_len >= chosen_len
    assert rejected_len >= chosen_len
