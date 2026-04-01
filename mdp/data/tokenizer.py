"""build_tokenizer — tokenizer YAML 설정을 토큰화 함수로 변환.

label 생성 전략은 fields(역할→컬럼 매핑)로부터 파생된다.
task 문자열이 아닌 label_strategy 상수로 분기한다.
"""

from __future__ import annotations

from typing import Any, Callable

# ── label strategy 상수 ──

LABEL_CAUSAL = "causal"  # text_generation, causal_lm, vision_language
LABEL_SEQ2SEQ = "seq2seq"  # seq2seq (text → target)
LABEL_COPY = "copy"  # text_classification (labels 그대로 통과)
LABEL_ALIGN = "align"  # token_classification (sub-word alignment)
LABEL_NONE = "none"  # feature_extraction, image-only 등


def derive_label_strategy(fields: dict[str, str] | None) -> str:
    """fields 딕셔너리로부터 label_strategy를 파생한다.

    Args:
        fields: ``{role: column_name}`` 매핑. ``None``이면 ``LABEL_NONE``.

    Returns:
        label strategy 상수 문자열.
    """
    if not fields:
        return LABEL_NONE

    roles = set(fields.keys())

    # target 역할이 있으면 seq2seq
    if "target" in roles:
        return LABEL_SEQ2SEQ

    # token_labels 역할이 있으면 alignment
    if "token_labels" in roles:
        return LABEL_ALIGN

    # text + label → classification (labels 복사)
    if "text" in roles and "label" in roles:
        return LABEL_COPY

    # text만 있으면 causal (text_generation)
    if "text" in roles:
        return LABEL_CAUSAL

    # image + text → causal (vision_language)
    if "image" in roles and "text" in roles:
        return LABEL_CAUSAL

    return LABEL_NONE


def build_tokenizer(
    config: dict[str, Any] | None,
    label_strategy: str = LABEL_CAUSAL,
) -> Callable | None:
    """tokenizer YAML 설정 → 토큰화 함수.

    YAML 예시::

        tokenizer:
          pretrained: "gpt2"
          max_length: 512
          padding: "max_length"
          truncation: true

    Args:
        config: tokenizer 설정 딕셔너리. ``None``이면 ``None`` 반환.
        label_strategy: label 생성 전략 상수.
            - ``"causal"``: labels = input_ids 복사.
            - ``"seq2seq"``: labels = tokenizer(text_target=...).
            - ``"copy"``: labels 그대로 통과.
            - ``"align"``: sub-word label alignment.
            - ``"none"``: label 생성 안 함.

    Returns:
        ``tokenize_fn(examples: dict) -> dict`` 또는 ``None``.
    """
    if config is None:
        return None

    from transformers import AutoTokenizer  # lazy import

    pretrained = config["pretrained"]
    max_length = config.get("max_length", 512)
    max_target_length = config.get("max_target_length", max_length)
    padding = config.get("padding", "max_length")
    truncation = config.get("truncation", True)
    chat_template = config.get("chat_template")
    is_split_into_words = config.get("is_split_into_words", False)

    # align(token_classification)은 word-level 입력이므로 강제 True
    if label_strategy == LABEL_ALIGN:
        is_split_into_words = True

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # pad_token이 없으면 eos_token으로 대체
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # chat_template 설정
    if chat_template:
        tokenizer.chat_template = chat_template

    def tokenize_fn(examples: dict[str, Any]) -> dict[str, Any]:
        # chat_template이 설정되어 있고 messages 키가 있으면 대화형 토큰화
        if chat_template and "messages" in examples:
            messages = examples["messages"]
            if isinstance(messages, list) and messages and isinstance(messages[0], list):
                # batched: list[list[dict]]
                texts = [
                    tokenizer.apply_chat_template(conv, tokenize=False)
                    for conv in messages
                ]
            else:
                # single: list[dict]
                texts = [tokenizer.apply_chat_template(messages, tokenize=False)]
        else:
            texts = examples.get("text", examples.get("input", []))

        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            is_split_into_words=is_split_into_words,
        )

        if label_strategy == LABEL_CAUSAL:
            encoded["labels"] = encoded["input_ids"].copy()

        elif label_strategy == LABEL_SEQ2SEQ:
            targets = examples.get("target", examples.get("summary", []))
            target_encoded = tokenizer(
                text_target=targets,
                max_length=max_target_length,
                padding=padding,
                truncation=truncation,
            )
            encoded["labels"] = target_encoded["input_ids"]

        elif label_strategy == LABEL_COPY:
            if "labels" in examples:
                encoded["labels"] = examples["labels"]
            elif "label" in examples:
                encoded["labels"] = examples["label"]

        elif label_strategy == LABEL_ALIGN:
            all_labels = examples.get("labels", examples.get("ner_tags", []))
            aligned_labels = []
            for i, label_seq in enumerate(all_labels):
                word_ids = encoded.word_ids(batch_index=i)
                prev_word_id = None
                label_ids = []
                for word_id in word_ids:
                    if word_id is None:
                        label_ids.append(-100)
                    elif word_id != prev_word_id:
                        label_ids.append(label_seq[word_id])
                    else:
                        # 서브워드 토큰: -100으로 마스킹
                        label_ids.append(-100)
                    prev_word_id = word_id
                aligned_labels.append(label_ids)
            encoded["labels"] = aligned_labels

        return encoded

    return tokenize_fn
