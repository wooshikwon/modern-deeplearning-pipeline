"""load_data — 데이터셋 로드 + transform/tokenization 적용."""

from __future__ import annotations

from typing import Any, Callable


def _apply_vision_transform(ds: Any, transform: Callable) -> Any:
    """Vision transform을 데이터셋에 적용한다."""
    ds.set_transform(
        lambda examples: {
            k: (transform(v) if k in ("image", "pixel_values") else v)
            for k, v in examples.items()
        }
    )
    return ds


def _apply_language_tokenization(
    ds: Any, tokenize_fn: Callable, streaming: bool,
) -> Any:
    """Language tokenization을 데이터셋에 적용한다."""
    ds = ds.map(tokenize_fn, batched=True)
    remove_cols = _columns_to_remove(ds)
    if remove_cols:
        ds = ds.remove_columns(remove_cols)
    if not streaming:
        ds.set_format("torch")
    return ds


def _columns_to_remove(ds: Any) -> list[str]:
    """토큰화 후 불필요한 원본 컬럼을 식별한다."""
    keep = {
        "input_ids", "attention_mask", "token_type_ids", "labels",
        "pixel_values", "image",
    }
    if hasattr(ds, "column_names"):
        return [c for c in ds.column_names if c not in keep]
    return []


def load_data(
    ds: Any,
    *,
    transform: Callable | None = None,
    tokenize_fn: Callable | None = None,
    streaming: bool = False,
) -> Any:
    """데이터셋에 transform과 tokenization을 적용한다.

    분기는 transform/tokenize_fn 존재 여부로 결정된다:
    - vision + language → multimodal
    - vision only → vision transform
    - language only → language tokenization

    Args:
        ds: 원본 데이터셋.
        transform: Vision transform 함수.
        tokenize_fn: Language tokenization 함수.
        streaming: 스트리밍 모드 여부.

    Returns:
        전처리가 적용된 데이터셋.
    """
    has_vision = transform is not None
    has_language = tokenize_fn is not None

    if has_vision and has_language:
        if streaming:
            raise ValueError(
                "streaming=True와 multimodal(vision+language) 조합은 지원되지 않습니다. "
                "set_transform이 IterableDataset에서 사용 불가합니다."
            )
        # Language tokenization (map + remove_columns, set_format 제외)
        ds = ds.map(tokenize_fn, batched=True)
        if not streaming:
            remove_cols = _columns_to_remove(ds)
            if remove_cols:
                ds = ds.remove_columns(remove_cols)
        # Combined set_transform: vision transform + torch tensor 변환
        # set_transform은 set_format을 덮어쓰므로 한 번에 처리해야 한다
        import torch as _torch

        def _multimodal_transform(examples: dict) -> dict:
            result = {}
            for k, v in examples.items():
                if k in ("image", "pixel_values"):
                    result[k] = transform(v)
                elif isinstance(v, list):
                    try:
                        result[k] = _torch.tensor(v)
                    except (ValueError, TypeError):
                        result[k] = v
                else:
                    result[k] = v
            return result

        ds.set_transform(_multimodal_transform)
    elif has_vision:
        ds = _apply_vision_transform(ds, transform)
    elif has_language:
        ds = _apply_language_tokenization(ds, tokenize_fn, streaming)

    return ds
