"""load_data — 데이터셋 로드 + transform/tokenization 적용."""

from __future__ import annotations

from typing import Any, Callable


def _apply_vision_transform(ds: Any, transform: Callable) -> Any:
    """Vision transform을 데이터셋에 적용한다."""
    ds.set_transform(
        lambda examples: {
            k: ([transform(img) for img in v] if k in ("image", "pixel_values") else v)
            for k, v in examples.items()
        }
    )
    return ds


def _apply_language_tokenization(
    ds: Any, tokenize_fn: Callable, streaming: bool,
    raw_columns: list[str] | None = None,
) -> Any:
    """Language tokenization을 데이터셋에 적용한다."""
    if raw_columns:
        ds = ds.map(tokenize_fn, batched=True, remove_columns=raw_columns)
    else:
        ds = ds.map(tokenize_fn, batched=True)
    if not streaming:
        ds.set_format("torch")
    return ds


def load_data(
    ds: Any,
    *,
    transform: Callable | None = None,
    tokenize_fn: Callable | None = None,
    streaming: bool = False,
    raw_columns: list[str] | None = None,
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
        raw_columns: 토큰화 후 제거할 원본 컬럼명 리스트.
            Recipe의 data.fields values에서 파생한다.

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
        if raw_columns:
            # image 컬럼은 set_transform에서 사용하므로 제거 대상에서 제외
            text_cols = [c for c in raw_columns if c not in ("image", "pixel_values")]
            ds = ds.map(tokenize_fn, batched=True, remove_columns=text_cols)
        else:
            ds = ds.map(tokenize_fn, batched=True)
        # Combined set_transform: vision transform + torch tensor 변환
        # set_transform은 set_format을 덮어쓰므로 한 번에 처리해야 한다
        import torch as _torch

        def _multimodal_transform(examples: dict) -> dict:
            result = {}
            for k, v in examples.items():
                if k in ("image", "pixel_values"):
                    result[k] = [transform(img) for img in v]
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
        ds = _apply_language_tokenization(ds, tokenize_fn, streaming, raw_columns)

    return ds
