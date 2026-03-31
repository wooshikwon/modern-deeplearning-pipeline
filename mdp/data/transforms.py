"""build_transforms — augmentation YAML 설정을 torchvision 파이프라인으로 변환."""

from __future__ import annotations

from typing import Any


def build_transforms(config: dict[str, Any] | None) -> Any:
    """augmentation YAML 설정 → ``torchvision.transforms.v2.Compose``.

    YAML 예시::

        augmentation:
          train:
            steps:
              - type: RandomResizedCrop
                params: {size: [224, 224]}
              - type: RandomHorizontalFlip
                params: {p: 0.5}
              - type: ToDtype
                params: {dtype: float32, scale: true}

    Args:
        config: ``steps`` 키를 가진 딕셔너리. ``None``이면 ``None`` 반환.

    Returns:
        ``torchvision.transforms.v2.Compose`` 또는 ``None``.
    """
    if config is None:
        return None

    steps = config.get("steps")
    if not steps:
        return None

    import torch  # lazy import
    from torchvision.transforms import v2  # lazy import

    _DTYPE_MAP: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    transforms_list = []
    for step in steps:
        transform_type = step["type"]
        params = dict(step.get("params", {}))

        # dtype 문자열 → torch.dtype 변환 특수 처리
        if "dtype" in params and isinstance(params["dtype"], str):
            dtype_str = params["dtype"]
            if dtype_str in _DTYPE_MAP:
                params["dtype"] = _DTYPE_MAP[dtype_str]
            else:
                raise ValueError(
                    f"알 수 없는 dtype: '{dtype_str}'. "
                    f"지원: {list(_DTYPE_MAP.keys())}"
                )

        transform_cls = getattr(v2, transform_type, None)
        if transform_cls is None:
            raise AttributeError(
                f"torchvision.transforms.v2에 '{transform_type}'이(가) 없습니다"
            )

        transforms_list.append(transform_cls(**params))

    return v2.Compose(transforms_list)
