"""BaseDataset ABC — dict를 반환하는 Dataset 규약."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """dict를 반환하는 Dataset. 키 규약:

    - Vision: ``{"pixel_values": Tensor, "labels": Tensor}``
    - Language: ``{"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}``
    - Detection: ``{"pixel_values": Tensor, "bboxes": list[Tensor], "labels": Tensor}``
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]: ...

    @abstractmethod
    def __len__(self) -> int: ...
