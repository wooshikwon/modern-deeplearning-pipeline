"""Synthetic data generators for E2E tests.

Each function returns a list of batch dicts containing torch tensors.
All data is random and deterministic (seeded) for reproducibility.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class _SimpleDataset:
    """Minimal dataset with a fixed length for DataLoader compatibility."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


class ListDataLoader:
    """Wraps a list of batch dicts as a DataLoader-like iterable.

    Provides __iter__, __len__, and a .dataset attribute with __len__.
    """

    def __init__(self, batches: list[dict[str, Tensor]]) -> None:
        self._batches = batches
        # Total sample count = sum of batch sizes (use first tensor in each batch)
        total = 0
        for b in batches:
            first_tensor = next(iter(b.values()))
            total += first_tensor.size(0)
        self.dataset = _SimpleDataset(total)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def make_vision_batches(
    num_batches: int = 5,
    batch_size: int = 4,
    num_classes: int = 2,
    image_size: int = 8,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Generate synthetic image classification batches.

    Returns list of dicts with 'pixel_values' (B,3,H,W) and 'labels' (B,).
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        batches.append(
            {
                "pixel_values": torch.randn(
                    batch_size, 3, image_size, image_size, generator=g
                ),
                "labels": torch.randint(
                    0, num_classes, (batch_size,), generator=g
                ),
            }
        )
    return batches


def make_feature_map_batches(
    num_batches: int = 5,
    batch_size: int = 4,
    image_size: int = 8,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Generate synthetic feature map batches (no labels needed).

    Returns list of dicts with 'pixel_values' (B,3,H,W).
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        batches.append(
            {
                "pixel_values": torch.randn(
                    batch_size, 3, image_size, image_size, generator=g
                ),
            }
        )
    return batches


def make_language_batches(
    num_batches: int = 5,
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Generate synthetic language model batches.

    Returns list of dicts with 'input_ids' (B, L).
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        batches.append(
            {
                "input_ids": torch.randint(
                    0, vocab_size, (batch_size, seq_len), generator=g
                ),
            }
        )
    return batches


def make_token_class_batches(
    num_batches: int = 5,
    batch_size: int = 4,
    seq_len: int = 16,
    vocab_size: int = 128,
    num_classes: int = 5,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Generate synthetic token classification batches.

    Returns list of dicts with 'input_ids' (B, L) and 'labels' (B, L).
    Some label positions are set to -100 (ignore).
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        labels = torch.randint(
            0, num_classes, (batch_size, seq_len), generator=g
        )
        # Mask ~20% of positions with -100
        mask = torch.rand(batch_size, seq_len, generator=g) < 0.2
        labels[mask] = -100

        batches.append(
            {
                "input_ids": torch.randint(
                    0, vocab_size, (batch_size, seq_len), generator=g
                ),
                "labels": labels,
            }
        )
    return batches


def make_multimodal_batches(
    num_batches: int = 5,
    batch_size: int = 4,
    image_size: int = 8,
    seq_len: int = 16,
    vocab_size: int = 128,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Generate synthetic multimodal batches (image + text).

    Returns list of dicts with 'pixel_values' (B,3,H,W) and 'input_ids' (B,L).
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        batches.append(
            {
                "pixel_values": torch.randn(
                    batch_size, 3, image_size, image_size, generator=g
                ),
                "input_ids": torch.randint(
                    0, vocab_size, (batch_size, seq_len), generator=g
                ),
            }
        )
    return batches


def make_segmentation_batches(
    num_batches: int = 5,
    batch_size: int = 4,
    num_classes: int = 3,
    image_size: int = 8,
    seed: int = 42,
) -> list[dict[str, Tensor]]:
    """Generate synthetic semantic segmentation batches.

    Returns list of dicts with 'pixel_values' (B,3,H,W) and 'labels' (B,H,W).
    """
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        batches.append(
            {
                "pixel_values": torch.randn(
                    batch_size, 3, image_size, image_size, generator=g
                ),
                "labels": torch.randint(
                    0, num_classes, (batch_size, image_size, image_size), generator=g
                ),
            }
        )
    return batches
