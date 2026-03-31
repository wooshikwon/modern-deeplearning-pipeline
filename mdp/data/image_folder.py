"""ImageFolderDataset — 디렉토리 구조 기반 이미지 분류 데이터셋."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mdp.data.base import BaseDataset

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


class ImageFolderDataset(BaseDataset):
    """root 하위 디렉토리 이름을 클래스 레이블로 사용하는 이미지 데이터셋.

    디렉토리 구조::

        root/
          cat/
            001.jpg
            002.png
          dog/
            001.jpg

    Args:
        root: 이미지 루트 디렉토리 경로.
        transform: PIL Image → Tensor 변환 함수.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx: dict[str, int] = {}
        self.samples: list[tuple[Path, int]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """root 하위 디렉토리를 sorted 순서로 스캔하여 class_to_idx, samples를 구성."""
        class_dirs = sorted(
            d for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

        for class_dir in class_dirs:
            idx = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    self.samples.append((img_path, idx))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image  # lazy import

        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"pixel_values": image, "labels": label}

    def __len__(self) -> int:
        return len(self.samples)
