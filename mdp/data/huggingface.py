"""HuggingFaceDataset — Hugging Face datasets 라이브러리 래퍼."""

from __future__ import annotations

from typing import Any, Callable

from mdp.data.base import BaseDataset


class HuggingFaceDataset(BaseDataset):
    """Hugging Face Hub 데이터셋을 MDP 키 규약으로 변환하는 래퍼.

    Args:
        name: 데이터셋 이름 (Hub ID 또는 로컬 경로).
        split: 분할 이름 (``"train"``, ``"test"`` 등).
        subset: 데이터셋 서브셋 (config) 이름.
        columns: MDP 키 → 실제 컬럼명 매핑. 예: ``{"image": "img", "labels": "label"}``.
        tokenize_fn: Language 데이터셋 전처리 함수.
        transform: Vision 데이터셋 이미지 변환 함수.
        streaming: ``True``이면 IterableDataset으로 로드.
    """

    def __init__(
        self,
        name: str,
        split: str,
        subset: str | None = None,
        columns: dict[str, str] | None = None,
        tokenize_fn: Callable | None = None,
        transform: Callable | None = None,
        streaming: bool = False,
    ) -> None:
        from datasets import load_dataset  # lazy import

        self.columns = columns or {}
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self._is_vision = any(
            k in self.columns for k in ("image", "mask")
        )

        self._ds = load_dataset(
            name,
            subset,
            split=split,
            streaming=streaming,
        )

        # Language 최적화: non-streaming + non-vision → 사전 토큰화 + torch format
        if not streaming and not self._is_vision and tokenize_fn is not None:
            self._ds = self._ds.map(tokenize_fn, batched=True)
            self._ds.set_format("torch")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._ds[idx]

        if self._is_vision:
            image_col = self.columns.get("image", "image")
            label_col = self.columns.get("labels", "label")

            image = item[image_col]
            # PIL Image가 아닌 경우 변환
            if not hasattr(image, "convert"):
                from PIL import Image  # lazy import

                image = Image.open(image).convert("RGB")

            result: dict[str, Any] = {}

            if "mask" in self.columns:
                # Segmentation 경로: mask를 tv_tensors로 래핑하여 공간 변환 동기화
                from torchvision import tv_tensors  # lazy import

                mask = item[self.columns["mask"]]
                mask = tv_tensors.Mask(mask)
                image = tv_tensors.Image(image)
                if self.transform is not None:
                    image, mask = self.transform(image, mask)
                result["pixel_values"] = image
                result["labels"] = mask
            else:
                # Classification 경로
                if self.transform is not None:
                    image = self.transform(image)
                result["pixel_values"] = image
                result["labels"] = item[label_col]

            # Multimodal 분기: vision + text가 모두 있으면 토큰화 결과 병합
            if self.tokenize_fn and "text" in self.columns:
                text = item[self.columns["text"]]
                tokenized = self.tokenize_fn(text)
                result.update(tokenized)

            return result

        # Language: 이미 토큰화 완료 → dict 그대로 반환
        return dict(item)

    def __len__(self) -> int:
        return len(self._ds)
