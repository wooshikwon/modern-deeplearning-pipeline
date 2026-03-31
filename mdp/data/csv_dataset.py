"""CSVDataset — CSV 파일 기반 범용 데이터셋."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mdp.data.base import BaseDataset


class CSVDataset(BaseDataset):
    """CSV 파일에서 이미지/텍스트/레이블을 읽는 범용 데이터셋.

    Args:
        csv_path: CSV 파일 경로.
        columns: MDP 키 → CSV 컬럼명 매핑.
            예: ``{"image": "filepath", "labels": "category"}``
        transform: Vision 데이터셋 이미지 변환 함수.
        tokenize_fn: Language 데이터셋 전처리 함수.
        root_dir: 이미지 파일 경로의 베이스 디렉토리.
    """

    def __init__(
        self,
        csv_path: str | Path,
        columns: dict[str, str] | None = None,
        transform: Callable | None = None,
        tokenize_fn: Callable | None = None,
        root_dir: str | Path | None = None,
    ) -> None:
        import pandas as pd  # lazy import

        self.df = pd.read_csv(csv_path)
        self.columns = columns or {}
        self.transform = transform
        self.tokenize_fn = tokenize_fn
        self.root_dir = Path(root_dir) if root_dir is not None else None

        # 문자열 레이블 → 정수 인코딩
        self.label_encoder: dict[str, int] | None = None
        label_col = self.columns.get("labels", "label")
        if label_col in self.df.columns:
            raw_labels = self.df[label_col]
            if raw_labels.dtype == object:
                unique_sorted = sorted(raw_labels.unique())
                self.label_encoder = {
                    name: idx for idx, name in enumerate(unique_sorted)
                }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        # Vision 분기: columns에 "image" 키가 있으면
        image_col = self.columns.get("image")
        if image_col is not None:
            from PIL import Image  # lazy import

            img_path = Path(row[image_col])
            if self.root_dir is not None:
                img_path = self.root_dir / img_path
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

            label_col = self.columns.get("labels", "label")
            label = row[label_col]
            if self.label_encoder is not None:
                label = self.label_encoder[label]

            return {"pixel_values": image, "labels": label}

        # Text 분기: columns에 "text" 키가 있으면
        text_col = self.columns.get("text")
        if text_col is not None and self.tokenize_fn is not None:
            text = row[text_col]
            tokenized = self.tokenize_fn({"text": [text]})
            result = {k: v[0] for k, v in tokenized.items()}

            label_col = self.columns.get("labels", "label")
            if label_col in row.index:
                label = row[label_col]
                if self.label_encoder is not None:
                    label = self.label_encoder[label]
                result["labels"] = label
            return result

        # 기본: 모든 컬럼 매핑 그대로 반환
        result: dict[str, Any] = {}
        for mdp_key, csv_col in self.columns.items():
            result[mdp_key] = row[csv_col]

        label_col = self.columns.get("labels", "label")
        if label_col in row.index and "labels" in result:
            if self.label_encoder is not None:
                result["labels"] = self.label_encoder[result["labels"]]

        return result

    def __len__(self) -> int:
        return len(self.df)
