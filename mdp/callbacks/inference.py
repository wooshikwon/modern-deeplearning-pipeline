"""DefaultOutputCallback -- 모델 출력을 후처리하여 파일로 저장하는 기본 출력 콜백.

기존 ``_postprocess`` + ``_save_results`` 의 동작을 콜백으로 캡슐화한다.
콜백이 없는 기존 사용법에서 자동 추가되어 하위 호환을 보장한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from mdp.callbacks.base import BaseInferenceCallback

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"parquet", "csv", "jsonl"}


def _to_numpy(t: torch.Tensor) -> Any:
    """Tensor를 numpy로 변환한다. numpy가 지원하지 않는 dtype(bf16 등)은 float32로 캐스팅."""
    import numpy as np  # noqa: F401 -- lazy import

    t = t.cpu()
    try:
        return t.numpy()
    except TypeError:
        return t.float().numpy()


def _postprocess(outputs: dict[str, torch.Tensor], task: str) -> dict[str, Any]:
    """출력 키 기반 모델 출력 후처리."""
    if "generated_ids" in outputs:
        return {"generated_ids": _to_numpy(outputs["generated_ids"])}

    if "boxes" in outputs:
        return {"boxes": _to_numpy(outputs["boxes"])}

    if "logits" in outputs:
        logits = outputs["logits"].float()
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        return {
            "prediction": _to_numpy(preds),
            "probabilities": _to_numpy(probs),
        }

    # fallback: 첫 번째 키
    first_key = next(iter(outputs))
    return {first_key: _to_numpy(outputs[first_key])}


def _save_results(
    records: list[dict[str, Any]],
    output_path: Path,
    output_format: str,
) -> Path:
    """결과를 parquet/csv/jsonl로 저장."""
    import pandas as pd  # lazy import

    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        path = output_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    elif output_format == "csv":
        path = output_path.with_suffix(".csv")
        df.to_csv(path, index=False)
    elif output_format == "jsonl":
        path = output_path.with_suffix(".jsonl")
        df.to_json(path, orient="records", lines=True)
    else:
        msg = f"Unsupported format: {output_format!r}. Use one of {_SUPPORTED_FORMATS}"
        raise ValueError(msg)

    logger.info("Saved %d records -> %s", len(df), path)
    return path


class DefaultOutputCallback(BaseInferenceCallback):
    """모델 출력을 후처리하여 파일로 저장하는 기본 출력 콜백.

    기존 ``_postprocess`` + ``_save_results`` 의 동작을 콜백으로 캡슐화한다.
    콜백이 없는 기존 사용법에서 자동 추가되어 하위 호환을 보장한다.

    Parameters
    ----------
    output_path:
        결과 파일 경로 (확장자는 output_format에 따라 자동 설정).
    output_format:
        ``parquet`` | ``csv`` | ``jsonl``.
    task:
        ``classification`` | ``detection`` | ``text_generation`` | 기타.
    """

    def __init__(
        self,
        output_path: str | Path,
        output_format: str = "parquet",
        task: str = "classification",
    ) -> None:
        if output_format not in _SUPPORTED_FORMATS:
            msg = f"Unsupported format: {output_format!r}. Use one of {_SUPPORTED_FORMATS}"
            raise ValueError(msg)

        self.output_path = Path(output_path)
        self.output_format = output_format
        self.task = task
        self._records: list[dict[str, Any]] = []
        self._result_path: Path | None = None

    def on_batch(self, batch_idx: int, batch: dict, outputs: dict, **kwargs) -> None:
        """배치별 후처리: _postprocess로 출력을 변환하고 records에 누적한다."""
        processed = _postprocess(outputs, self.task)

        # 배치 전체를 한 번에 레코드로 변환
        batch_size = next(iter(processed.values())).shape[0]
        for k in processed:
            processed[k] = processed[k].tolist()
        for i in range(batch_size):
            self._records.append({k: v[i] for k, v in processed.items()})

    def teardown(self, **kwargs) -> None:
        """누적된 records를 파일로 저장한다."""
        if self._records:
            self._result_path = _save_results(
                self._records, self.output_path, self.output_format,
            )

    @property
    def result_path(self) -> Path | None:
        """teardown 후 저장된 파일 경로. teardown 전이면 None."""
        return self._result_path
