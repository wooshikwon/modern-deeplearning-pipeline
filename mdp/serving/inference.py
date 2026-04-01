"""배치 추론 — 모델 포워드 + 태스크별 후처리 + 결과 저장."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"parquet", "csv", "jsonl"}


def _detect_device(device: str | torch.device | None) -> torch.device:
    """device 인자가 None이면 GPU/MPS/CPU 순으로 자동 감지."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _postprocess(outputs: dict[str, torch.Tensor], task: str) -> dict[str, Any]:
    """출력 키 기반 모델 출력 후처리.

    Parameters
    ----------
    outputs:
        모델이 반환한 텐서 딕셔너리.
    task:
        태스크명 (현재는 사용하지 않으나 시그니처 유지).

    Returns
    -------
    dict:
        후처리된 numpy 배열 딕셔너리.
    """
    if "generated_ids" in outputs:
        return {"generated_ids": outputs["generated_ids"].cpu().numpy()}

    if "boxes" in outputs:
        return {"boxes": outputs["boxes"].cpu().numpy()}

    if "logits" in outputs:
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        return {
            "prediction": preds.cpu().numpy(),
            "probabilities": probs.cpu().numpy(),
        }

    # fallback: 첫 번째 키
    first_key = next(iter(outputs))
    return {first_key: outputs[first_key].cpu().numpy()}


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

    logger.info("Saved %d records → %s", len(df), path)
    return path


def run_batch_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_path: str | Path,
    output_format: str = "parquet",
    task: str = "classification",
    device: str | torch.device | None = None,
) -> Path:
    """배치 추론을 실행하고 결과를 파일로 저장한다.

    Parameters
    ----------
    model:
        추론에 사용할 PyTorch 모델. ``dict`` 를 반환해야 한다.
    dataloader:
        입력 데이터를 제공하는 DataLoader.
    output_path:
        결과 파일 경로 (확장자는 format에 따라 자동 설정).
    output_format:
        ``parquet`` | ``csv`` | ``jsonl``.
    task:
        ``classification`` | ``detection`` | ``text_generation`` | 기타.
    device:
        추론 디바이스. ``None`` 이면 자동 감지.

    Returns
    -------
    Path:
        저장된 결과 파일 경로.
    """
    if output_format not in _SUPPORTED_FORMATS:
        msg = f"Unsupported format: {output_format!r}. Use one of {_SUPPORTED_FORMATS}"
        raise ValueError(msg)

    dev = _detect_device(device)
    logger.info("Running batch inference on %s (task=%s)", dev, task)

    model = model.to(dev)
    model.eval()

    all_records: list[dict[str, Any]] = []

    use_generate = hasattr(model, "generate")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # batch를 device로 이동
            if isinstance(batch, torch.Tensor):
                batch = batch.to(dev)
            elif isinstance(batch, dict):
                batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [v.to(dev) if isinstance(v, torch.Tensor) else v for v in batch]

            # generate() 메서드가 있으면 생성, 없으면 forward
            if use_generate and isinstance(batch, dict):
                generated_ids = model.generate(**batch)
                outputs = {"generated_ids": generated_ids}
            elif isinstance(batch, torch.Tensor):
                outputs = model(batch)
            elif isinstance(batch, dict):
                outputs = model(**batch)
            elif isinstance(batch, (list, tuple)):
                outputs = model(*batch)
            else:
                outputs = model(batch)

            # 모델이 텐서 하나만 반환하면 dict로 감싼다
            if isinstance(outputs, torch.Tensor):
                outputs = {"logits": outputs}

            processed = _postprocess(outputs, task)

            # 배치 내 각 샘플을 레코드로 분리
            batch_size = next(iter(processed.values())).shape[0]
            for i in range(batch_size):
                record = {k: v[i].tolist() for k, v in processed.items()}
                all_records.append(record)

            if (batch_idx + 1) % 50 == 0:
                logger.info("  processed %d batches", batch_idx + 1)

    logger.info("Inference complete: %d samples", len(all_records))
    return _save_results(all_records, Path(output_path), output_format)
