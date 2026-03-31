"""ONNX 변환 및 추론 — torch.onnx.export + onnxruntime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    output_path: str | Path,
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> Path:
    """PyTorch 모델을 ONNX 형식으로 변환한다.

    Parameters
    ----------
    model:
        변환할 PyTorch 모델.
    dummy_input:
        모델 트레이싱에 사용할 더미 입력 텐서.
    output_path:
        ONNX 파일 저장 경로.
    opset_version:
        ONNX opset 버전. 기본 17.
    dynamic_axes:
        동적 축 설정. ``None`` 이면 batch 축(dim 0)만 동적으로 설정.

    Returns
    -------
    Path:
        저장된 ONNX 파일 경로.
    """
    output_path = Path(output_path).with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # 기본 dynamic_axes: 모든 입출력의 batch 축
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    input_names = list({k for k in dynamic_axes if not k.startswith("output")}) or ["input"]
    output_names = [k for k in dynamic_axes if k.startswith("output")] or ["output"]

    logger.info("Exporting model to ONNX: %s (opset=%d)", output_path, opset_version)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info("ONNX export complete: %s", output_path)
    return output_path


def run_onnx_inference(
    onnx_path: str | Path,
    inputs: dict[str, Any],
) -> list[Any]:
    """ONNX 모델로 추론을 실행한다.

    Parameters
    ----------
    onnx_path:
        ONNX 모델 파일 경로.
    inputs:
        입력 이름 → numpy 배열 매핑.

    Returns
    -------
    list:
        모델 출력 리스트 (numpy 배열들).
    """
    import onnxruntime as ort  # lazy import

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        msg = f"ONNX model not found: {onnx_path}"
        raise FileNotFoundError(msg)

    logger.info("Running ONNX inference: %s", onnx_path)

    session = ort.InferenceSession(str(onnx_path))
    outputs = session.run(None, inputs)

    logger.info("ONNX inference complete: %d outputs", len(outputs))
    return outputs
