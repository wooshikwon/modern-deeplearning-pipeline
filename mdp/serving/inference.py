"""배치 추론 — 모델 포워드 + 콜백 dispatch + (선택) metric 평가.

출력 후처리(softmax, numpy 변환)와 파일 저장은 ``DefaultOutputCallback`` 이 담당한다.
추론 루프 본문은 ``forward_fn(batch)`` -> 콜백 ``on_batch`` -> metric ``update`` 만 수행한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mdp.models.forward import make_forward_fn

logger = logging.getLogger(__name__)


def _detect_device(device: str | torch.device | None) -> torch.device:
    """device 인자가 None이면 GPU/MPS/CPU 순으로 자동 감지."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_make_forward_fn = make_forward_fn


def run_batch_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_path: str | Path,
    output_format: str = "parquet",
    task: str = "classification",
    device: str | torch.device | None = None,
    metrics: list[Any] | None = None,
    callbacks: list[Any] | None = None,
    tokenizer: Any = None,
    metadata: list[dict] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """배치 추론을 실행한다.

    출력 후처리와 파일 저장은 콜백(``DefaultOutputCallback``)이 담당한다.
    추론 루프 본문은 ``forward_fn(batch)`` -> 콜백 dispatch -> metric update만 수행한다.
    ``DefaultOutputCallback`` 이 콜백 리스트에 포함되어 있으면 해당 콜백이 배치별
    후처리(softmax, numpy 변환)와 파일 저장을 수행하고, 포함되어 있지 않으면
    출력 파일이 생성되지 않는다 (콜백 전용 모드).

    Parameters
    ----------
    model:
        추론에 사용할 PyTorch 모델. ``dict`` 를 반환해야 한다.
    dataloader:
        입력 데이터를 제공하는 DataLoader.
    output_path:
        결과 파일 경로 (확장자는 format에 따라 자동 설정).
        ``DefaultOutputCallback`` 이 콜백 리스트에 포함된 경우에만 사용된다.
    output_format:
        ``parquet`` | ``csv`` | ``jsonl``.
        ``DefaultOutputCallback`` 이 콜백 리스트에 포함된 경우에만 사용된다.
    task:
        ``classification`` | ``detection`` | ``text_generation`` | 기타.
        ``DefaultOutputCallback`` 이 콜백 리스트에 포함된 경우에만 사용된다.
    device:
        추론 디바이스. ``None`` 이면 자동 감지.
    metrics:
        평가 metric 리스트. 각 metric은 ``update(outputs, batch)`` / ``compute()``
        프로토콜. None이면 metric 평가 없음. metric은 raw 텐서(outputs)를 직접
        받으므로 ``DefaultOutputCallback`` 유무와 무관하게 동작한다.
    callbacks:
        추론 콜백 리스트. ``BaseInferenceCallback`` 인스턴스는 setup/on_batch/teardown
        lifecycle이 자동 dispatch된다. None이면 콜백 없음.
    tokenizer:
        토크나이저. ``BaseInferenceCallback.setup()`` 에 전달된다. None 가능.
    metadata:
        샘플별 메타데이터 레코드 리스트. pretrained 분기에서 토큰화 전에 추출된
        원본 컬럼(label, topic 등)이다. None이면 메타데이터 없음.
        각 배치의 해당 슬라이스가 ``on_batch(metadata=...)`` kwargs로 콜백에 전달된다.

    Returns
    -------
    tuple[Path | None, dict]:
        (결과 파일 경로, 평가 metric 결과 dict).
        ``DefaultOutputCallback`` 이 콜백 리스트에 포함되어 있으면 해당 콜백의
        ``result_path`` 를, 그렇지 않으면 ``None`` 을 반환한다.
        metrics가 None이면 빈 dict.
    """
    dev = _detect_device(device)
    logger.info("Running batch inference on %s (task=%s)", dev, task)
    if callbacks:
        logger.info("Inference callbacks: %d loaded", len(callbacks))

    from mdp.callbacks.base import BaseInferenceCallback

    if not hasattr(model, "hf_device_map"):
        model = model.to(dev)
    model.eval()

    # Inference callback lifecycle — setup
    inference_cbs = [
        cb for cb in (callbacks or [])
        if isinstance(cb, BaseInferenceCallback)
    ]
    for cb in inference_cbs:
        try:
            cb.setup(model=model, tokenizer=tokenizer)
        except Exception as e:
            if getattr(cb, "critical", False):
                raise
            logger.warning("Inference callback %s.setup 실패: %s", type(cb).__name__, e)

    # Intervention 메타데이터 태깅 (setup 완료 후)
    from mdp.callbacks.interventions import apply_intervention_tags
    apply_intervention_tags(inference_cbs)

    forward_fn = make_forward_fn(model)
    _sample_offset = 0  # metadata 슬라이싱용 누적 오프셋
    _total_samples = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # batch를 device로 이동
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(dev) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(dev)

                outputs = forward_fn(batch)

                # 현재 배치 크기 — metadata 슬라이싱에 사용
                current_batch_size = next(iter(outputs.values())).shape[0]

                # Inference callback — on_batch (metadata 슬라이스 전달)
                meta_slice = None
                if metadata is not None:
                    meta_slice = metadata[_sample_offset:_sample_offset + current_batch_size]
                for cb in inference_cbs:
                    try:
                        cb.on_batch(batch_idx=batch_idx, batch=batch, outputs=outputs, metadata=meta_slice)
                    except Exception as e:
                        if getattr(cb, "critical", False):
                            raise
                        logger.warning("Inference callback %s.on_batch 실패: %s", type(cb).__name__, e)

                _sample_offset += current_batch_size
                _total_samples += current_batch_size

                # Metric 업데이트 — raw 텐서를 직접 받으므로 콜백 유무와 무관
                if metrics:
                    for m in metrics:
                        m.update(outputs, batch)

                if (batch_idx + 1) % 50 == 0:
                    logger.info("  processed %d batches", batch_idx + 1)
    finally:
        # Inference callback lifecycle — teardown (always runs)
        for cb in inference_cbs:
            try:
                cb.teardown()
            except Exception as e:
                if getattr(cb, "critical", False):
                    raise
                logger.warning("Inference callback %s.teardown 실패: %s", type(cb).__name__, e)

    logger.info("Inference complete: %d samples", _total_samples)

    # DefaultOutputCallback에서 result_path 추출
    from mdp.callbacks.inference import DefaultOutputCallback

    output_cb = next(
        (cb for cb in inference_cbs if isinstance(cb, DefaultOutputCallback)),
        None,
    )
    result_path = output_cb.result_path if output_cb else None

    # Metric 집계
    eval_results: dict[str, Any] = {}
    if metrics:
        for m in metrics:
            name = type(m).__name__
            eval_results[name] = m.compute()

    return result_path, eval_results
