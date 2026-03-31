"""vLLM 서버 관리 — OpenAI-호환 API 서버 시작."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def start_vllm_server(
    model_name: str,
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    quantization: str | None = None,
) -> subprocess.Popen:
    """vLLM OpenAI-호환 API 서버를 시작한다.

    Parameters
    ----------
    model_name:
        HuggingFace 모델 이름 또는 로컬 경로.
    port:
        API 서버 포트. 기본 8000.
    tensor_parallel_size:
        텐서 병렬 GPU 수. 기본 1.
    max_model_len:
        최대 시퀀스 길이. ``None`` 이면 모델 기본값 사용.
    quantization:
        양자화 방법 (예: ``awq``, ``gptq``). ``None`` 이면 비활성.

    Returns
    -------
    subprocess.Popen:
        vLLM 서버 프로세스 핸들.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
    ]

    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])

    if quantization is not None:
        cmd.extend(["--quantization", quantization])

    logger.info("Starting vLLM server: %s", " ".join(cmd))

    process = subprocess.Popen(cmd)  # noqa: S603
    logger.info("vLLM server started (PID %d) on port %d", process.pid, port)
    return process
