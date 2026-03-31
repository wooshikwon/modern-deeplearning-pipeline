"""TorchServe 패키징 및 서버 관리 — MAR 아카이브 생성, 서버 시작/중지."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_mar(
    model_path: str | Path,
    handler: str,
    model_name: str,
    output_dir: str | Path = "model_store",
) -> Path:
    """torch-model-archiver로 MAR 아카이브를 생성한다.

    Parameters
    ----------
    model_path:
        직렬화된 모델 파일 경로 (``.pt`` 또는 ``.pth``).
    handler:
        TorchServe 핸들러 (예: ``image_classifier``, 커스텀 핸들러 경로).
    model_name:
        모델 이름 (MAR 파일명이 된다).
    output_dir:
        MAR 파일 출력 디렉토리. 기본 ``model_store``.

    Returns
    -------
    Path:
        생성된 MAR 파일 경로.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mar_path = output_dir / f"{model_name}.mar"

    cmd = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--version", "1.0",
        "--serialized-file", str(model_path),
        "--handler", handler,
        "--export-path", str(output_dir),
        "--force",
    ]

    logger.info("Creating MAR archive: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("torch-model-archiver failed: %s", result.stderr)
        msg = f"torch-model-archiver failed (exit {result.returncode}): {result.stderr}"
        raise RuntimeError(msg)

    logger.info("MAR archive created: %s", mar_path)
    return mar_path


def start_torchserve(
    model_store: str | Path = "model_store",
    port: int = 8080,
    workers: int = 1,
) -> subprocess.Popen:
    """TorchServe를 시작한다.

    Parameters
    ----------
    model_store:
        모델 저장소 디렉토리.
    port:
        Inference API 포트. 기본 8080.
    workers:
        워커 프로세스 수. 기본 1.

    Returns
    -------
    subprocess.Popen:
        TorchServe 프로세스 핸들.
    """
    model_store = Path(model_store)
    if not model_store.exists():
        msg = f"Model store not found: {model_store}"
        raise FileNotFoundError(msg)

    cmd = [
        "torchserve",
        "--start",
        "--model-store", str(model_store),
        "--ts-config", "",
        "--foreground",
    ]

    # config.properties 대신 환경변수로 포트/워커 설정
    env_overrides = {
        "TS_INFERENCE_ADDRESS": f"http://0.0.0.0:{port}",
        "TS_NUMBER_OF_NETTY_THREADS": str(workers),
    }

    import os
    env = {**os.environ, **env_overrides}

    # --ts-config 빈 문자열 제거 → 깔끔한 커맨드
    cmd = [c for c in cmd if c]

    logger.info("Starting TorchServe: port=%d, workers=%d, store=%s", port, workers, model_store)

    process = subprocess.Popen(cmd, env=env)  # noqa: S603
    logger.info("TorchServe started (PID %d)", process.pid)
    return process


def stop_torchserve() -> None:
    """TorchServe를 중지한다."""
    cmd = ["torchserve", "--stop"]
    logger.info("Stopping TorchServe")
    subprocess.run(cmd, capture_output=True, text=True, check=False)
