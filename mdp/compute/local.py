"""LocalExecutor -- 로컬 머신에서 학습을 실행한다."""

from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from pathlib import Path

from mdp.compute.base import BaseExecutor
from mdp.compute.job_manager import JobManager
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)


class LocalExecutor(BaseExecutor):
    """로컬 실행기.

    단일 GPU / CPU 환경에서는 동기적으로 Trainer를 실행한다.
    멀티 GPU(CUDA device_count > 1)이면 ``torchrun`` subprocess를 통해
    분산 학습을 실행한다.
    """

    def __init__(self, job_manager: JobManager | None = None) -> None:
        self._job_manager = job_manager or JobManager()

    def run(self, settings: Settings) -> str:
        """학습을 실행하고 job_id를 반환한다.

        GPU가 여러 개이면 subprocess로 ``torchrun`` 을 실행하고,
        그렇지 않으면 in-process로 Trainer를 직접 호출한다.
        """
        job_id = uuid.uuid4().hex
        settings_json = settings.model_dump_json()
        self._job_manager.create_job(
            job_id=job_id,
            executor="local",
            settings_json=settings_json,
        )

        try:
            gpu_count = self._detect_gpu_count()
            if gpu_count > 1:
                self._run_distributed(settings, gpu_count)
            else:
                self._run_single(settings)
            self._job_manager.update_status(job_id, "completed")
        except Exception as e:
            logger.exception("로컬 학습 실패")
            self._job_manager.update_status(job_id, "failed", error=str(e))
            raise

        return job_id

    def stop(self, job_id: str) -> None:
        """로컬 동기 실행은 중지를 지원하지 않는다."""
        raise NotImplementedError(
            "LocalExecutor는 동기 실행이므로 stop()을 지원하지 않습니다. "
            "프로세스를 직접 종료하세요."
        )

    def status(self, job_id: str) -> str:
        """작업 상태를 JobManager에서 조회한다."""
        record = self._job_manager.get_job(job_id)
        if record is None:
            return "completed"
        return record.status

    # ── Internal ──

    @staticmethod
    def _detect_gpu_count() -> int:
        """사용 가능한 CUDA GPU 수를 반환한다."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 0

    @staticmethod
    def _run_single(settings: Settings) -> None:
        """단일 프로세스에서 Trainer를 실행한다."""
        from mdp.factory.factory import Factory
        from mdp.training.trainer import Trainer

        factory = Factory(settings)
        model = factory.create_model()
        dataloaders = factory.create_dataloaders()

        trainer = Trainer(
            settings=settings,
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders.get("val"),
        )
        trainer.train()

    @staticmethod
    def _run_distributed(settings: Settings, nproc: int) -> None:
        """``torchrun`` subprocess로 분산 학습을 실행한다.

        임시 설정 파일을 생성하여 torchrun 워커에 전달한다.
        """
        import tempfile

        # Settings를 임시 JSON으로 저장
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="mdp_settings_"
        ) as f:
            f.write(settings.model_dump_json(indent=2))
            settings_path = f.name

        try:
            # torchrun 실행 스크립트 경로
            launch_script = Path(__file__).parent / "_torchrun_entry.py"

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                str(launch_script),
                "--settings-path",
                settings_path,
            ]

            logger.info("torchrun 실행: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True,
            )
            logger.info("torchrun 완료 (returncode=%d)", result.returncode)
        finally:
            # 임시 파일 정리
            Path(settings_path).unlink(missing_ok=True)
