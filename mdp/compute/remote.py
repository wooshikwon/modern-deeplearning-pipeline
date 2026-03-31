"""RemoteExecutor -- SSH를 통해 원격 단일 노드에서 학습을 실행한다."""

from __future__ import annotations

import logging
import uuid

from mdp.compute.base import BaseExecutor
from mdp.compute.job_manager import JobManager
from mdp.compute.ssh import rsync_to_remote, run_remote
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)

_DEFAULT_REMOTE_DIR = "~/mdp_workspace"
_DEFAULT_EXCLUDE = [
    ".git",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    "*.egg-info",
    "mlruns",
    "checkpoints",
    "outputs",
    "wandb",
]


class RemoteExecutor(BaseExecutor):
    """단일 원격 노드에서 SSH를 통해 학습을 실행한다.

    1. rsync로 프로젝트 코드를 원격에 전송
    2. SSH로 ``torchrun`` 명령을 실행
    3. JobManager에 상태를 기록
    """

    def __init__(self, job_manager: JobManager | None = None) -> None:
        self._job_manager = job_manager or JobManager()

    def run(self, settings: Settings) -> str:
        """원격 학습을 실행한다.

        Args:
            settings: 통합 설정. ``config.compute`` 에 host, user, ssh_key,
                      working_dir, gpus 정보가 필요하다.

        Returns:
            job_id 문자열.
        """
        compute = settings.config.compute
        host = compute.host
        user = compute.user
        key_path = compute.ssh_key
        remote_dir = compute.working_dir or _DEFAULT_REMOTE_DIR
        gpus = compute.gpus

        if host is None or user is None:
            raise ValueError(
                "RemoteExecutor는 config.compute.host와 user가 필수입니다."
            )

        job_id = uuid.uuid4().hex
        self._job_manager.create_job(
            job_id=job_id,
            executor="remote",
            settings_json=settings.model_dump_json(),
        )

        try:
            # 1. rsync 코드 전송
            import os

            local_dir = os.getcwd()
            rsync_to_remote(
                local_dir=local_dir,
                remote_dir=remote_dir,
                host=host,
                user=user,
                key_path=key_path,
                exclude=_DEFAULT_EXCLUDE,
            )

            # 2. GPU 수 결정
            nproc = self._resolve_nproc(gpus, host, user, key_path)

            # 3. torchrun 명령 실행
            cmd = (
                f"cd {remote_dir} && "
                f"python -m torch.distributed.run "
                f"--nproc_per_node={nproc} "
                f"-m mdp.compute._torchrun_entry "
                f"--settings-path settings.json"
            )

            # Settings JSON을 원격에 저장
            settings_json = settings.model_dump_json()
            run_remote(
                host=host,
                user=user,
                key_path=key_path,
                command=f"cat > {remote_dir}/settings.json << 'MDPEOF'\n{settings_json}\nMDPEOF",
            )

            # torchrun 실행
            run_remote(
                host=host,
                user=user,
                key_path=key_path,
                command=cmd,
                stream=True,
            )

            self._job_manager.update_status(job_id, "completed")
        except Exception as e:
            logger.exception("원격 학습 실패")
            self._job_manager.update_status(job_id, "failed", error=str(e))
            raise

        return job_id

    def stop(self, job_id: str) -> None:
        """원격 학습 프로세스를 종료한다."""
        record = self._job_manager.get_job(job_id)
        if record is None:
            raise ValueError(f"작업을 찾을 수 없습니다: {job_id}")

        if record.settings is None:
            raise ValueError("작업에 Settings 정보가 없습니다.")

        settings = Settings.model_validate_json(record.settings)
        compute = settings.config.compute

        try:
            run_remote(
                host=compute.host,
                user=compute.user,
                key_path=compute.ssh_key,
                command="pkill -f 'torch.distributed.run.*mdp'",
            )
        except RuntimeError:
            logger.warning("원격 프로세스 종료 시도 (이미 종료되었을 수 있음)")

        self._job_manager.update_status(job_id, "stopped")

    def status(self, job_id: str) -> str:
        """작업 상태를 JobManager에서 조회한다."""
        record = self._job_manager.get_job(job_id)
        if record is None:
            return "completed"
        return record.status

    @staticmethod
    def _resolve_nproc(
        gpus: int | str,
        host: str,
        user: str,
        key_path: str | None,
    ) -> int:
        """원격 GPU 수를 결정한다."""
        if isinstance(gpus, int):
            return max(gpus, 1)

        # "auto" -- 원격에서 GPU 수 조회
        try:
            output = run_remote(
                host=host,
                user=user,
                key_path=key_path,
                command="python3 -c 'import torch; print(torch.cuda.device_count())'",
            )
            count = int(output.strip())
            return max(count, 1)
        except Exception:
            logger.warning("원격 GPU 수 조회 실패, 기본값 1 사용")
            return 1
