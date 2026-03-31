"""MultiNodeExecutor -- 여러 원격 노드에서 분산 학습을 실행한다."""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

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


class MultiNodeExecutor(BaseExecutor):
    """멀티 노드 분산 학습 실행기.

    ``config.compute.nodes`` 리스트의 첫 번째 노드가 master가 된다.
    모든 노드에 rsync로 코드를 전송한 뒤 ThreadPoolExecutor로 병렬 torchrun을 실행한다.
    """

    def __init__(self, job_manager: JobManager | None = None) -> None:
        self._job_manager = job_manager or JobManager()

    def run(self, settings: Settings) -> str:
        """멀티 노드 분산 학습을 실행한다.

        Args:
            settings: 통합 설정. ``config.compute.nodes`` 에 노드 정보가 필요하다.
                각 노드는 ``{"host": str, "user": str, "ssh_key": str?, "gpus": int?}``
                형태.

        Returns:
            job_id 문자열.
        """
        compute = settings.config.compute
        nodes = compute.nodes

        if not nodes or len(nodes) < 2:
            raise ValueError(
                "MultiNodeExecutor는 config.compute.nodes에 2개 이상의 노드가 필요합니다."
            )

        job_id = uuid.uuid4().hex
        self._job_manager.create_job(
            job_id=job_id,
            executor="multi_node",
            settings_json=settings.model_dump_json(),
        )

        try:
            master_node = nodes[0]
            master_host = master_node["host"]
            master_port = master_node.get("port", 29500)
            remote_dir = compute.working_dir or _DEFAULT_REMOTE_DIR

            # 1. 모든 노드에 rsync (병렬)
            self._rsync_all_nodes(nodes, remote_dir)

            # 2. Settings JSON을 모든 노드에 전송
            settings_json = settings.model_dump_json()
            self._distribute_settings(nodes, remote_dir, settings_json)

            # 3. 병렬 torchrun 실행
            self._launch_all_nodes(
                nodes=nodes,
                master_host=master_host,
                master_port=master_port,
                remote_dir=remote_dir,
            )

            self._job_manager.update_status(job_id, "completed")
        except Exception as e:
            logger.exception("멀티 노드 학습 실패")
            self._job_manager.update_status(job_id, "failed", error=str(e))
            raise

        return job_id

    def stop(self, job_id: str) -> None:
        """모든 노드의 학습 프로세스를 종료한다."""
        record = self._job_manager.get_job(job_id)
        if record is None:
            raise ValueError(f"작업을 찾을 수 없습니다: {job_id}")

        if record.settings is None:
            raise ValueError("작업에 Settings 정보가 없습니다.")

        settings = Settings.model_validate_json(record.settings)
        nodes = settings.config.compute.nodes or []

        for node in nodes:
            try:
                run_remote(
                    host=node["host"],
                    user=node["user"],
                    key_path=node.get("ssh_key"),
                    command="pkill -f 'torch.distributed.run.*mdp'",
                )
            except (RuntimeError, Exception):
                logger.warning(
                    "노드 %s 프로세스 종료 시도 (이미 종료되었을 수 있음)",
                    node["host"],
                )

        self._job_manager.update_status(job_id, "stopped")

    def status(self, job_id: str) -> str:
        """작업 상태를 JobManager에서 조회한다."""
        record = self._job_manager.get_job(job_id)
        if record is None:
            return "completed"
        return record.status

    # ── Internal ──

    @staticmethod
    def _rsync_all_nodes(
        nodes: list[dict[str, Any]],
        remote_dir: str,
    ) -> None:
        """모든 노드에 병렬 rsync."""
        import os

        local_dir = os.getcwd()

        def _sync_one(node: dict[str, Any]) -> None:
            rsync_to_remote(
                local_dir=local_dir,
                remote_dir=remote_dir,
                host=node["host"],
                user=node["user"],
                key_path=node.get("ssh_key"),
                exclude=_DEFAULT_EXCLUDE,
            )

        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            futures = {pool.submit(_sync_one, n): n for n in nodes}
            for future in as_completed(futures):
                node = futures[future]
                future.result()  # raises on error
                logger.info("rsync 완료: %s", node["host"])

    @staticmethod
    def _distribute_settings(
        nodes: list[dict[str, Any]],
        remote_dir: str,
        settings_json: str,
    ) -> None:
        """모든 노드에 Settings JSON을 전송한다."""
        for node in nodes:
            run_remote(
                host=node["host"],
                user=node["user"],
                key_path=node.get("ssh_key"),
                command=(
                    f"cat > {remote_dir}/settings.json << 'MDPEOF'\n"
                    f"{settings_json}\n"
                    f"MDPEOF"
                ),
            )

    @staticmethod
    def _launch_all_nodes(
        nodes: list[dict[str, Any]],
        master_host: str,
        master_port: int,
        remote_dir: str,
    ) -> None:
        """ThreadPoolExecutor로 모든 노드에서 torchrun을 병렬 실행한다."""
        nnodes = len(nodes)

        def _launch_one(rank: int, node: dict[str, Any]) -> None:
            nproc = node.get("gpus", 1)
            cmd = (
                f"cd {remote_dir} && "
                f"python -m torch.distributed.run "
                f"--nproc_per_node={nproc} "
                f"--nnodes={nnodes} "
                f"--node_rank={rank} "
                f"--master_addr={master_host} "
                f"--master_port={master_port} "
                f"-m mdp.compute._torchrun_entry "
                f"--settings-path settings.json"
            )
            run_remote(
                host=node["host"],
                user=node["user"],
                key_path=node.get("ssh_key"),
                command=cmd,
                stream=True,
            )

        with ThreadPoolExecutor(max_workers=nnodes) as pool:
            futures = {
                pool.submit(_launch_one, rank, node): node
                for rank, node in enumerate(nodes)
            }
            errors: list[Exception] = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    node = futures[future]
                    logger.error("노드 %s 실행 실패: %s", node["host"], e)
                    errors.append(e)

            if errors:
                raise RuntimeError(
                    f"멀티 노드 학습 중 {len(errors)}개 노드 실패"
                ) from errors[0]
