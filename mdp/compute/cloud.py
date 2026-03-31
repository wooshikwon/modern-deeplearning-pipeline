"""CloudExecutor -- SkyPilot을 통해 클라우드에서 학습을 실행한다."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from mdp.compute.base import BaseExecutor
from mdp.compute.job_manager import JobManager
from mdp.settings.schema import Settings

logger = logging.getLogger(__name__)

_SKYPILOT_YAML_TEMPLATE = """\
name: {cluster_name}

resources:
  cloud: {cloud}
  accelerators: {accelerators}
  use_spot: {use_spot}

num_nodes: {num_nodes}

workdir: .

setup: |
  pip install -e ".[all]"
{extra_setup}

run: |
  python -m torch.distributed.run \\
    --nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \\
    -m mdp.compute._torchrun_entry \\
    --settings-path settings.json
"""


class CloudExecutor(BaseExecutor):
    """SkyPilot 기반 클라우드 실행기.

    ``sky launch`` 로 클러스터를 프로비저닝하고 학습을 실행한다.
    ``skypilot`` 은 lazy import로 처리한다.
    """

    def __init__(self, job_manager: JobManager | None = None) -> None:
        self._job_manager = job_manager or JobManager()

    def run(self, settings: Settings) -> str:
        """클라우드 학습을 실행한다.

        SkyPilot YAML을 생성하고 ``sky launch`` 를 실행한다.

        Args:
            settings: 통합 설정. ``config.compute.distributed`` 에
                      cloud, accelerators 등 SkyPilot 옵션을 지정한다.

        Returns:
            job_id 문자열.
        """
        import sky  # lazy import

        job_id = uuid.uuid4().hex
        cluster_name = f"mdp-{job_id[:8]}"

        self._job_manager.create_job(
            job_id=job_id,
            executor="cloud",
            settings_json=settings.model_dump_json(),
        )

        try:
            # 1. Settings JSON 저장
            settings_path = Path("settings.json")
            settings_path.write_text(settings.model_dump_json(indent=2))

            # 2. SkyPilot YAML 생성
            yaml_path = self._generate_yaml(settings, cluster_name)

            # 3. sky launch
            task = sky.Task.from_yaml(str(yaml_path))
            sky.launch(task, cluster_name=cluster_name)

            self._job_manager.update_status(job_id, "completed")
        except Exception as e:
            logger.exception("클라우드 학습 실패")
            self._job_manager.update_status(job_id, "failed", error=str(e))
            raise
        finally:
            # 임시 YAML 정리
            if "yaml_path" in locals():
                Path(yaml_path).unlink(missing_ok=True)

        return job_id

    def stop(self, job_id: str) -> None:
        """클라우드 클러스터를 종료한다."""
        import sky  # lazy import

        cluster_name = f"mdp-{job_id[:8]}"

        try:
            sky.down(cluster_name)
            logger.info("클러스터 종료: %s", cluster_name)
        except Exception as e:
            logger.warning("클러스터 종료 실패: %s", e)

        self._job_manager.update_status(job_id, "stopped")

    def status(self, job_id: str) -> str:
        """작업 상태를 조회한다.

        JobManager 레코드를 먼저 확인하고,
        ``running`` 상태이면 ``sky status`` 로 실제 클러스터 상태를 교차 검증한다.
        """
        record = self._job_manager.get_job(job_id)
        if record is None:
            return "completed"

        if record.status == "running":
            return self._check_sky_status(job_id)

        return record.status

    def _check_sky_status(self, job_id: str) -> str:
        """SkyPilot 클러스터 상태를 확인한다."""
        try:
            import sky  # lazy import

            cluster_name = f"mdp-{job_id[:8]}"
            statuses = sky.status(cluster_names=[cluster_name])

            if not statuses:
                return "completed"

            cluster_status = statuses[0]["status"]
            if cluster_status.name == "UP":
                return "running"
            return "completed"
        except Exception:
            logger.warning("SkyPilot 상태 조회 실패, DB 상태 반환")
            record = self._job_manager.get_job(job_id)
            return record.status if record else "completed"

    @staticmethod
    def _generate_yaml(settings: Settings, cluster_name: str) -> str:
        """SkyPilot YAML 파일을 생성하고 경로를 반환한다."""
        compute = settings.config.compute
        dist_config: dict[str, Any] = (
            compute.distributed if isinstance(compute.distributed, dict) else {}
        )

        cloud = dist_config.get("cloud", "aws")
        accelerators = dist_config.get("accelerators", "A100:1")
        use_spot = str(dist_config.get("use_spot", False)).lower()
        num_nodes = dist_config.get("num_nodes", 1)

        # 추가 setup commands
        setup_cmds = settings.config.environment_setup.setup_commands
        extra_setup = ""
        if setup_cmds:
            extra_setup = "\n".join(f"  {cmd}" for cmd in setup_cmds)

        yaml_content = _SKYPILOT_YAML_TEMPLATE.format(
            cluster_name=cluster_name,
            cloud=cloud,
            accelerators=accelerators,
            use_spot=use_spot,
            num_nodes=num_nodes,
            extra_setup=extra_setup,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            prefix="mdp_sky_",
        ) as f:
            f.write(yaml_content)
            return f.name
