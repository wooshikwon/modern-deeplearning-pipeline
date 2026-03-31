"""mdp.compute -- 실행 레이어.

실행 환경(local, remote, multi-node, cloud)에 따라 학습을 디스패치한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mdp.compute.base import BaseExecutor
from mdp.compute.estimator import MemoryEstimator
from mdp.compute.job_manager import JobManager
from mdp.compute.local import LocalExecutor

if TYPE_CHECKING:
    from mdp.compute.cloud import CloudExecutor
    from mdp.compute.multi_node import MultiNodeExecutor
    from mdp.compute.remote import RemoteExecutor

# Executor 레지스트리: config.compute.target → Executor 클래스 경로
EXECUTOR_MAP: dict[str, str] = {
    "local": "mdp.compute.local.LocalExecutor",
    "remote": "mdp.compute.remote.RemoteExecutor",
    "multi_node": "mdp.compute.multi_node.MultiNodeExecutor",
    "cloud": "mdp.compute.cloud.CloudExecutor",
}


def get_executor(target: str, **kwargs) -> BaseExecutor:
    """target 이름으로 Executor 인스턴스를 생성한다.

    Args:
        target: ``"local"`` | ``"remote"`` | ``"multi_node"`` | ``"cloud"``.
        **kwargs: Executor 생성자에 전달할 추가 인자.

    Returns:
        ``BaseExecutor`` 구현체 인스턴스.

    Raises:
        ValueError: 알 수 없는 target.
    """
    if target not in EXECUTOR_MAP:
        raise ValueError(
            f"알 수 없는 executor target: '{target}'. "
            f"사용 가능: {list(EXECUTOR_MAP.keys())}"
        )

    class_path = EXECUTOR_MAP[target]
    module_path, _, class_name = class_path.rpartition(".")

    import importlib

    module = importlib.import_module(module_path)
    klass = getattr(module, class_name)
    return klass(**kwargs)


__all__ = [
    "BaseExecutor",
    "LocalExecutor",
    "MemoryEstimator",
    "JobManager",
    "EXECUTOR_MAP",
    "get_executor",
]
