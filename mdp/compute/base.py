"""BaseExecutor ABC -- 모든 실행기의 인터페이스를 정의한다."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mdp.settings.schema import Settings


class BaseExecutor(ABC):
    """실행 레이어 추상 기반 클래스.

    모든 Executor(local, remote, multi-node, cloud)는 이 인터페이스를 구현한다.
    ``run()`` 은 job_id를 반환하고, ``stop()`` / ``status()`` 로 작업을 제어한다.
    """

    @abstractmethod
    def run(self, settings: Settings) -> str:
        """학습 작업을 실행하고 job_id를 반환한다.

        Args:
            settings: Recipe + Config 통합 설정 객체.

        Returns:
            고유한 job_id 문자열.
        """
        ...

    @abstractmethod
    def stop(self, job_id: str) -> None:
        """실행 중인 작업을 중지한다.

        Args:
            job_id: ``run()`` 이 반환한 작업 식별자.
        """
        ...

    @abstractmethod
    def status(self, job_id: str) -> str:
        """작업 상태를 조회한다.

        Args:
            job_id: ``run()`` 이 반환한 작업 식별자.

        Returns:
            ``"running"`` | ``"completed"`` | ``"failed"`` | ``"stopped"`` 중 하나.
        """
        ...
