"""SSH 유틸리티 -- paramiko SSH 연결, rsync, 원격 명령 실행."""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def ssh_connect(
    host: str,
    user: str,
    key_path: str | None = None,
) -> Any:
    """paramiko SSHClient를 생성하여 원격 호스트에 연결한다.

    ``paramiko`` 는 lazy import로 처리한다.

    Args:
        host: 원격 호스트 주소.
        user: SSH 사용자명.
        key_path: SSH 키 파일 경로 (``None`` 이면 기본 키 사용).

    Returns:
        연결된 ``paramiko.SSHClient`` 인스턴스.
    """
    import paramiko  # lazy import

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "username": user,
    }
    if key_path is not None:
        connect_kwargs["key_filename"] = key_path

    client.connect(**connect_kwargs)
    logger.info("SSH 연결 성공: %s@%s", user, host)
    return client


def rsync_to_remote(
    local_dir: str,
    remote_dir: str,
    host: str,
    user: str,
    key_path: str | None = None,
    exclude: list[str] | None = None,
) -> None:
    """rsync로 로컬 디렉토리를 원격에 동기화한다.

    Args:
        local_dir: 로컬 소스 디렉토리 (trailing slash 자동 추가).
        remote_dir: 원격 대상 디렉토리.
        host: 원격 호스트 주소.
        user: SSH 사용자명.
        key_path: SSH 키 파일 경로.
        exclude: rsync --exclude 패턴 목록.
    """
    # trailing slash 보장
    if not local_dir.endswith("/"):
        local_dir += "/"

    cmd = ["rsync", "-avz", "--delete"]

    if key_path is not None:
        cmd.extend(["-e", f"ssh -i {key_path}"])

    if exclude is not None:
        for pattern in exclude:
            cmd.extend(["--exclude", pattern])

    cmd.append(local_dir)
    cmd.append(f"{user}@{host}:{remote_dir}")

    logger.info("rsync 실행: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    logger.info("rsync 완료: %s → %s@%s:%s", local_dir, user, host, remote_dir)


def run_remote(
    host: str,
    user: str,
    key_path: str | None = None,
    command: str = "",
    stream: bool = False,
) -> str:
    """SSH로 원격 명령을 실행하고 stdout을 반환한다.

    Args:
        host: 원격 호스트 주소.
        user: SSH 사용자명.
        key_path: SSH 키 파일 경로.
        command: 실행할 셸 명령.
        stream: ``True`` 이면 stdout을 실시간으로 출력한다.

    Returns:
        명령의 stdout 문자열.

    Raises:
        RuntimeError: 원격 명령이 실패한 경우.
    """
    client = ssh_connect(host, user, key_path)
    try:
        _, stdout_ch, stderr_ch = client.exec_command(command)

        if stream:
            output_lines: list[str] = []
            for line in stdout_ch:
                stripped = line.strip("\n")
                print(stripped)
                output_lines.append(stripped)
            stdout = "\n".join(output_lines)
        else:
            stdout = stdout_ch.read().decode("utf-8")

        stderr = stderr_ch.read().decode("utf-8")
        exit_code = stdout_ch.channel.recv_exit_status()

        if exit_code != 0:
            raise RuntimeError(
                f"원격 명령 실패 (exit={exit_code}): {stderr.strip()}"
            )

        return stdout
    finally:
        client.close()
