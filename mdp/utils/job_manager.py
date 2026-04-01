"""JobManager -- SQLite 기반 작업 상태 관리."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JobStatus = Literal["running", "completed", "failed", "stopped"]

_DEFAULT_DB_DIR = Path.home() / ".mdp"
_DEFAULT_DB_NAME = "jobs.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    executor    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    settings    TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    error       TEXT
)
"""


@dataclass
class JobRecord:
    """작업 레코드."""

    job_id: str
    executor: str
    status: JobStatus
    settings: str | None
    created_at: float
    updated_at: float
    error: str | None = None


class JobManager:
    """SQLite CRUD 래퍼.

    DB 파일은 ``~/.mdp/jobs.db`` 에 저장된다.
    별도의 마이그레이션 없이 테이블을 자동 생성한다.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
            db_path = _DEFAULT_DB_DIR / _DEFAULT_DB_NAME
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    # ── Connection management ──

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self) -> None:
        self.conn.execute(_CREATE_TABLE_SQL)
        self.conn.commit()

    def close(self) -> None:
        """DB 연결을 닫는다."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── CRUD ──

    def create_job(
        self,
        job_id: str,
        executor: str,
        settings_json: str | None = None,
    ) -> JobRecord:
        """새 작업 레코드를 생성한다.

        Args:
            job_id: 고유 작업 식별자.
            executor: 실행기 이름 (``"local"``, ``"remote"`` 등).
            settings_json: Settings JSON 직렬화 (선택).

        Returns:
            생성된 ``JobRecord``.
        """
        now = time.time()
        self.conn.execute(
            "INSERT INTO jobs (job_id, executor, status, settings, created_at, updated_at) "
            "VALUES (?, ?, 'running', ?, ?, ?)",
            (job_id, executor, settings_json, now, now),
        )
        self.conn.commit()
        return JobRecord(
            job_id=job_id,
            executor=executor,
            status="running",
            settings=settings_json,
            created_at=now,
            updated_at=now,
        )

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
    ) -> None:
        """작업 상태를 갱신한다.

        Args:
            job_id: 대상 작업 식별자.
            status: 새 상태.
            error: 에러 메시지 (``"failed"`` 시).
        """
        now = time.time()
        self.conn.execute(
            "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
            (status, error, now, job_id),
        )
        self.conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        """작업 레코드를 조회한다.

        Args:
            job_id: 대상 작업 식별자.

        Returns:
            ``JobRecord`` 또는 ``None``.
        """
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """작업 목록을 조회한다.

        Args:
            status: 필터링할 상태 (``None`` 이면 전체).
            limit: 최대 반환 개수.

        Returns:
            ``JobRecord`` 리스트 (최신순).
        """
        if status is not None:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            executor=row["executor"],
            status=row["status"],
            settings=row["settings"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"],
        )
