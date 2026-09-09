"""Async measurement job store + background workers.

Backends (env ``JOB_STORE``):
  - ``memory`` (default) — in-process dict; lost on restart
  - ``sqlite`` — persist status/result/error under ``JOB_STORE_PATH``
    (default ``./data/jobs.sqlite3``)

**Single-process assumption:** the thread pool and store are still not safe
across multiple uvicorn workers. Use one worker (or sticky routing). SQLite
survives process restart for terminal jobs; in-flight queued/running rows are
marked failed on store open after a crash/restart.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from head_biometrics.pipeline import (
    DependencyError,
    DetectionError,
    MeasurementResult,
    PipelineError,
    ScaleOptions,
    UploadTooLargeError,
    measure_upload,
)

logger = logging.getLogger(__name__)

_MAX_WORKERS = 2
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="measure-job")
_lock = threading.Lock()


@dataclass
class JobRecord:
    job_id: str
    status: str  # queued | running | succeeded | failed
    created_at: str
    updated_at: str
    filename: Optional[str] = None
    clockwise: bool = False
    scale_mode: Optional[str] = None
    result: Optional[Any] = None  # MeasurementResult on success
    error: Optional[str] = None
    error_code: Optional[int] = None  # HTTP-ish: 422 / 503 / 413 / 500


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _result_to_json(result: Optional[MeasurementResult]) -> Optional[str]:
    if result is None:
        return None
    payload = {
        "circumference": result.circumference,
        "front_to_nape": result.front_to_nape,
        "ear_to_ear": result.ear_to_ear,
        "head_width": result.head_width,
        "length": result.length,
        "demo_mode": result.demo_mode,
        "note": result.note,
        "scale_mode": result.scale_mode,
        "scale_note": result.scale_note,
        "mm_per_pixel": result.mm_per_pixel,
        "confidence": result.confidence,
        "warnings": list(result.warnings or []),
        "scale_frames_used": result.scale_frames_used,
    }
    return json.dumps(payload)


def _result_from_json(raw: Optional[str]) -> Optional[MeasurementResult]:
    if not raw:
        return None
    data = json.loads(raw)
    return MeasurementResult(
        circumference=int(data["circumference"]),
        front_to_nape=int(data["front_to_nape"]),
        ear_to_ear=int(data["ear_to_ear"]),
        head_width=int(data["head_width"]),
        length=int(data["length"]),
        demo_mode=bool(data.get("demo_mode", False)),
        note=data.get("note"),
        scale_mode=data.get("scale_mode"),
        scale_note=data.get("scale_note"),
        mm_per_pixel=data.get("mm_per_pixel"),
        confidence=data.get("confidence"),
        warnings=list(data.get("warnings") or []),
        scale_frames_used=data.get("scale_frames_used"),
    )


class JobStore(Protocol):
    def get(self, job_id: str) -> Optional[JobRecord]: ...

    def put(self, record: JobRecord) -> None: ...

    def update(self, job_id: str, **kwargs: Any) -> None: ...

    def clear(self) -> None: ...


class MemoryJobStore:
    """In-process dict store (default)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def put(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)
            job.updated_at = _utcnow_iso()

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()


class SqliteJobStore:
    """SQLite-backed job metadata (status / result / error).

    Still single-process: workers run in this process's thread pool. Persistence
    lets clients poll completed jobs after a restart; orphaned queued/running
    rows from a prior crash are marked failed on open.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    filename TEXT,
                    clockwise INTEGER NOT NULL DEFAULT 0,
                    scale_mode TEXT,
                    result_json TEXT,
                    error TEXT,
                    error_code INTEGER
                )
                """
            )
            self._conn.commit()
            self._fail_orphaned_locked()

    def _fail_orphaned_locked(self) -> None:
        now = _utcnow_iso()
        cur = self._conn.execute(
            "UPDATE jobs SET status = ?, error = ?, error_code = ?, updated_at = ? "
            "WHERE status IN ('queued', 'running')",
            (
                "failed",
                "Job interrupted by process restart (single-process store).",
                500,
                now,
            ),
        )
        if cur.rowcount:
            logger.warning(
                "Marked %s orphaned sqlite job(s) as failed after store open",
                cur.rowcount,
            )
        self._conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            filename=row["filename"],
            clockwise=bool(row["clockwise"]),
            scale_mode=row["scale_mode"],
            result=_result_from_json(row["result_json"]),
            error=row["error"],
            error_code=row["error_code"],
        )

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def put(self, record: JobRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    job_id, status, created_at, updated_at, filename, clockwise,
                    scale_mode, result_json, error, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    record.status,
                    record.created_at,
                    record.updated_at,
                    record.filename,
                    1 if record.clockwise else 0,
                    record.scale_mode,
                    _result_to_json(record.result),
                    record.error,
                    record.error_code,
                ),
            )
            self._conn.commit()

    def update(self, job_id: str, **kwargs: Any) -> None:
        allowed = {
            "status",
            "filename",
            "clockwise",
            "scale_mode",
            "result",
            "error",
            "error_code",
        }
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            )
            row = cur.fetchone()
            if row is None:
                return
            record = self._row_to_record(row)
            for k, v in kwargs.items():
                if k not in allowed:
                    continue
                setattr(record, k, v)
            record.updated_at = _utcnow_iso()
            self._conn.execute(
                """
                UPDATE jobs SET
                    status = ?, updated_at = ?, filename = ?, clockwise = ?,
                    scale_mode = ?, result_json = ?, error = ?, error_code = ?
                WHERE job_id = ?
                """,
                (
                    record.status,
                    record.updated_at,
                    record.filename,
                    1 if record.clockwise else 0,
                    record.scale_mode,
                    _result_to_json(record.result),
                    record.error,
                    record.error_code,
                    job_id,
                ),
            )
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_store: Optional[JobStore] = None


def _make_store_from_env() -> JobStore:
    backend = os.environ.get("JOB_STORE", "memory").strip().lower() or "memory"
    if backend == "sqlite":
        path = os.environ.get("JOB_STORE_PATH", "./data/jobs.sqlite3").strip()
        if not path:
            path = "./data/jobs.sqlite3"
        logger.info("Job store: sqlite at %s", path)
        return SqliteJobStore(path)
    if backend != "memory":
        logger.warning("Unknown JOB_STORE=%r — falling back to memory", backend)
    logger.info("Job store: memory")
    return MemoryJobStore()


def get_store() -> JobStore:
    global _store
    with _lock:
        if _store is None:
            _store = _make_store_from_env()
        return _store


def configure_job_store(
    backend: str = "memory",
    path: Optional[str] = None,
) -> JobStore:
    """Replace the active store (tests / explicit setup)."""
    global _store
    with _lock:
        if isinstance(_store, SqliteJobStore):
            try:
                _store.close()
            except Exception:  # noqa: BLE001
                pass
        if backend == "sqlite":
            _store = SqliteJobStore(path or "./data/jobs.sqlite3")
        else:
            _store = MemoryJobStore()
        return _store


def get_job(job_id: str) -> Optional[JobRecord]:
    return get_store().get(job_id)


def _update_job(job_id: str, **kwargs: Any) -> None:
    get_store().update(job_id, **kwargs)


def _run_job(
    job_id: str,
    file_bytes: bytes,
    filename: Optional[str],
    clockwise: bool,
    scale_options: ScaleOptions,
    max_bytes: int,
) -> None:
    from io import BytesIO

    _update_job(job_id, status="running")
    try:
        result = measure_upload(
            BytesIO(file_bytes),
            filename=filename,
            clockwise=clockwise,
            scale_options=scale_options,
            max_bytes=max_bytes,
        )
        _update_job(job_id, status="succeeded", result=result, error=None, error_code=None)
    except UploadTooLargeError as exc:
        _update_job(job_id, status="failed", error=str(exc), error_code=413)
    except DetectionError as exc:
        _update_job(job_id, status="failed", error=str(exc), error_code=422)
    except DependencyError as exc:
        _update_job(
            job_id,
            status="failed",
            error=(
                f"{exc}. Install CV dependencies (see README) or set DEMO_MODE=1 "
                "for a clearly labeled mock response."
            ),
            error_code=503,
        )
    except PipelineError as exc:
        logger.exception("Job %s pipeline error", job_id)
        _update_job(job_id, status="failed", error=str(exc), error_code=500)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s unexpected failure", job_id)
        _update_job(
            job_id,
            status="failed",
            error=f"Unexpected server error: {exc}",
            error_code=500,
        )


def submit_measure_job(
    *,
    file_bytes: bytes,
    filename: Optional[str],
    clockwise: bool,
    scale_options: ScaleOptions,
    max_bytes: int,
) -> JobRecord:
    """Create a queued job and schedule background processing. Returns the record."""
    now = _utcnow_iso()
    job_id = str(uuid.uuid4())
    record = JobRecord(
        job_id=job_id,
        status="queued",
        created_at=now,
        updated_at=now,
        filename=filename,
        clockwise=clockwise,
        scale_mode=scale_options.scale_mode,
    )
    get_store().put(record)

    _executor.submit(
        _run_job,
        job_id,
        file_bytes,
        filename,
        clockwise,
        scale_options,
        max_bytes,
    )
    return record


def clear_jobs_for_tests() -> None:
    """Reset store contents and ensure memory backend for subsequent tests."""
    configure_job_store("memory")
    get_store().clear()
