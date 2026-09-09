"""In-memory async measurement job store (MVP).

NOT multi-worker / multi-process safe — each process has its own dict.
Use a single uvicorn worker (or sticky routing) for this MVP, or replace
with Redis/DB for production.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from head_biometrics.pipeline import (
    DependencyError,
    DetectionError,
    PipelineError,
    ScaleOptions,
    UploadTooLargeError,
    measure_upload,
)

logger = logging.getLogger(__name__)

# Cap concurrent background measurements (videos are CPU/GPU heavy).
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


_JOBS: Dict[str, JobRecord] = {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_job(job_id: str) -> Optional[JobRecord]:
    with _lock:
        return _JOBS.get(job_id)


def _update_job(job_id: str, **kwargs) -> None:
    with _lock:
        job = _JOBS.get(job_id)
        if job is None:
            return
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = _utcnow_iso()


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
    with _lock:
        _JOBS[job_id] = record

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
    """Reset in-memory store (tests only)."""
    with _lock:
        _JOBS.clear()
