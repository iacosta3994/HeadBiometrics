"""Lightweight in-process metrics counters (JSON exposition at GET /metrics)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger("head_biometrics.access")

_lock = threading.Lock()
_counters: Dict[str, int] = {
    "requests_total": 0,
    "measure_success": 0,
    "measure_failure": 0,
    "jobs_created": 0,
}


def reset_metrics_for_tests() -> None:
    with _lock:
        for k in _counters:
            _counters[k] = 0


def incr(name: str, amount: int = 1) -> None:
    with _lock:
        if name not in _counters:
            _counters[name] = 0
        _counters[name] += amount


def snapshot() -> Dict[str, Any]:
    with _lock:
        return dict(_counters)


def record_request(path: str, method: str, status_code: int, duration_ms: float) -> None:
    incr("requests_total")
    # Structured access log for measure endpoints
    if path.startswith("/v1/measure"):
        logger.info(
            "access method=%s path=%s status=%s duration_ms=%.1f",
            method,
            path,
            status_code,
            duration_ms,
        )


def record_measure_outcome(success: bool) -> None:
    if success:
        incr("measure_success")
    else:
        incr("measure_failure")


def record_job_created() -> None:
    incr("jobs_created")
