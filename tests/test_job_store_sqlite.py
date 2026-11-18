"""SQLite job store durability tests (tmp path)."""

from __future__ import annotations

import time
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from head_biometrics.app import app
from head_biometrics.jobs import (
    JobRecord,
    SqliteJobStore,
    clear_jobs_for_tests,
    configure_job_store,
    get_job,
    submit_measure_job,
)
from head_biometrics.pipeline import MeasurementResult, ScaleOptions


@pytest.fixture
def sqlite_store(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    store = configure_job_store("sqlite", str(db))
    yield store, db
    clear_jobs_for_tests()


def _result(**kwargs):
    base = dict(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        note=None,
        scale_mode="mm_per_pixel",
        scale_note=None,
        mm_per_pixel=0.25,
        confidence=0.9,
        warnings=["heuristic"],
        scale_frames_used=None,
    )
    base.update(kwargs)
    return MeasurementResult(**base)


def test_sqlite_put_get_update(sqlite_store):
    store, _db = sqlite_store
    record = JobRecord(
        job_id="abc-123",
        status="queued",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        filename="clip.mp4",
        clockwise=True,
        scale_mode="mm_per_pixel",
    )
    store.put(record)
    loaded = store.get("abc-123")
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.filename == "clip.mp4"
    assert loaded.clockwise is True

    result = _result()
    store.update("abc-123", status="succeeded", result=result, error=None, error_code=None)
    loaded2 = store.get("abc-123")
    assert loaded2.status == "succeeded"
    assert loaded2.result is not None
    assert loaded2.result.circumference == 560
    assert loaded2.result.warnings == ["heuristic"]
    assert loaded2.updated_at != record.updated_at


def test_sqlite_survives_reopen(tmp_path):
    db = tmp_path / "persist.sqlite3"
    store1 = SqliteJobStore(db)
    result = _result(scale_mode="card", scale_frames_used=4)
    store1.put(
        JobRecord(
            job_id="persist-1",
            status="succeeded",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            filename="a.mp4",
            clockwise=False,
            scale_mode="card",
            result=result,
        )
    )
    store1.close()

    store2 = SqliteJobStore(db)
    loaded = store2.get("persist-1")
    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.result is not None
    assert loaded.result.scale_mode == "card"
    assert loaded.result.scale_frames_used == 4
    store2.close()
    clear_jobs_for_tests()


def test_sqlite_marks_orphaned_on_open(tmp_path):
    db = tmp_path / "orphan.sqlite3"
    store1 = SqliteJobStore(db)
    store1.put(
        JobRecord(
            job_id="orphan-1",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            filename="a.mp4",
        )
    )
    store1.close()

    store2 = SqliteJobStore(db)
    loaded = store2.get("orphan-1")
    assert loaded is not None
    assert loaded.status == "failed"
    assert "restart" in (loaded.error or "").lower()
    store2.close()
    clear_jobs_for_tests()


def test_sqlite_job_api_end_to_end(sqlite_store):
    _store, _db = sqlite_store
    client = TestClient(app)
    mock = _result()
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.jobs.measure_upload", return_value=mock):
        resp = client.post(
            "/v1/measure/jobs",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        final = None
        for _ in range(50):
            poll = client.get(f"/v1/measure/jobs/{job_id}")
            final = poll.json()
            if final["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

    assert final["status"] == "succeeded"
    assert final["result"]["measurements_mm"]["circumference"] == 560
    # Direct store read still has result after API success
    record = get_job(job_id)
    assert record is not None
    assert record.status == "succeeded"
    assert record.result is not None
