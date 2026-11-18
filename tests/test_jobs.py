"""Async measurement job API tests (mocked pipeline)."""

from __future__ import annotations

import time
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from head_biometrics.app import app
from head_biometrics.jobs import clear_jobs_for_tests
from head_biometrics.pipeline import DetectionError, MeasurementResult


@pytest.fixture
def client():
    clear_jobs_for_tests()
    yield TestClient(app)
    clear_jobs_for_tests()


def _mock_result(**kwargs):
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
        warnings=[],
        scale_frames_used=None,
    )
    base.update(kwargs)
    return MeasurementResult(**base)


def test_job_create_and_succeed(client):
    mock = _mock_result()
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.jobs.measure_upload", return_value=mock):
        resp = client.post(
            "/v1/measure/jobs",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"  # create always reports queued
        assert body["job_id"]

        job_id = body["job_id"]
        # Poll until terminal (thread pool should finish quickly with mock)
        final = None
        for _ in range(50):
            poll = client.get(f"/v1/measure/jobs/{job_id}")
            assert poll.status_code == 200
            final = poll.json()
            if final["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

    assert final is not None
    assert final["status"] == "succeeded"
    assert final["result"]["measurements_mm"]["circumference"] == 560
    assert final["result"]["meta"]["scale_mode"] == "mm_per_pixel"
    assert final["error"] is None
    assert final["filename"] == "clip.mp4"


def test_job_failed_detection(client):
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch(
        "head_biometrics.jobs.measure_upload",
        side_effect=DetectionError("Magstripe scale detection failed."),
    ):
        resp = client.post(
            "/v1/measure/jobs",
            files={"video": fake_video},
            data={"scale_mode": "magstripe"},
        )
        job_id = resp.json()["job_id"]
        final = None
        for _ in range(50):
            poll = client.get(f"/v1/measure/jobs/{job_id}")
            final = poll.json()
            if final["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

    assert final["status"] == "failed"
    assert "Magstripe" in final["error"]
    assert final["result"] is None


def test_job_unknown_404(client):
    resp = client.get("/v1/measure/jobs/does-not-exist")
    assert resp.status_code == 404


def test_sync_measure_still_works(client):
    mock = _mock_result(scale_mode="magstripe", mm_per_pixel=0.2, confidence=0.7)
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")
    with patch("head_biometrics.app.measure_upload", return_value=mock):
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "magstripe"},
        )
    assert resp.status_code == 200
    assert resp.json()["measurements_mm"]["circumference"] == 560
