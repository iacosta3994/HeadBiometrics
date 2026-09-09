"""GET /metrics and measure outcome counters."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from head_biometrics.app import app
from head_biometrics import metrics as metrics_mod
from head_biometrics.jobs import clear_jobs_for_tests
from head_biometrics.pipeline import DetectionError, MeasurementResult


@pytest.fixture
def client():
    metrics_mod.reset_metrics_for_tests()
    clear_jobs_for_tests()
    yield TestClient(app)
    clear_jobs_for_tests()
    metrics_mod.reset_metrics_for_tests()


def test_metrics_endpoint_shape(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("requests_total", "measure_success", "measure_failure", "jobs_created"):
        assert key in body
        assert isinstance(body[key], int)


def test_metrics_increment_on_measure(client):
    before = client.get("/metrics").json()
    mock = MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        scale_mode="mm_per_pixel",
        mm_per_pixel=0.25,
        confidence=0.9,
        warnings=[],
    )
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    with patch("head_biometrics.app.measure_upload", return_value=mock):
        ok = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
        )
    assert ok.status_code == 200

    with patch(
        "head_biometrics.app.measure_upload",
        side_effect=DetectionError("nope"),
    ):
        bad = client.post(
            "/v1/measure",
            files={"video": ("clip.mp4", BytesIO(b"x"), "video/mp4")},
            data={"scale_mode": "magstripe"},
        )
    assert bad.status_code == 422

    with patch("head_biometrics.jobs.measure_upload", return_value=mock):
        job = client.post(
            "/v1/measure/jobs",
            files={"video": ("clip.mp4", BytesIO(b"x"), "video/mp4")},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
        )
    assert job.status_code == 200

    after = client.get("/metrics").json()
    assert after["measure_success"] == before["measure_success"] + 1
    assert after["measure_failure"] == before["measure_failure"] + 1
    assert after["jobs_created"] == before["jobs_created"] + 1
    assert after["requests_total"] > before["requests_total"]
