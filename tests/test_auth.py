"""Optional API_KEY auth tests."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from head_biometrics.app import app
from head_biometrics.jobs import clear_jobs_for_tests
from head_biometrics.pipeline import MeasurementResult


@pytest.fixture
def client():
    clear_jobs_for_tests()
    yield TestClient(app)
    clear_jobs_for_tests()


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret-key")
    return "test-secret-key"


def _mock_result():
    return MeasurementResult(
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


def test_health_and_version_open_with_api_key(client, api_key):
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/v1/scale-modes").status_code == 200


def test_measure_401_without_key(client, api_key):
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    resp = client.post(
        "/v1/measure",
        files={"video": fake_video},
        data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
    )
    assert resp.status_code == 401
    assert "API key" in resp.json()["detail"]


def test_measure_401_wrong_key(client, api_key):
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    resp = client.post(
        "/v1/measure",
        files={"video": fake_video},
        data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_measure_ok_with_x_api_key(client, api_key):
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    with patch("head_biometrics.app.measure_upload", return_value=_mock_result()):
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
            headers={"X-API-Key": api_key},
        )
    assert resp.status_code == 200
    assert resp.json()["measurements_mm"]["circumference"] == 560


def test_measure_ok_with_bearer(client, api_key):
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    with patch("head_biometrics.app.measure_upload", return_value=_mock_result()):
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
    assert resp.status_code == 200


def test_jobs_require_key(client, api_key):
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    resp = client.post(
        "/v1/measure/jobs",
        files={"video": fake_video},
        data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
    )
    assert resp.status_code == 401
    assert client.get("/v1/measure/jobs/anything").status_code == 401


def test_no_api_key_env_leaves_open(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    fake_video = ("clip.mp4", BytesIO(b"x"), "video/mp4")
    with patch("head_biometrics.app.measure_upload", return_value=_mock_result()):
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.25"},
        )
    assert resp.status_code == 200
