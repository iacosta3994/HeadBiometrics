"""Lightweight API tests with the CV pipeline mocked (no TF/OpenCV required)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from head_biometrics.app import app
from head_biometrics.pipeline import DetectionError, MeasurementResult, ScaleOptions, validate_scale_options


@pytest.fixture
def client():
    return TestClient(app)


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "pipeline_available" in body
    assert "demo_mode" in body


def test_measure_missing_file(client):
    resp = client.post("/v1/measure")
    # FastAPI validation error for missing required upload
    assert resp.status_code == 422


def test_measure_with_mocked_pipeline(client):
    mock_result = MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        note=None,
        scale_mode="magstripe",
        scale_note=None,
    )
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.app.measure_upload", return_value=mock_result) as mocked:
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"clockwise": "false", "scale_mode": "magstripe"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["measurements_mm"]["circumference"] == 560
    assert body["measurements_mm"]["front_to_nape"] == 340
    assert body["measurements_mm"]["ear_to_ear"] == 150
    assert body["measurements_mm"]["head_width"] == 155
    assert body["measurements_mm"]["length"] == 195
    assert body["meta"]["clockwise"] is False
    assert body["meta"]["filename"] == "clip.mp4"
    assert body["meta"]["demo_mode"] is False
    assert body["meta"]["scale_mode"] == "magstripe"
    mocked.assert_called_once()
    # Scale options should be forwarded
    call_kwargs = mocked.call_args.kwargs
    assert "scale_options" in call_kwargs
    assert call_kwargs["scale_options"].scale_mode == "magstripe"


def test_measure_detection_failure_returns_422(client):
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch(
        "head_biometrics.app.measure_upload",
        side_effect=DetectionError("Magstripe scale detection failed."),
    ):
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"clockwise": "true"},
        )

    assert resp.status_code == 422
    assert "Magstripe" in resp.json()["detail"]


def test_measure_empty_filename_400(client):
    fake_video = ("", BytesIO(b"x"), "video/mp4")
    resp = client.post("/v1/measure", files={"video": fake_video})
    # Starlette/FastAPI may treat empty filename as missing → 422, or our check → 400
    assert resp.status_code in (400, 422)


def test_measure_forwards_mm_per_pixel_scale(client):
    mock_result = MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        scale_mode="mm_per_pixel",
        scale_note=None,
    )
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.app.measure_upload", return_value=mock_result) as mocked:
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={
                "scale_mode": "mm_per_pixel",
                "mm_per_pixel": "0.25",
            },
        )

    assert resp.status_code == 200
    assert resp.json()["meta"]["scale_mode"] == "mm_per_pixel"
    opts = mocked.call_args.kwargs["scale_options"]
    assert opts.scale_mode == "mm_per_pixel"
    assert opts.mm_per_pixel == pytest.approx(0.25)


def test_measure_forwards_ipd_scale(client):
    mock_result = MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        scale_mode="ipd",
        scale_note="population prior",
    )
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.app.measure_upload", return_value=mock_result) as mocked:
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "ipd", "ipd_mm": "64"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["scale_mode"] == "ipd"
    assert body["meta"]["scale_note"] == "population prior"
    opts = mocked.call_args.kwargs["scale_options"]
    assert opts.ipd_mm == pytest.approx(64.0)


def test_validate_scale_options_mm_per_pixel_missing():
    with pytest.raises(DetectionError, match="mm_per_pixel"):
        validate_scale_options(ScaleOptions(scale_mode="mm_per_pixel"))


def test_validate_scale_options_reference_missing():
    with pytest.raises(DetectionError, match="reference_width"):
        validate_scale_options(
            ScaleOptions(scale_mode="reference_mm", reference_width_mm=85.6)
        )


def test_validate_scale_options_invalid_mode():
    with pytest.raises(DetectionError, match="Invalid scale_mode"):
        validate_scale_options(ScaleOptions(scale_mode="banana"))


def test_validate_scale_options_ok_modes():
    validate_scale_options(ScaleOptions(scale_mode="magstripe"))
    validate_scale_options(ScaleOptions(scale_mode="mm_per_pixel", mm_per_pixel=0.2))
    validate_scale_options(
        ScaleOptions(
            scale_mode="reference_mm",
            reference_width_mm=85.6,
            reference_width_px=400,
        )
    )
    validate_scale_options(ScaleOptions(scale_mode="ipd", ipd_mm=63))
