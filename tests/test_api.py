"""Lightweight API tests with the CV pipeline mocked (no TF/OpenCV required)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from head_biometrics.app import app
from head_biometrics.pipeline import (
    DetectionError,
    MeasurementResult,
    ScaleOptions,
    UploadTooLargeError,
    compute_quality_meta,
    validate_scale_options,
)


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
    assert "version" in body
    assert body["version"]


def test_version_endpoint(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert "HeadBiometrics" in body["name"]


def test_scale_modes_catalog(client):
    resp = client.get("/v1/scale-modes")
    assert resp.status_code == 200
    body = resp.json()
    ids = {m["id"] for m in body["modes"]}
    assert {"magstripe", "card", "aruco", "mm_per_pixel", "reference_mm", "ipd"} <= ids
    card = next(m for m in body["modes"] if m["id"] == "card")
    assert "id1_card" in card["aliases"]
    assert card["required_fields"] == []
    mm = next(m for m in body["modes"] if m["id"] == "mm_per_pixel")
    assert "mm_per_pixel" in mm["required_fields"]
    aruco = next(m for m in body["modes"] if m["id"] == "aruco")
    assert "aruco_marker_length_mm" in aruco["required_fields"]
    assert "aruco_dict" in aruco["optional_fields"]


def test_measure_missing_file(client):
    resp = client.post("/v1/measure")
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
        mm_per_pixel=0.21,
        confidence=0.7,
        warnings=["heuristic"],
        scale_frames_used=None,
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
    assert body["meta"]["mm_per_pixel"] == pytest.approx(0.21)
    assert body["meta"]["confidence"] == pytest.approx(0.7)
    assert body["meta"]["warnings"] == ["heuristic"]
    mocked.assert_called_once()
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
        mm_per_pixel=0.25,
        confidence=0.9,
        warnings=[],
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
    assert resp.json()["meta"]["mm_per_pixel"] == pytest.approx(0.25)
    opts = mocked.call_args.kwargs["scale_options"]
    assert opts.scale_mode == "mm_per_pixel"
    assert opts.mm_per_pixel == pytest.approx(0.25)


def test_measure_forwards_card_scale(client):
    mock_result = MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        scale_mode="card",
        scale_note="Scale from ISO ID-1 card auto-detect",
        mm_per_pixel=0.214,
        confidence=0.92,
        warnings=[],
        scale_frames_used=12,
    )
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.app.measure_upload", return_value=mock_result) as mocked:
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "card"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["scale_mode"] == "card"
    assert "ISO ID-1" in body["meta"]["scale_note"]
    assert body["meta"]["mm_per_pixel"] == pytest.approx(0.214)
    assert body["meta"]["scale_frames_used"] == 12
    opts = mocked.call_args.kwargs["scale_options"]
    assert opts.scale_mode == "card"


def test_measure_forwards_aruco_scale(client):
    mock_result = MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=False,
        scale_mode="aruco",
        scale_note="Scale from ArUco marker",
        mm_per_pixel=0.2,
        confidence=0.72,
        warnings=["Scale derived from only 3 frame detections — moderate support."],
        scale_frames_used=3,
    )
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch("head_biometrics.app.measure_upload", return_value=mock_result) as mocked:
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={
                "scale_mode": "aruco",
                "aruco_marker_length_mm": "40",
                "aruco_dict": "4x4_50",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["meta"]["scale_mode"] == "aruco"
    assert "ArUco" in body["meta"]["scale_note"]
    assert body["meta"]["scale_frames_used"] == 3
    assert body["meta"]["confidence"] == pytest.approx(0.72)
    opts = mocked.call_args.kwargs["scale_options"]
    assert opts.scale_mode == "aruco"
    assert opts.aruco_marker_length_mm == pytest.approx(40.0)
    assert opts.aruco_dict == "4x4_50"


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
        confidence=0.4,
        warnings=["IPD scale uses a population prior"],
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


def test_measure_upload_too_large_413(client):
    fake_video = ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")

    with patch(
        "head_biometrics.app.measure_upload",
        side_effect=UploadTooLargeError(
            "Upload exceeds maximum size of 104857600 bytes (100 MB)."
        ),
    ):
        resp = client.post(
            "/v1/measure",
            files={"video": fake_video},
            data={"scale_mode": "mm_per_pixel", "mm_per_pixel": "0.2"},
        )

    assert resp.status_code == 413
    assert "100" in resp.json()["detail"]


def test_validate_scale_options_mm_per_pixel_missing():
    with pytest.raises(DetectionError, match="mm_per_pixel"):
        validate_scale_options(ScaleOptions(scale_mode="mm_per_pixel"))


def test_validate_scale_options_reference_missing():
    with pytest.raises(DetectionError, match="reference_width"):
        validate_scale_options(
            ScaleOptions(scale_mode="reference_mm", reference_width_mm=85.6)
        )


def test_validate_scale_options_aruco_missing_length():
    with pytest.raises(DetectionError, match="aruco_marker_length_mm"):
        validate_scale_options(ScaleOptions(scale_mode="aruco"))


def test_validate_scale_options_invalid_mode():
    with pytest.raises(DetectionError, match="Invalid scale_mode"):
        validate_scale_options(ScaleOptions(scale_mode="banana"))


def test_validate_scale_options_ok_modes():
    validate_scale_options(ScaleOptions(scale_mode="magstripe"))
    validate_scale_options(ScaleOptions(scale_mode="card"))
    validate_scale_options(ScaleOptions(scale_mode="id1_card"))
    validate_scale_options(ScaleOptions(scale_mode="mm_per_pixel", mm_per_pixel=0.2))
    validate_scale_options(
        ScaleOptions(
            scale_mode="reference_mm",
            reference_width_mm=85.6,
            reference_width_px=400,
        )
    )
    validate_scale_options(ScaleOptions(scale_mode="ipd", ipd_mm=63))
    validate_scale_options(
        ScaleOptions(
            scale_mode="aruco",
            aruco_marker_length_mm=40.0,
            aruco_dict="4x4_50",
        )
    )


def test_compute_quality_meta_ipd_caps_confidence():
    conf, warns = compute_quality_meta(scale_mode="ipd")
    assert conf <= 0.45
    assert any("IPD" in w or "population" in w for w in warns)
    assert any("heuristic" in w.lower() or "medical" in w.lower() for w in warns)


def test_compute_quality_meta_few_frames():
    conf, warns = compute_quality_meta(scale_mode="aruco", scale_frames_used=1)
    assert conf <= 0.55
    assert any("1 frame" in w for w in warns)
