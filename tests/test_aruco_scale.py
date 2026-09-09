"""Unit tests for ArUco marker scale detection (synthetic markers only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.aruco_scale import (
    DEFAULT_ARUCO_DICT,
    aruco_mm_per_pixel_from_frames,
    detect_aruco_in_frame,
    resolve_dictionary_id,
)
from src.scale_modes import ScaleError, from_aruco, normalize_scale_mode


def _synthetic_aruco_frame(
    marker_id: int = 0,
    side_px: int = 200,
    dict_name: str = "4x4_50",
    canvas: int = 400,
    pad: int = 80,
) -> np.ndarray:
    """BGR image with a generated ArUco marker on a white pad."""
    import cv2
    import cv2.aruco as aruco

    _name, dict_id = resolve_dictionary_id(dict_name)
    dictionary = aruco.getPredefinedDictionary(dict_id)
    marker = aruco.generateImageMarker(dictionary, marker_id, side_px)
    gray = np.full((canvas, canvas), 255, dtype=np.uint8)
    x0, y0 = pad, pad
    gray[y0 : y0 + side_px, x0 : x0 + side_px] = marker
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_normalize_aruco_mode():
    assert normalize_scale_mode("aruco") == "aruco"
    assert normalize_scale_mode("ARUCO") == "aruco"


def test_resolve_dictionary_defaults():
    name, dict_id = resolve_dictionary_id(None)
    assert name == "DICT_4X4_50"
    assert isinstance(dict_id, int)  # DICT_4X4_50 == 0 in OpenCV
    name2, _ = resolve_dictionary_id("5x5_100")
    assert name2 == "DICT_5X5_100"


def test_detect_synthetic_marker_resolves_scale():
    side_px = 200
    length_mm = 40.0
    img = _synthetic_aruco_frame(side_px=side_px)
    _name, dict_id = resolve_dictionary_id(DEFAULT_ARUCO_DICT)
    hits = detect_aruco_in_frame(img, dict_id)
    assert hits, "expected to detect synthetic ArUco marker"
    assert hits[0]["side_px"] == pytest.approx(side_px, rel=0.05)

    mm, n, dname = aruco_mm_per_pixel_from_frames(
        [img, img], marker_length_mm=length_mm, aruco_dict="4x4_50"
    )
    assert n >= 2
    assert dname == "DICT_4X4_50"
    expected = length_mm / side_px
    assert mm == pytest.approx(expected, rel=0.05)


def test_from_aruco_across_frames():
    imgs = [_synthetic_aruco_frame(side_px=180) for _ in range(3)]
    pixel_mm, note, n = from_aruco(imgs, marker_length_mm=36.0, aruco_dict="4x4_50")
    assert n >= 1
    assert pixel_mm[0] == pytest.approx(36.0 / 180.0, rel=0.08)
    assert "ArUco" in note


def test_from_aruco_5x5_dict():
    img = _synthetic_aruco_frame(marker_id=1, side_px=160, dict_name="5x5_100")
    pixel_mm, note, n = from_aruco([img], marker_length_mm=32.0, aruco_dict="5x5_100")
    assert n >= 1
    assert pixel_mm[0] == pytest.approx(32.0 / 160.0, rel=0.08)
    assert "DICT_5X5_100" in note


def test_from_aruco_no_marker_clear_error():
    blank = [np.full((240, 320, 3), 200, dtype=np.uint8)]
    with pytest.raises(ScaleError, match="ArUco|marker"):
        from_aruco(blank, marker_length_mm=40.0)


def test_from_aruco_requires_length():
    imgs = [_synthetic_aruco_frame()]
    with pytest.raises(ScaleError, match="aruco_marker_length_mm"):
        from_aruco(imgs, marker_length_mm=0)
