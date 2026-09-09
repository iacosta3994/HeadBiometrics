"""Unit tests for ISO ID-1 card scale detection (synthetic images only)."""

from __future__ import annotations

import numpy as np
import pytest

from src.card_scale import (
    ID1_ASPECT,
    ID1_LONG_MM,
    aggregate_mm_per_pixel,
    detect_card_in_frame,
    card_mm_per_pixel_from_frames,
)
from src.scale_modes import ScaleError, from_card, normalize_scale_mode


def _draw_id1_card(
    canvas_h: int = 480,
    canvas_w: int = 640,
    long_px: int = 320,
    *,
    offset=(80, 100),
    fill=220,
    bg=30,
) -> np.ndarray:
    """BGR image with a solid rectangle of ISO ID-1 aspect ratio."""
    short_px = int(round(long_px / ID1_ASPECT))
    img = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    x0, y0 = offset
    x1, y1 = x0 + long_px, y0 + short_px
    assert x1 < canvas_w and y1 < canvas_h
    img[y0:y1, x0:x1] = (fill, fill, fill)
    # Dark border helps Canny / threshold find the quad edges.
    img[y0:y1, x0 : x0 + 2] = 0
    img[y0:y1, x1 - 2 : x1] = 0
    img[y0 : y0 + 2, x0:x1] = 0
    img[y1 - 2 : y1, x0:x1] = 0
    return img


def test_normalize_id1_alias():
    assert normalize_scale_mode("id1_card") == "card"
    assert normalize_scale_mode("CARD") == "card"
    assert normalize_scale_mode("magstripe") == "magstripe"


def test_detect_card_synthetic_rectangle():
    long_px = 400
    img = _draw_id1_card(long_px=long_px, offset=(50, 60))
    hit = detect_card_in_frame(img)
    assert hit is not None, "expected to detect synthetic ID-1 rectangle"
    expected = ID1_LONG_MM / long_px
    # Allow generous tolerance: minAreaRect / contour approx may shrink a few px.
    assert hit["mm_per_pixel"] == pytest.approx(expected, rel=0.08)
    assert hit["width_px"] == pytest.approx(long_px, rel=0.08)
    assert hit["aspect"] == pytest.approx(ID1_ASPECT, rel=0.1)


def test_detect_card_rejects_wrong_aspect():
    img = np.full((400, 400, 3), 30, dtype=np.uint8)
    # Square — not ID-1
    img[50:250, 50:250] = 220
    hit = detect_card_in_frame(img)
    assert hit is None


def test_aggregate_median():
    assert aggregate_mm_per_pixel([0.2, 0.21, 0.19, 0.2]) == pytest.approx(0.2)


def test_from_card_across_frames():
    imgs = [
        _draw_id1_card(long_px=360, offset=(40, 50)),
        _draw_id1_card(long_px=360, offset=(45, 55)),
        _draw_id1_card(long_px=362, offset=(42, 52)),
    ]
    pixel_mm, note = from_card(imgs)
    assert pixel_mm[0] == pytest.approx(ID1_LONG_MM / 360.0, rel=0.1)
    assert "ISO ID-1" in note
    assert "longest-side" in note


def test_from_card_failure_clear_message():
    blank = [np.zeros((240, 320, 3), dtype=np.uint8)]
    with pytest.raises(ScaleError, match="ISO ID-1|credit/ID card"):
        from_card(blank)


def test_card_mm_per_pixel_from_frames_count():
    imgs = [_draw_id1_card(long_px=300, offset=(30, 40)) for _ in range(3)]
    mm, n = card_mm_per_pixel_from_frames(imgs)
    assert n >= 1
    assert mm == pytest.approx(ID1_LONG_MM / 300.0, rel=0.1)
