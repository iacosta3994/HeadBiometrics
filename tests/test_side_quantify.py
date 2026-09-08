"""Unit tests for headless side point selection (no TF / no GUI)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.side_quantify import (
    SidePointError,
    auto_select_side_points,
    dis_in_points,
    side_mm_metrics_postmtrp,
)


def _synthetic_marks(facing_left: bool = True):
    """68-ish landmark array; only indices we use need realistic values."""
    marks = np.zeros((68, 2), dtype=np.uint32)
    # Face roughly in left half if facing left (nose left of box center).
    if facing_left:
        marks[30] = [40, 80]   # nose tip
        marks[21] = [50, 50]
        marks[22] = [60, 50]
        marks[27] = [55, 55]
    else:
        marks[30] = [160, 80]
        marks[21] = [140, 50]
        marks[22] = [150, 50]
        marks[27] = [145, 55]
    return marks


def _synthetic_contour(width=200, height=150):
    """Simple rectangular contour as Nx1x2 int array (OpenCV style)."""
    pts = []
    for x in range(10, width - 10):
        pts.append([x, 20])
    for y in range(20, height - 10):
        pts.append([width - 11, y])
    for x in range(width - 11, 9, -1):
        pts.append([x, height - 11])
    for y in range(height - 11, 19, -1):
        pts.append([10, y])
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def test_dis_in_points():
    assert dis_in_points((0, 0), (3, 4)) == 5


def test_auto_select_side_points_facing_left():
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    face = [30, 40, 90, 120]  # center x = 60; nose at 40 → facing left
    marks = _synthetic_marks(facing_left=True)
    contour = _synthetic_contour()

    lf, lb, ff, nape = auto_select_side_points(
        img, face=face, marks=marks, contour=contour
    )
    assert lf[0] == 55  # mid of 21/22
    # Back should be toward +x (right side of silhouette)
    assert lb[0] > lf[0]
    assert ff == (40, 80)
    assert nape[0] > ff[0]


def test_auto_select_side_points_facing_right():
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    face = [110, 40, 170, 120]  # center 140; nose 160 → facing right
    marks = _synthetic_marks(facing_left=False)
    contour = _synthetic_contour()

    lf, lb, ff, nape = auto_select_side_points(
        img, face=face, marks=marks, contour=contour
    )
    assert lb[0] < lf[0]
    assert nape[0] < ff[0]


def test_side_mm_metrics_headless_does_not_call_imshow():
    """Default path must not open a GUI window."""
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    face = [30, 40, 90, 120]
    marks = _synthetic_marks(facing_left=True)
    contour = _synthetic_contour()

    with patch("src.side_quantify.cv2.imshow") as imshow_mock, patch(
        "src.side_quantify.auto_select_side_points",
        return_value=((55, 50), (180, 60), (40, 80), (170, 100)),
    ), patch(
        "src.side_quantify.img_head_contour_side",
        return_value=400.0,
    ):
        front2nape, length = side_mm_metrics_postmtrp(
            img, pixel_mm=[0.5], interactive=False
        )

    imshow_mock.assert_not_called()
    # length ≈ dist((55,50),(180,60)) * 0.5
    assert length == int(round(np.sqrt((55 - 180) ** 2 + (50 - 60) ** 2)) * 0.5)
    # front2nape = int(400 * 0.5) then * 0.5 → 100
    assert front2nape == int(int(400.0 * 0.5) * 0.5)


def test_side_mm_metrics_propagates_side_point_error():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch(
        "src.side_quantify.auto_select_side_points",
        side_effect=SidePointError("No face detected on side (wide) frame."),
    ):
        with pytest.raises(SidePointError, match="No face"):
            side_mm_metrics_postmtrp(img, [0.2], interactive=False)
