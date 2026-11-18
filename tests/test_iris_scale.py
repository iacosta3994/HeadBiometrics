"""Unit tests for iris-diameter scale mode (mocked landmarks; no TF)."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.scale_modes import (
    DEFAULT_IRIS_MM,
    IRIS_EYE_WIDTH_FRACTION,
    ScaleError,
    estimate_iris_px,
    from_iris,
)


def _fake_marks():
    """68-ish landmark array with usable eye points (indices 36–46)."""
    marks = np.zeros((68, 2), dtype=np.float32)
    # Left eye: outer 36, upper 37/38, inner 39, lower 40/41
    marks[36] = [10, 20]
    marks[37] = [15, 10]  # upper
    marks[38] = [20, 10]
    marks[39] = [30, 20]  # inner — eye width 20
    marks[40] = [15, 30]  # lower → aperture ||37-40|| = 20
    marks[41] = [20, 30]
    # Right eye: inner 42, upper 43/44, outer 45, lower 46/47
    marks[42] = [50, 20]
    marks[43] = [55, 12]  # upper
    marks[44] = [60, 12]
    marks[45] = [70, 20]  # outer — eye width 20
    marks[46] = [55, 28]  # lower → aperture ||43-46|| = 16
    marks[47] = [60, 28]
    return marks


def _install_fake_face_modules(marks):
    """Stub Proctoring_AI face modules so estimate_iris_px needs no TensorFlow."""
    det = types.ModuleType("src.Proctoring_AI.face_detector")
    det.get_face_detector = MagicMock(return_value="face-model")
    det.find_faces = MagicMock(return_value=[[0, 0, 80, 80]])

    lm = types.ModuleType("src.Proctoring_AI.face_landmarks")
    lm.get_landmark_model = MagicMock(return_value="lm-model")
    lm.detect_marks = MagicMock(return_value=marks)

    # Ensure parent packages exist as modules
    pkg = sys.modules.get("src.Proctoring_AI")
    if pkg is None:
        pkg = types.ModuleType("src.Proctoring_AI")
        pkg.__path__ = []  # mark as package
        sys.modules["src.Proctoring_AI"] = pkg

    return {
        "src.Proctoring_AI.face_detector": det,
        "src.Proctoring_AI.face_landmarks": lm,
    }


def test_estimate_iris_px_uses_eyelid_aperture():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    marks = _fake_marks()
    with patch.dict(sys.modules, _install_fake_face_modules(marks)):
        px = estimate_iris_px(img)
    # mean of left aperture 20 and right aperture 16
    assert px == pytest.approx(18.0)


def test_estimate_iris_px_falls_back_to_eye_width_fraction():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    marks = _fake_marks()
    # Collapse eyelid apertures
    marks[37] = marks[40].copy()
    marks[43] = marks[46].copy()
    with patch.dict(sys.modules, _install_fake_face_modules(marks)):
        px = estimate_iris_px(img)
    # mean eye width 20 * fraction
    assert px == pytest.approx(20.0 * IRIS_EYE_WIDTH_FRACTION)


def test_estimate_iris_px_no_face():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mods = _install_fake_face_modules(_fake_marks())
    mods["src.Proctoring_AI.face_detector"].find_faces = MagicMock(return_value=[])
    with patch.dict(sys.modules, mods):
        with pytest.raises(ScaleError, match="No face"):
            estimate_iris_px(img)


def test_from_iris_returns_pixel_mm_and_note():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch("src.scale_modes.best_iris_px", return_value=39.0):
        pixel_mm, note = from_iris([img], iris_mm=11.7)
    assert pixel_mm[0] == pytest.approx(11.7 / 39.0)
    assert "iris_mm=11.7" in note
    assert "approximate" in note.lower() or "Approximate" in note
    assert "37" in note and "40" in note


def test_from_iris_rejects_bad_prior():
    with pytest.raises(ScaleError, match="iris_mm"):
        from_iris([np.zeros((10, 10, 3), dtype=np.uint8)], iris_mm=0)


def test_default_iris_mm():
    assert DEFAULT_IRIS_MM == pytest.approx(11.7)
