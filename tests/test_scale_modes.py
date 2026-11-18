"""Unit tests for scale helpers (no magstripe / TF required)."""

from __future__ import annotations

import pytest

from src.scale_modes import (
    ScaleError,
    as_pixel_mm,
    from_mm_per_pixel,
    from_reference,
)


def test_as_pixel_mm():
    assert as_pixel_mm(0.25) == [0.25]


def test_as_pixel_mm_rejects_non_positive():
    with pytest.raises(ScaleError):
        as_pixel_mm(0)
    with pytest.raises(ScaleError):
        as_pixel_mm(-1)


def test_from_mm_per_pixel():
    assert from_mm_per_pixel(0.1)[0] == pytest.approx(0.1)


def test_from_reference():
    # 85.6 mm over 428 px → 0.2 mm/px
    ppmm = from_reference(85.6, 428)
    assert ppmm[0] == pytest.approx(0.2)


def test_from_reference_rejects_bad():
    with pytest.raises(ScaleError):
        from_reference(0, 100)
    with pytest.raises(ScaleError):
        from_reference(85.6, 0)
