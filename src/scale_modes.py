"""Alternate pixel→mm scale resolution (magstripe optional).

Legacy ``video_to_pixel_mm`` returns a nested sequence where index ``[0]`` is
mm-per-pixel. All helpers here return the same shape: ``[mm_per_pixel]``.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_IPD_MM = 63.0
SCALE_MODES = frozenset({"magstripe", "mm_per_pixel", "reference_mm", "ipd"})


class ScaleError(Exception):
    """Raised when scale cannot be resolved from the chosen mode/inputs."""


def as_pixel_mm(mm_per_pixel: float) -> list:
    """Wrap a scalar mm/pixel into the legacy ``pixel_mm[0]`` contract."""
    if mm_per_pixel is None or float(mm_per_pixel) <= 0:
        raise ScaleError("mm_per_pixel must be a positive float.")
    return [float(mm_per_pixel)]


def from_mm_per_pixel(mm_per_pixel: float) -> list:
    return as_pixel_mm(mm_per_pixel)


def from_reference(reference_width_mm: float, reference_width_px: float) -> list:
    if reference_width_mm is None or float(reference_width_mm) <= 0:
        raise ScaleError("reference_width_mm must be a positive float.")
    if reference_width_px is None or float(reference_width_px) <= 0:
        raise ScaleError("reference_width_px must be a positive float.")
    return as_pixel_mm(float(reference_width_mm) / float(reference_width_px))


def estimate_ipd_px(img) -> float:
    """Interpupillary distance in pixels from eye-corner landmarks on ``img``."""
    try:
        from src.Proctoring_AI.face_detector import find_faces, get_face_detector
        from src.Proctoring_AI.face_landmarks import detect_marks, get_landmark_model
    except Exception as exc:  # noqa: BLE001
        raise ScaleError(f"Face/landmark models unavailable for IPD scale: {exc}") from exc

    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    faces = find_faces(img, face_model)
    if not faces:
        raise ScaleError("No face detected for IPD scale.")
    face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    marks = detect_marks(img, landmark_model, face)
    if marks is None or len(marks) < 46:
        raise ScaleError("Facial landmarks incomplete for IPD scale.")

    # Approximate pupil centers as midpoints of each eye's outer/inner corners.
    left = (marks[36].astype(float) + marks[39].astype(float)) / 2.0
    right = (marks[42].astype(float) + marks[45].astype(float)) / 2.0
    ipd_px = float(np.linalg.norm(left - right))
    if ipd_px <= 1.0:
        raise ScaleError(f"Degenerate IPD pixel distance: {ipd_px}")
    return ipd_px


def best_ipd_px(img_array: Sequence) -> float:
    """Max IPD across frames (most frontal ≈ largest projected IPD)."""
    best: Optional[float] = None
    last_err: Optional[Exception] = None
    for img in img_array:
        try:
            px = estimate_ipd_px(img)
            if best is None or px > best:
                best = px
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if best is None:
        detail = f" Last error: {last_err}" if last_err else ""
        raise ScaleError(
            "Could not estimate IPD from any video frame." + detail
        )
    return best


def from_ipd(
    img_array: Sequence,
    ipd_mm: float = DEFAULT_IPD_MM,
) -> Tuple[list, str]:
    """Return (pixel_mm, scale_note) using a population IPD prior in mm."""
    if ipd_mm is None or float(ipd_mm) <= 0:
        raise ScaleError("ipd_mm must be a positive float.")
    ipd_px = best_ipd_px(img_array)
    mm_per_pixel = float(ipd_mm) / float(ipd_px)
    note = (
        f"Scale from interpupillary distance prior (ipd_mm={float(ipd_mm)}, "
        f"ipd_px={ipd_px:.1f}). Adult mean IPD ≈ 63 mm is a population prior — "
        "approximate; less accurate than a physical reference (magstripe / "
        "known object width)."
    )
    return as_pixel_mm(mm_per_pixel), note


def from_magstripe(img_array: Sequence) -> list:
    from src.video_to_mm import video_to_pixel_mm

    try:
        pixel_mm = video_to_pixel_mm(img_array)
    except (ValueError, TypeError, IndexError) as exc:
        raise ScaleError(
            "Magstripe scale detection failed. Ensure a credit-card-style "
            f"magnetic stripe is visible, or use another scale_mode. Details: {exc}"
        ) from exc
    if pixel_mm is None or (hasattr(pixel_mm, "__len__") and len(pixel_mm) == 0):
        raise ScaleError(
            "Magstripe scale detection returned no usable pixel/mm estimate."
        )
    return pixel_mm
