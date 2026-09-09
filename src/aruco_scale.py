"""ArUco marker scale detection for pixel→mm conversion.

Detects printed ArUco markers (OpenCV ``cv2.aruco``) and derives::

    mm_per_pixel = aruco_marker_length_mm / median_side_length_px

across detections, then median-aggregates over frames.

Requires ``cv2.aruco`` (bundled with recent ``opencv-python-headless`` and always
with ``opencv-contrib-python-headless``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Friendly name → OpenCV DICT_* constant name
ARUCO_DICT_ALIASES = {
    "4x4_50": "DICT_4X4_50",
    "dict_4x4_50": "DICT_4X4_50",
    "4x4_100": "DICT_4X4_100",
    "5x5_50": "DICT_5X5_50",
    "5x5_100": "DICT_5X5_100",
    "dict_5x5_100": "DICT_5X5_100",
    "6x6_50": "DICT_6X6_50",
    "6x6_250": "DICT_6X6_250",
    "7x7_50": "DICT_7X7_50",
}

DEFAULT_ARUCO_DICT = "4x4_50"


class ArucoUnavailableError(ImportError):
    """Raised when cv2.aruco cannot be imported."""


def _import_aruco():
    """Return the ``cv2.aruco`` module or raise ``ArucoUnavailableError``."""
    try:
        import cv2.aruco as aruco  # type: ignore

        return aruco
    except Exception as exc:  # noqa: BLE001
        raise ArucoUnavailableError(
            "cv2.aruco is unavailable. Install opencv-contrib-python-headless "
            "(or a recent opencv-python-headless build that bundles aruco). "
            f"Details: {exc}"
        ) from exc


def resolve_dictionary_id(name: Optional[str]) -> Tuple[str, int]:
    """Map a friendly dict name to ``(canonical_name, opencv_dict_id)``."""
    aruco = _import_aruco()
    raw = (name or DEFAULT_ARUCO_DICT).strip().lower().replace("-", "_")
    const_name = ARUCO_DICT_ALIASES.get(raw)
    if const_name is None:
        # Allow passing DICT_4X4_50 directly
        upper = raw.upper()
        if not upper.startswith("DICT_"):
            upper = "DICT_" + upper
        const_name = upper
    if not hasattr(aruco, const_name):
        known = ", ".join(sorted(ARUCO_DICT_ALIASES.keys()))
        raise ValueError(
            f"Unknown aruco_dict={name!r}. Try one of: {known} "
            f"(or OpenCV names like DICT_4X4_50)."
        )
    return const_name, int(getattr(aruco, const_name))


def _mean_side_px(corners: np.ndarray) -> float:
    """Mean side length in pixels from a (1,4,2) or (4,2) corner array."""
    pts = np.asarray(corners, dtype=float).reshape(-1, 2)
    if len(pts) != 4:
        return 0.0
    sides = [
        float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)
    ]
    return float(np.mean(sides))


def detect_aruco_in_frame(
    img: np.ndarray,
    dictionary_id: int,
) -> List[Dict[str, Any]]:
    """Detect ArUco markers; return list of {id, side_px, corners}."""
    aruco = _import_aruco()
    if img is None:
        return []
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    params = aruco.DetectorParameters()
    hits: List[Dict[str, Any]] = []

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _rejected = detector.detectMarkers(gray)
    else:
        # Legacy OpenCV < 4.7 path
        corners, ids, _rejected = aruco.detectMarkers(
            gray, dictionary, parameters=params
        )

    if ids is None or len(ids) == 0:
        return hits
    for i, marker_id in enumerate(ids.flatten()):
        side = _mean_side_px(corners[i])
        if side <= 1.0:
            continue
        hits.append(
            {
                "id": int(marker_id),
                "side_px": side,
                "corners": corners[i],
            }
        )
    return hits


def collect_aruco_scales(
    img_array: Sequence,
    marker_length_mm: float,
    dictionary_id: int,
) -> List[float]:
    """mm_per_pixel estimates from every marker detection across frames."""
    if marker_length_mm is None or float(marker_length_mm) <= 0:
        raise ValueError("aruco_marker_length_mm must be a positive float")
    length = float(marker_length_mm)
    scales: List[float] = []
    for img in img_array:
        try:
            hits = detect_aruco_in_frame(img, dictionary_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("aruco detect frame error: %s", exc)
            continue
        for hit in hits:
            scales.append(length / float(hit["side_px"]))
    return scales


def aggregate_mm_per_pixel(scales: Sequence[float]) -> float:
    """Median aggregate with light std filtering (same idea as card_scale)."""
    if not scales:
        raise ValueError("empty scales")
    arr = np.asarray(scales, dtype=float)
    if len(arr) >= 5:
        med = float(np.median(arr))
        std = float(np.std(arr))
        if std > 0:
            keep = arr[np.abs(arr - med) <= 2.5 * std]
            if len(keep) >= 3:
                arr = keep
    return float(np.median(arr))


def aruco_mm_per_pixel_from_frames(
    img_array: Sequence,
    marker_length_mm: float,
    aruco_dict: Optional[str] = None,
) -> Tuple[float, int, str]:
    """Detect markers across frames.

    Returns ``(mm_per_pixel, n_detections, dict_canonical_name)``.
    """
    const_name, dict_id = resolve_dictionary_id(aruco_dict)
    scales = collect_aruco_scales(img_array, marker_length_mm, dict_id)
    if not scales:
        raise LookupError(
            f"no ArUco markers detected (dict={const_name}). "
            "Ensure a printed marker is fully visible and "
            "aruco_marker_length_mm matches the physical side length."
        )
    return aggregate_mm_per_pixel(scales), len(scales), const_name
