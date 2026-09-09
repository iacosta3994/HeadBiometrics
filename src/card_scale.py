"""ISO ID-1 credit/ID card auto-detect for pixel→mm scale.

Physical size (ISO/IEC 7810 ID-1): **85.60 × 53.98 mm** (aspect ≈ 1.5858).

Scale derivation (documented choice)
------------------------------------
Prefer the **longest side** of the detected quad / min-area rectangle::

    mm_per_pixel = 85.60 / max(width_px, height_px)

Using the long edge is more stable than the short edge under partial
perspective foreshortening, and avoids mixing diagonal conventions with the
legacy magstripe path (which uses diagonal ratios for stripe geometry).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ISO/IEC 7810 ID-1
ID1_LONG_MM = 85.60
ID1_SHORT_MM = 53.98
ID1_ASPECT = ID1_LONG_MM / ID1_SHORT_MM  # ≈ 1.58577

# Detection tolerances
ASPECT_TOLERANCE = 0.12  # ±12% relative to ID-1 aspect
MIN_AREA_FRACTION = 0.008  # ≥ 0.8% of frame
MAX_AREA_FRACTION = 0.55  # ≤ 55% of frame
MIN_RECTANGULARITY = 0.72  # contour_area / minAreaRect area
MIN_SIDE_PX = 40  # reject tiny noise quads


def _quad_side_lengths(approx: np.ndarray) -> List[float]:
    pts = approx.reshape(-1, 2).astype(float)
    sides: List[float] = []
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        sides.append(float(np.linalg.norm(a - b)))
    return sides


def _score_quad(
    contour: np.ndarray,
    approx: np.ndarray,
    frame_area: float,
) -> Optional[Dict[str, Any]]:
    """Score a 4-point approx as an ID-1 card candidate, or None if rejected."""
    if len(approx) != 4:
        return None

    rect = cv2.minAreaRect(contour)
    (_cx, _cy), (rw, rh), _angle = rect
    if rw <= 0 or rh <= 0:
        return None

    long_px = float(max(rw, rh))
    short_px = float(min(rw, rh))
    if long_px < MIN_SIDE_PX or short_px < 1.0:
        return None

    aspect = long_px / short_px
    aspect_err = abs(aspect - ID1_ASPECT) / ID1_ASPECT
    if aspect_err > ASPECT_TOLERANCE:
        return None

    box_area = long_px * short_px
    area_frac = box_area / float(frame_area) if frame_area > 0 else 0.0
    if area_frac < MIN_AREA_FRACTION or area_frac > MAX_AREA_FRACTION:
        return None

    contour_area = float(cv2.contourArea(contour))
    if box_area <= 0:
        return None
    rectangularity = contour_area / box_area
    if rectangularity < MIN_RECTANGULARITY:
        return None

    # Prefer aspect closeness + rectangularity + mid-sized cards (not tiny).
    score = (
        (1.0 - aspect_err) * 0.45
        + rectangularity * 0.35
        + min(area_frac / 0.15, 1.0) * 0.20
    )

    mm_per_pixel = ID1_LONG_MM / long_px
    return {
        "width_px": long_px,
        "height_px": short_px,
        "aspect": aspect,
        "area_frac": area_frac,
        "rectangularity": rectangularity,
        "score": score,
        "mm_per_pixel": mm_per_pixel,
        "approx": approx,
    }


def _binary_masks(gray: np.ndarray) -> List[np.ndarray]:
    """Build several edge/binary views so glare / low contrast still yield quads."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    masks: List[np.ndarray] = []

    # Canny (two sensitivities)
    for lo, hi in ((40, 120), (60, 180)):
        masks.append(cv2.Canny(blurred, lo, hi))

    # Adaptive threshold (both polarities)
    for method in (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C):
        for invert in (False, True):
            th = cv2.adaptiveThreshold(
                blurred,
                255,
                method,
                cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY,
                11,
                2,
            )
            masks.append(th)

    # Morphological close on the primary Canny to join broken card edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masks.append(cv2.morphologyEx(masks[0], cv2.MORPH_CLOSE, kernel))
    return masks


def detect_card_in_frame(img: np.ndarray) -> Optional[Dict[str, Any]]:
    """Find the best ISO ID-1 card quad in a BGR/gray frame.

    Returns a dict with ``mm_per_pixel``, ``width_px``, ``height_px``, ``score``,
    or ``None`` if no plausible card is found.
    """
    if img is None:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0:
        return None

    best: Optional[Dict[str, Any]] = None
    for mask in _binary_masks(gray):
        contours, _hier = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            if peri < (MIN_SIDE_PX * 2):
                continue
            # Try a few epsilons — cards are nearly rectangular but may be soft.
            for eps_frac in (0.02, 0.03, 0.04, 0.055):
                approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
                if len(approx) != 4:
                    continue
                if not cv2.isContourConvex(approx):
                    continue
                cand = _score_quad(contour, approx, frame_area)
                if cand is None:
                    continue
                if best is None or cand["score"] > best["score"]:
                    best = cand
    return best


def collect_card_scales(img_array: Sequence) -> List[float]:
    """Return mm_per_pixel estimates from every frame that yields a card."""
    scales: List[float] = []
    for img in img_array:
        try:
            hit = detect_card_in_frame(img)
        except Exception as exc:  # noqa: BLE001 — never crash the pipeline
            logger.debug("card detect frame error: %s", exc)
            continue
        if hit is not None and hit["mm_per_pixel"] > 0:
            scales.append(float(hit["mm_per_pixel"]))
    return scales


def aggregate_mm_per_pixel(scales: Sequence[float]) -> float:
    """Median aggregate (robust to outlier frames), with light std filtering."""
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


def card_mm_per_pixel_from_frames(img_array: Sequence) -> Tuple[float, int]:
    """Detect ID-1 card across frames; return (mm_per_pixel, n_detections)."""
    scales = collect_card_scales(img_array)
    if not scales:
        raise LookupError("no card detections")
    return aggregate_mm_per_pixel(scales), len(scales)
