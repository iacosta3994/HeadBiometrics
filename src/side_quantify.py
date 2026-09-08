"""Side-view head metrics (length, front-to-nape).

Default path is fully headless: facial landmarks + silhouette extremes pick
points automatically. Pass ``interactive=True`` to fall back to the legacy
OpenCV mouse GUI for local debugging only.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from src.canny_edge_detection_cv2 import auto_canny_face, make_sobel_face
from src.face_contour_width import get_contour, img_head_contour_side

logger = logging.getLogger(__name__)

Point = Tuple[int, int]


class SidePointError(Exception):
    """Raised when landmarks or silhouette cannot yield measurement points."""


def point_return(img):
    """Interactive mouse picker (legacy). Returns (point_a, point_b)."""
    points = []

    def point_select(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        if event == cv2.EVENT_LBUTTONUP:
            points.append([x, y])

    cv2.namedWindow("img")
    cv2.setMouseCallback("img", point_select)
    cv2.imshow("img", img)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return points[-2], points[-1]


def dis_in_points(pointA, pointB):
    (xA, yA) = pointA
    (xB, yB) = pointB
    return round(np.sqrt((xA - xB) ** 2 + (yA - yB) ** 2))


def _as_point(pt) -> Point:
    return (int(pt[0]), int(pt[1]))


def _head_silhouette_contour(img):
    """Largest external contour from the same canny+sobel path used for f2nape."""
    edge = auto_canny_face(img)
    edge = make_sobel_face(edge)
    if len(edge.shape) == 3:
        gray = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
    else:
        gray = edge
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_list = get_contour(contours)
    if contour_list is None or len(contour_list) == 0:
        raise SidePointError("No head silhouette contour found on side frame.")
    return max(contour_list, key=cv2.contourArea)


def _detect_face_and_marks(img):
    """Run Proctoring_AI face detector + landmark model on ``img``."""
    try:
        from src.Proctoring_AI.face_detector import find_faces, get_face_detector
        from src.Proctoring_AI.face_landmarks import detect_marks, get_landmark_model
    except Exception as exc:  # noqa: BLE001
        raise SidePointError(
            f"Face/landmark models unavailable for headless side metrics: {exc}"
        ) from exc

    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    faces = find_faces(img, face_model)
    if not faces:
        raise SidePointError("No face detected on side (wide) frame.")
    # Prefer the largest face box.
    face = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    marks = detect_marks(img, landmark_model, face)
    if marks is None or len(marks) < 31:
        raise SidePointError("Facial landmarks incomplete on side frame.")
    return face, marks


def _facing_direction(face, marks) -> str:
    """Return 'left' or 'right' based on nose tip vs face-box center."""
    nose = marks[30]
    face_cx = (face[0] + face[2]) / 2.0
    # Nose tip ahead of face center indicates facing that way on a profile.
    if float(nose[0]) < face_cx:
        return "left"
    return "right"


def _glabella_point(marks) -> Point:
    """Approx glabella / mid-brow from landmarks 21, 22 (fallback: 27)."""
    try:
        p21, p22 = marks[21], marks[22]
        return _as_point(((int(p21[0]) + int(p22[0])) // 2, (int(p21[1]) + int(p22[1])) // 2))
    except (IndexError, TypeError):
        return _as_point(marks[27])


def _farthest_contour_point(contour, origin: Point, prefer_x_sign: int) -> Point:
    """Pick silhouette point farthest from ``origin`` in the preferred x direction.

    ``prefer_x_sign`` is -1 (left/back when facing right) or +1 (right/back when
    facing left). Among candidates on that side of ``origin``, choose max
    distance; if none, fall back to global farthest.
    """
    ox, oy = origin
    pts = contour.reshape(-1, 2)
    best = None
    best_d = -1.0
    fallback = None
    fallback_d = -1.0
    for x, y in pts:
        dx = float(x) - ox
        dy = float(y) - oy
        d = dx * dx + dy * dy
        if d > fallback_d:
            fallback_d = d
            fallback = (int(x), int(y))
        if prefer_x_sign < 0 and x > ox:
            continue
        if prefer_x_sign > 0 and x < ox:
            continue
        if d > best_d:
            best_d = d
            best = (int(x), int(y))
    return best if best is not None else fallback


def _nape_point(contour, front: Point, facing: str) -> Point:
    """Silhouette extreme toward the back and somewhat downward (nape)."""
    fx, fy = front
    pts = contour.reshape(-1, 2)
    # Score: back-direction x displacement + downward bias.
    best = None
    best_score = None
    for x, y in pts:
        if facing == "left":
            # Face looks left → back is +x
            back = float(x) - fx
        else:
            back = fx - float(x)
        down = float(y) - fy
        if back <= 0:
            continue
        # Prefer back strongly; mild preference for lower points (nape vs crown).
        score = back + 0.35 * max(down, 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best = (int(x), int(y))
    if best is None:
        # Fallback: farthest back-side point without down bias.
        prefer = 1 if facing == "left" else -1
        best = _farthest_contour_point(contour, front, prefer)
    if best is None:
        raise SidePointError("Could not locate nape on head silhouette.")
    # Ensure x differs from front so img_head_contour_side slope is defined.
    if best[0] == fx:
        nudge = 1 if facing == "left" else -1
        best = (best[0] + nudge, best[1])
    return best


def auto_select_side_points(
    img,
    face=None,
    marks=None,
    contour=None,
) -> Tuple[Point, Point, Point, Point]:
    """Return (length_front, length_back, f2nape_front, f2nape_nape).

    Optional ``face`` / ``marks`` / ``contour`` let unit tests inject fakes
    without loading TF models.
    """
    if face is None or marks is None:
        face, marks = _detect_face_and_marks(img)
    if contour is None:
        contour = _head_silhouette_contour(img)

    facing = _facing_direction(face, marks)
    logger.debug("Side auto-select: facing=%s", facing)

    length_front = _glabella_point(marks)
    prefer_back = 1 if facing == "left" else -1
    length_back = _farthest_contour_point(contour, length_front, prefer_back)
    if length_back is None:
        raise SidePointError("Could not locate back-of-head on silhouette.")

    f2nape_front = _as_point(marks[30])  # nose tip
    f2nape_nape = _nape_point(contour, f2nape_front, facing)

    return length_front, length_back, f2nape_front, f2nape_nape


def side_mm_metrics_postmtrp(wide_img, pixel_mm, interactive: bool = False):
    """Compute front-to-nape and length in millimeters from a side frame.

    Parameters
    ----------
    wide_img : ndarray
        Side / profile frame.
    pixel_mm : sequence
        Scale factor; ``pixel_mm[0]`` is mm per pixel (legacy contract).
    interactive : bool
        If True, use OpenCV mouse picks (GUI). Default False (headless API).
    """
    if interactive:
        print("select length points")
        eyebrows, back = point_return(wide_img)
        length = dis_in_points(eyebrows, back)
        length = int(length * pixel_mm[0])

        print("select front to nape")
        front, nape = point_return(wide_img)
        front2nape = img_head_contour_side(wide_img, front, nape)
        front2nape = int(front2nape * pixel_mm[0])
        front2nape = int(front2nape * 0.5)
        return front2nape, length

    try:
        length_front, length_back, f2nape_front, f2nape_nape = auto_select_side_points(
            wide_img
        )
    except SidePointError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SidePointError(f"Headless side point selection failed: {exc}") from exc

    length = dis_in_points(length_front, length_back)
    length = int(length * pixel_mm[0])

    try:
        front2nape = img_head_contour_side(wide_img, f2nape_front, f2nape_nape)
    except Exception as exc:  # noqa: BLE001
        raise SidePointError(
            f"Front-to-nape contour arc failed after auto point pick: {exc}"
        ) from exc

    front2nape = int(front2nape * pixel_mm[0])
    front2nape = int(front2nape * 0.5)
    return front2nape, length
