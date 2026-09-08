"""Service layer wrapping the legacy OpenCV/TF measurement pipeline.

Prefers importing the existing ``src`` modules. When the CV stack cannot be
imported and ``DEMO_MODE=1``, returns clearly labeled mock measurements.
Never silently invents real-looking results without the demo flag.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

logger = logging.getLogger(__name__)

# Repo root must be on sys.path so ``import src.*`` resolves (model paths inside
# the legacy code are also relative to the process cwd / repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEMO_MODE = os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

CIRCUMFERENCE_FACTOR = 0.834626841674


class PipelineError(Exception):
    """Base error for measurement pipeline failures."""


class DetectionError(PipelineError):
    """Raised when the video cannot be measured (missing face, magstripe, etc.)."""


class DependencyError(PipelineError):
    """Raised when required CV/TF dependencies are unavailable."""


@dataclass(frozen=True)
class MeasurementResult:
    circumference: int
    front_to_nape: int
    ear_to_ear: int
    head_width: int
    length: int
    demo_mode: bool = False
    note: Optional[str] = None


def _try_import_pipeline():
    """Import legacy modules. Returns callables or raises DependencyError."""
    try:
        from src.key_frame_extraction import split_frames
        from src.video_to_mm import video_to_pixel_mm
        from src.narrow_wide_img_select import narrow_wide_img
        from src.front_quantify import front_mm_metrics_postmtrp
        from src.side_quantify import side_mm_metrics_postmtrp
    except Exception as exc:  # noqa: BLE001 — surface any import/runtime dep failure
        raise DependencyError(
            f"CV/TF pipeline dependencies are unavailable: {exc}"
        ) from exc
    return (
        split_frames,
        video_to_pixel_mm,
        narrow_wide_img,
        front_mm_metrics_postmtrp,
        side_mm_metrics_postmtrp,
    )


def pipeline_available() -> bool:
    try:
        _try_import_pipeline()
        return True
    except DependencyError:
        return False


def _demo_result(reason: str) -> MeasurementResult:
    return MeasurementResult(
        circumference=560,
        front_to_nape=340,
        ear_to_ear=150,
        head_width=155,
        length=195,
        demo_mode=True,
        note=(
            "DEMO_MODE mock measurements — not derived from the uploaded video. "
            f"Reason: {reason}"
        ),
    )


def run_measurements(video_path: Union[str, Path], clockwise: bool = False) -> MeasurementResult:
    """Run the legacy pipeline and return all five millimeter metrics.

    Mirrors ``src.main.run`` but also exposes ``head_width`` and ``length``.
    """
    try:
        (
            split_frames,
            video_to_pixel_mm,
            narrow_wide_img,
            front_mm_metrics_postmtrp,
            side_mm_metrics_postmtrp,
        ) = _try_import_pipeline()
    except DependencyError as exc:
        if DEMO_MODE:
            logger.warning("DEMO_MODE active: %s", exc)
            return _demo_result(str(exc))
        raise

    path = str(video_path)
    try:
        img_array = split_frames(path, clockwise)
        if img_array is None or len(img_array) == 0:
            raise DetectionError("No frames could be extracted from the video.")

        try:
            pixel_mm = video_to_pixel_mm(img_array)
        except (ValueError, TypeError, IndexError) as exc:
            raise DetectionError(
                "Magstripe scale detection failed. Ensure a credit-card-style "
                f"magnetic stripe is visible in the video. Details: {exc}"
            ) from exc

        if pixel_mm is None or (hasattr(pixel_mm, "__len__") and len(pixel_mm) == 0):
            raise DetectionError(
                "Magstripe scale detection returned no usable pixel/mm estimate."
            )

        try:
            narrow_img, wide_img = narrow_wide_img(img_array)
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(
                f"Face / head-pose detection failed: {exc}"
            ) from exc

        if narrow_img is None or wide_img is None:
            raise DetectionError(
                "Could not select narrow (front) and wide (side) frames from the video."
            )

        try:
            ear_to_ear_mm, head_width_mm = front_mm_metrics_postmtrp(narrow_img, pixel_mm)
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"Front measurement failed: {exc}") from exc

        try:
            front2nape_mm, length_mm = side_mm_metrics_postmtrp(wide_img, pixel_mm)
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(
                "Side measurement failed. Note: the legacy side pipeline uses an "
                f"interactive OpenCV GUI for point selection. Details: {exc}"
            ) from exc

        circumference_mm = int(
            ((head_width_mm * 2) + (length_mm * 2)) * CIRCUMFERENCE_FACTOR
        )

        return MeasurementResult(
            circumference=int(circumference_mm),
            front_to_nape=int(front2nape_mm),
            ear_to_ear=int(ear_to_ear_mm),
            head_width=int(head_width_mm),
            length=int(length_mm),
            demo_mode=False,
            note=None,
        )
    except DetectionError:
        raise
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Unexpected pipeline failure: {exc}") from exc


def measure_upload(
    file_obj: BinaryIO,
    filename: Optional[str],
    clockwise: bool = False,
) -> MeasurementResult:
    """Persist an uploaded video to a temp file and run measurements."""
    suffix = Path(filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        if tmp_path.stat().st_size == 0:
            raise DetectionError("Uploaded file is empty.")
        return run_measurements(tmp_path, clockwise=clockwise)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temp upload %s", tmp_path)
