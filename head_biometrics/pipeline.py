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
from typing import BinaryIO, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Repo root must be on sys.path so ``import src.*`` resolves (model paths inside
# the legacy code are also relative to the process cwd / repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEMO_MODE = os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

CIRCUMFERENCE_FACTOR = 0.834626841674

VALID_SCALE_MODES = frozenset({"magstripe", "mm_per_pixel", "reference_mm", "ipd"})


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
    scale_mode: Optional[str] = None
    scale_note: Optional[str] = None


@dataclass(frozen=True)
class ScaleOptions:
    scale_mode: str = "magstripe"
    mm_per_pixel: Optional[float] = None
    reference_width_mm: Optional[float] = None
    reference_width_px: Optional[float] = None
    ipd_mm: float = 63.0


def _try_import_pipeline():
    """Import legacy modules. Returns callables or raises DependencyError."""
    try:
        from src.key_frame_extraction import split_frames
        from src.narrow_wide_img_select import narrow_wide_img
        from src.front_quantify import front_mm_metrics_postmtrp
        from src.side_quantify import side_mm_metrics_postmtrp
        from src import scale_modes
    except Exception as exc:  # noqa: BLE001 — surface any import/runtime dep failure
        raise DependencyError(
            f"CV/TF pipeline dependencies are unavailable: {exc}"
        ) from exc
    return (
        split_frames,
        narrow_wide_img,
        front_mm_metrics_postmtrp,
        side_mm_metrics_postmtrp,
        scale_modes,
    )


def pipeline_available() -> bool:
    try:
        _try_import_pipeline()
        return True
    except DependencyError:
        return False


def _demo_result(reason: str, scale_mode: str = "magstripe") -> MeasurementResult:
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
        scale_mode=scale_mode,
        scale_note=None,
    )


def validate_scale_options(opts: ScaleOptions) -> None:
    """Raise DetectionError if form fields are inconsistent with scale_mode."""
    mode = (opts.scale_mode or "magstripe").strip().lower()
    if mode not in VALID_SCALE_MODES:
        raise DetectionError(
            f"Invalid scale_mode={opts.scale_mode!r}. "
            f"Expected one of: {', '.join(sorted(VALID_SCALE_MODES))}."
        )
    if mode == "mm_per_pixel":
        if opts.mm_per_pixel is None:
            raise DetectionError(
                "scale_mode=mm_per_pixel requires form field mm_per_pixel (float > 0)."
            )
        if float(opts.mm_per_pixel) <= 0:
            raise DetectionError("mm_per_pixel must be a positive float.")
    elif mode == "reference_mm":
        if opts.reference_width_mm is None or opts.reference_width_px is None:
            raise DetectionError(
                "scale_mode=reference_mm requires reference_width_mm and "
                "reference_width_px (both > 0). ISO ID-1 auto-detection is a "
                "follow-up; supply explicit pixel width from the client for now."
            )
        if float(opts.reference_width_mm) <= 0 or float(opts.reference_width_px) <= 0:
            raise DetectionError(
                "reference_width_mm and reference_width_px must be positive."
            )
    elif mode == "ipd":
        if opts.ipd_mm is None or float(opts.ipd_mm) <= 0:
            raise DetectionError("ipd_mm must be a positive float (default 63).")


def resolve_pixel_mm(img_array: Sequence, opts: ScaleOptions, scale_modes_mod):
    """Return ``(pixel_mm, scale_note)`` compatible with quantify helpers."""
    mode = (opts.scale_mode or "magstripe").strip().lower()
    validate_scale_options(ScaleOptions(
        scale_mode=mode,
        mm_per_pixel=opts.mm_per_pixel,
        reference_width_mm=opts.reference_width_mm,
        reference_width_px=opts.reference_width_px,
        ipd_mm=opts.ipd_mm if opts.ipd_mm is not None else 63.0,
    ))

    try:
        if mode == "magstripe":
            return scale_modes_mod.from_magstripe(img_array), None
        if mode == "mm_per_pixel":
            return scale_modes_mod.from_mm_per_pixel(opts.mm_per_pixel), None
        if mode == "reference_mm":
            return (
                scale_modes_mod.from_reference(
                    opts.reference_width_mm, opts.reference_width_px
                ),
                (
                    "Scale from client-supplied reference object "
                    f"({opts.reference_width_mm} mm / {opts.reference_width_px} px). "
                    "Automatic ISO ID-1 card detection is not enabled yet."
                ),
            )
        if mode == "ipd":
            ipd_mm = float(opts.ipd_mm) if opts.ipd_mm is not None else 63.0
            return scale_modes_mod.from_ipd(img_array, ipd_mm=ipd_mm)
    except scale_modes_mod.ScaleError as exc:
        raise DetectionError(str(exc)) from exc

    raise DetectionError(f"Unhandled scale_mode: {mode}")


def run_measurements(
    video_path: Union[str, Path],
    clockwise: bool = False,
    scale_options: Optional[ScaleOptions] = None,
) -> MeasurementResult:
    """Run the legacy pipeline and return all five millimeter metrics.

    Mirrors ``src.main.run`` but also exposes ``head_width`` and ``length``.
    """
    opts = scale_options or ScaleOptions()
    mode = (opts.scale_mode or "magstripe").strip().lower()

    try:
        (
            split_frames,
            narrow_wide_img,
            front_mm_metrics_postmtrp,
            side_mm_metrics_postmtrp,
            scale_modes,
        ) = _try_import_pipeline()
    except DependencyError as exc:
        if DEMO_MODE:
            logger.warning("DEMO_MODE active: %s", exc)
            return _demo_result(str(exc), scale_mode=mode)
        raise

    path = str(video_path)
    try:
        # Validate scale form fields early (before heavy CV) so bad requests 422.
        try:
            validate_scale_options(ScaleOptions(
                scale_mode=mode,
                mm_per_pixel=opts.mm_per_pixel,
                reference_width_mm=opts.reference_width_mm,
                reference_width_px=opts.reference_width_px,
                ipd_mm=opts.ipd_mm if opts.ipd_mm is not None else 63.0,
            ))
        except DetectionError:
            raise

        img_array = split_frames(path, clockwise)
        if img_array is None or len(img_array) == 0:
            raise DetectionError("No frames could be extracted from the video.")

        pixel_mm, scale_note = resolve_pixel_mm(img_array, opts, scale_modes)

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
            front2nape_mm, length_mm = side_mm_metrics_postmtrp(
                wide_img, pixel_mm, interactive=False
            )
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"Side measurement failed: {exc}") from exc

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
            scale_mode=mode,
            scale_note=scale_note,
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
    scale_options: Optional[ScaleOptions] = None,
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
        return run_measurements(
            tmp_path, clockwise=clockwise, scale_options=scale_options
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temp upload %s", tmp_path)
