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
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Repo root must be on sys.path so ``import src.*`` resolves (model paths inside
# the legacy code are also relative to the process cwd / repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEMO_MODE = os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

CIRCUMFERENCE_FACTOR = 0.834626841674

# Canonical modes after alias normalization (id1_card → card).
VALID_SCALE_MODES = frozenset(
    {"magstripe", "mm_per_pixel", "reference_mm", "ipd", "iris", "card", "aruco"}
)

# Default upload limit (bytes). Overridable via env for ops.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))

# Package version (also exposed via /health and /version)
try:
    from head_biometrics import __version__ as PACKAGE_VERSION
except Exception:  # noqa: BLE001
    PACKAGE_VERSION = "0.0.0"


class PipelineError(Exception):
    """Base error for measurement pipeline failures."""


class DetectionError(PipelineError):
    """Raised when the video cannot be measured (missing face, magstripe, etc.)."""


class DependencyError(PipelineError):
    """Raised when required CV/TF dependencies are unavailable."""


class UploadTooLargeError(PipelineError):
    """Raised when an upload exceeds MAX_UPLOAD_BYTES."""


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
    mm_per_pixel: Optional[float] = None
    # Quality / confidence (heuristic — not calibrated, not medical-grade)
    confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    scale_frames_used: Optional[int] = None


@dataclass(frozen=True)
class ScaleOptions:
    scale_mode: str = "magstripe"
    mm_per_pixel: Optional[float] = None
    reference_width_mm: Optional[float] = None
    reference_width_px: Optional[float] = None
    ipd_mm: float = 63.0
    iris_mm: float = 11.7
    aruco_dict: Optional[str] = "4x4_50"
    aruco_marker_length_mm: Optional[float] = None


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


def compute_quality_meta(
    *,
    scale_mode: str,
    scale_frames_used: Optional[int] = None,
    demo_mode: bool = False,
    had_narrow_wide: bool = True,
) -> tuple:
    """Return ``(confidence 0–1, warnings[])`` — heuristic only, not calibrated.

    Documented intent: inform clients; never block a successful measurement.
    """
    warnings: List[str] = []
    conf = 0.85

    if demo_mode:
        return 0.0, [
            "DEMO_MODE mock measurements — confidence is zero; not from video."
        ]

    mode = (scale_mode or "magstripe").strip().lower()
    if mode == "id1_card":
        mode = "card"

    if mode == "ipd":
        conf = min(conf, 0.40)
        warnings.append(
            "IPD scale uses a population prior (≈63 mm adult mean) — approximate only; "
            "prefer magstripe, card, aruco, or a measured reference when accuracy matters."
        )
    elif mode == "iris":
        conf = min(conf, 0.35)
        warnings.append(
            "Iris scale uses an adult iris-diameter prior (≈11.7 mm) and a crude "
            "eyelid-landmark aperture proxy — approximate only; prefer magstripe, "
            "card, aruco, or a measured reference when accuracy matters."
        )
    elif mode == "mm_per_pixel":
        conf = min(conf, 0.90)
        warnings.append(
            "Scale is client-supplied mm_per_pixel; accuracy depends on that value."
        )
    elif mode == "reference_mm":
        conf = min(conf, 0.80)
        warnings.append(
            "Reference scale depends on client-supplied reference_width_px."
        )
    elif mode in {"magstripe", "card", "aruco"}:
        n = int(scale_frames_used) if scale_frames_used is not None else None
        if n is None:
            # Magstripe path does not always expose a count
            conf = min(conf, 0.70)
            if mode == "magstripe":
                warnings.append(
                    "Magstripe scale resolved; per-frame detection count not available."
                )
        elif n <= 0:
            conf = min(conf, 0.25)
            warnings.append("No scale detections recorded (unexpected).")
        elif n == 1:
            conf = min(conf, 0.55)
            warnings.append("Scale derived from only 1 frame detection — low support.")
        elif n < 5:
            conf = min(conf, 0.72)
            warnings.append(
                f"Scale derived from only {n} frame detections — moderate support."
            )
        else:
            conf = min(conf, 0.92)

    if not had_narrow_wide:
        conf = min(conf, 0.35)
        warnings.append(
            "Narrow/wide head-pose extremes were missing or weak; measurements may be off."
        )

    warnings.append(
        "confidence is a heuristic (not calibrated) — not medical-grade."
    )
    # Clamp
    conf = max(0.0, min(1.0, float(conf)))
    return conf, warnings


def _demo_result(reason: str, scale_mode: str = "magstripe") -> MeasurementResult:
    conf, warns = compute_quality_meta(
        scale_mode=scale_mode, demo_mode=True, had_narrow_wide=True
    )
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
        mm_per_pixel=None,
        confidence=conf,
        warnings=warns,
        scale_frames_used=None,
    )


def _extract_mm_per_pixel(pixel_mm) -> Optional[float]:
    """Pull a scalar mm/pixel from the legacy nested ``pixel_mm`` contract."""
    try:
        first = pixel_mm[0]
        while isinstance(first, (list, tuple)):
            first = first[0]
        val = float(first)
        return val if val > 0 else None
    except (IndexError, TypeError, ValueError):
        return None


def validate_scale_options(opts: ScaleOptions) -> None:
    """Raise DetectionError if form fields are inconsistent with scale_mode."""
    try:
        from src.scale_modes import normalize_scale_mode

        mode = normalize_scale_mode(opts.scale_mode)
    except Exception:
        mode = (opts.scale_mode or "magstripe").strip().lower()
        if mode == "id1_card":
            mode = "card"

    if mode not in VALID_SCALE_MODES:
        raise DetectionError(
            f"Invalid scale_mode={opts.scale_mode!r}. "
            f"Expected one of: {', '.join(sorted(VALID_SCALE_MODES))} "
            "(alias: id1_card → card)."
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
                "reference_width_px (both > 0). For automatic ISO ID-1 detection "
                "use scale_mode=card instead."
            )
        if float(opts.reference_width_mm) <= 0 or float(opts.reference_width_px) <= 0:
            raise DetectionError(
                "reference_width_mm and reference_width_px must be positive."
            )
    elif mode == "ipd":
        if opts.ipd_mm is None or float(opts.ipd_mm) <= 0:
            raise DetectionError("ipd_mm must be a positive float (default 63).")
    elif mode == "iris":
        if opts.iris_mm is None or float(opts.iris_mm) <= 0:
            raise DetectionError("iris_mm must be a positive float (default 11.7).")
    elif mode == "aruco":
        if opts.aruco_marker_length_mm is None:
            raise DetectionError(
                "scale_mode=aruco requires aruco_marker_length_mm (float > 0) — "
                "physical side length of the printed marker in millimeters."
            )
        if float(opts.aruco_marker_length_mm) <= 0:
            raise DetectionError("aruco_marker_length_mm must be a positive float.")
    # card / magstripe: no extra fields required


def resolve_pixel_mm(img_array: Sequence, opts: ScaleOptions, scale_modes_mod):
    """Return ``(pixel_mm, scale_note, scale_frames_used)``."""
    mode = scale_modes_mod.normalize_scale_mode(opts.scale_mode)
    validate_scale_options(
        ScaleOptions(
            scale_mode=mode,
            mm_per_pixel=opts.mm_per_pixel,
            reference_width_mm=opts.reference_width_mm,
            reference_width_px=opts.reference_width_px,
            ipd_mm=opts.ipd_mm if opts.ipd_mm is not None else 63.0,
            iris_mm=opts.iris_mm if opts.iris_mm is not None else 11.7,
            aruco_dict=opts.aruco_dict,
            aruco_marker_length_mm=opts.aruco_marker_length_mm,
        )
    )

    try:
        if mode == "magstripe":
            return scale_modes_mod.from_magstripe(img_array), None, None
        if mode == "mm_per_pixel":
            return scale_modes_mod.from_mm_per_pixel(opts.mm_per_pixel), None, None
        if mode == "reference_mm":
            return (
                scale_modes_mod.from_reference(
                    opts.reference_width_mm, opts.reference_width_px
                ),
                (
                    "Scale from client-supplied reference object "
                    f"({opts.reference_width_mm} mm / {opts.reference_width_px} px)."
                ),
                None,
            )
        if mode == "ipd":
            ipd_mm = float(opts.ipd_mm) if opts.ipd_mm is not None else 63.0
            pixel_mm, note = scale_modes_mod.from_ipd(img_array, ipd_mm=ipd_mm)
            return pixel_mm, note, None
        if mode == "iris":
            iris_mm = float(opts.iris_mm) if opts.iris_mm is not None else 11.7
            pixel_mm, note = scale_modes_mod.from_iris(img_array, iris_mm=iris_mm)
            return pixel_mm, note, None
        if mode == "card":
            pixel_mm, note, n = scale_modes_mod.from_card(img_array)
            return pixel_mm, note, n
        if mode == "aruco":
            pixel_mm, note, n = scale_modes_mod.from_aruco(
                img_array,
                marker_length_mm=float(opts.aruco_marker_length_mm),
                aruco_dict=opts.aruco_dict,
            )
            return pixel_mm, note, n
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

    try:
        (
            split_frames,
            narrow_wide_img,
            front_mm_metrics_postmtrp,
            side_mm_metrics_postmtrp,
            scale_modes,
        ) = _try_import_pipeline()
    except DependencyError as exc:
        mode_early = (opts.scale_mode or "magstripe").strip().lower()
        if mode_early == "id1_card":
            mode_early = "card"
        if DEMO_MODE:
            logger.warning("DEMO_MODE active: %s", exc)
            return _demo_result(str(exc), scale_mode=mode_early)
        raise

    mode = scale_modes.normalize_scale_mode(opts.scale_mode)

    path = str(video_path)
    try:
        # Validate scale form fields early (before heavy CV) so bad requests 422.
        try:
            validate_scale_options(
                ScaleOptions(
                    scale_mode=mode,
                    mm_per_pixel=opts.mm_per_pixel,
                    reference_width_mm=opts.reference_width_mm,
                    reference_width_px=opts.reference_width_px,
                    ipd_mm=opts.ipd_mm if opts.ipd_mm is not None else 63.0,
                    iris_mm=opts.iris_mm if opts.iris_mm is not None else 11.7,
                    aruco_dict=opts.aruco_dict,
                    aruco_marker_length_mm=opts.aruco_marker_length_mm,
                )
            )
        except DetectionError:
            raise

        img_array = split_frames(path, clockwise)
        if img_array is None or len(img_array) == 0:
            raise DetectionError("No frames could be extracted from the video.")

        pixel_mm, scale_note, scale_frames_used = resolve_pixel_mm(
            img_array, opts, scale_modes
        )
        used_mm_per_pixel = _extract_mm_per_pixel(pixel_mm)

        had_narrow_wide = True
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

        conf, warns = compute_quality_meta(
            scale_mode=mode,
            scale_frames_used=scale_frames_used,
            demo_mode=False,
            had_narrow_wide=had_narrow_wide,
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
            mm_per_pixel=used_mm_per_pixel,
            confidence=conf,
            warnings=warns,
            scale_frames_used=scale_frames_used,
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
    max_bytes: Optional[int] = None,
) -> MeasurementResult:
    """Persist an uploaded video to a temp file and run measurements."""
    limit = MAX_UPLOAD_BYTES if max_bytes is None else int(max_bytes)
    suffix = Path(filename or "upload.mp4").suffix or ".mp4"
    total = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = file_obj.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise UploadTooLargeError(
                    f"Upload exceeds maximum size of {limit} bytes "
                    f"({limit // (1024 * 1024)} MB)."
                )
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
