"""FastAPI application exposing head biometric measurements over HTTP."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from head_biometrics import __version__
from head_biometrics.auth import require_api_key
from head_biometrics.jobs import get_job, submit_measure_job
from head_biometrics import metrics as metrics_mod
from head_biometrics.models import (
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    MeasureMeta,
    MeasureResponse,
    MeasurementsMm,
    ScaleModeInfo,
    ScaleModesResponse,
    VersionResponse,
)
from head_biometrics.pipeline import (
    DEMO_MODE,
    MAX_UPLOAD_BYTES,
    DependencyError,
    DetectionError,
    MeasurementResult,
    PipelineError,
    ScaleOptions,
    UploadTooLargeError,
    measure_upload,
    pipeline_available,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HeadBiometrics API",
    description=(
        "HTTP service wrapping the legacy OpenCV/TensorFlow head-measurement "
        "pipeline. Scale can come from a magstripe card, a full ISO ID-1 "
        "credit/ID card (scale_mode=card), a printed ArUco marker "
        "(scale_mode=aruco), an explicit mm/pixel value, a known reference "
        "object width, an IPD population prior, or an iris-diameter prior "
        "(scale_mode=iris). Long videos can use async jobs at "
        "POST /v1/measure/jobs. Confidence/warnings in meta are heuristic "
        "only — not medical-grade."
    ),
    version=__version__,
)

# Optional CORS via env: CORS_ORIGINS="*" or "https://a.com,https://b.com"
_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_raw:
    _origins = ["*"] if _cors_raw == "*" else [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for origins: %s", _origins)


@app.middleware("http")
async def metrics_and_access_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0
    path = request.url.path
    metrics_mod.record_request(path, request.method, response.status_code, duration_ms)
    return response


SCALE_MODE_CATALOG = [
    ScaleModeInfo(
        id="magstripe",
        aliases=[],
        required_fields=[],
        optional_fields=[],
        description=(
            "Detect credit-card magnetic stripe aspect ratios in frames and "
            "derive mm/pixel (legacy default)."
        ),
    ),
    ScaleModeInfo(
        id="card",
        aliases=["id1_card"],
        required_fields=[],
        optional_fields=[],
        description=(
            "Auto-detect a full ISO ID-1 credit/ID card (85.60x53.98 mm) via "
            "quad contours (ordered corner distances for longest side); "
            "mm/pixel = 85.60 / longest_side_px (median across frames). "
            "Show the card flat and fully visible."
        ),
    ),
    ScaleModeInfo(
        id="aruco",
        aliases=[],
        required_fields=["aruco_marker_length_mm"],
        optional_fields=["aruco_dict"],
        description=(
            "Detect a printed ArUco marker (OpenCV cv2.aruco; default dict "
            "4x4_50, also try 5x5_100). Require aruco_marker_length_mm = "
            "physical side length in mm. mm/pixel = length_mm / side_px "
            "(median across detections)."
        ),
    ),
    ScaleModeInfo(
        id="mm_per_pixel",
        aliases=[],
        required_fields=["mm_per_pixel"],
        optional_fields=[],
        description="Caller supplies millimeters per pixel; skips auto scale detection.",
    ),
    ScaleModeInfo(
        id="reference_mm",
        aliases=[],
        required_fields=["reference_width_mm", "reference_width_px"],
        optional_fields=[],
        description=(
            "Known physical object width in mm and its measured width in pixels. "
            "For automatic ISO ID-1 detection prefer scale_mode=card; for markers "
            "prefer scale_mode=aruco."
        ),
    ),
    ScaleModeInfo(
        id="ipd",
        aliases=[],
        required_fields=[],
        optional_fields=["ipd_mm"],
        description=(
            "Interpupillary distance from eye landmarks vs adult mean prior "
            "(default ipd_mm=63). Approximate — see meta.scale_note / warnings."
        ),
    ),
    ScaleModeInfo(
        id="iris",
        aliases=[],
        required_fields=[],
        optional_fields=["iris_mm"],
        description=(
            "Adult iris-diameter prior (default iris_mm=11.7) vs crude landmark "
            "proxy (eyelid aperture 37-40 / 43-46; fallback eye-width fraction). "
            "Approximate population prior — lower confidence like IPD; not "
            "medical-grade."
        ),
    ),
]


def _build_measure_response(
    result: MeasurementResult,
    *,
    clockwise: bool,
    filename: Optional[str],
    scale_mode_fallback: str,
) -> MeasureResponse:
    return MeasureResponse(
        measurements_mm=MeasurementsMm(
            circumference=result.circumference,
            front_to_nape=result.front_to_nape,
            ear_to_ear=result.ear_to_ear,
            head_width=result.head_width,
            length=result.length,
        ),
        meta=MeasureMeta(
            clockwise=clockwise,
            filename=filename,
            demo_mode=result.demo_mode,
            note=result.note,
            scale_mode=result.scale_mode or scale_mode_fallback,
            scale_note=result.scale_note,
            mm_per_pixel=result.mm_per_pixel,
            confidence=result.confidence,
            warnings=list(result.warnings or []),
            scale_frames_used=result.scale_frames_used,
        ),
    )


def _parse_scale_options(
    *,
    scale_mode: str,
    mm_per_pixel: Optional[float],
    reference_width_mm: Optional[float],
    reference_width_px: Optional[float],
    ipd_mm: Optional[float],
    iris_mm: Optional[float],
    aruco_dict: Optional[str],
    aruco_marker_length_mm: Optional[float],
) -> ScaleOptions:
    return ScaleOptions(
        scale_mode=(scale_mode or "magstripe").strip().lower(),
        mm_per_pixel=mm_per_pixel,
        reference_width_mm=reference_width_mm,
        reference_width_px=reference_width_px,
        ipd_mm=63.0 if ipd_mm is None else float(ipd_mm),
        iris_mm=11.7 if iris_mm is None else float(iris_mm),
        aruco_dict=(aruco_dict or "4x4_50").strip() if aruco_dict else "4x4_50",
        aruco_marker_length_mm=aruco_marker_length_mm,
    )


async def _read_upload_capped(
    video: UploadFile,
    request: Request,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Tuple[bytes, Optional[str]]:
    """Validate upload metadata and read body with a hard size cap."""
    if video is None:
        raise HTTPException(status_code=400, detail="Missing video file upload.")

    filename: Optional[str] = video.filename
    content_type = (video.content_type or "").lower()

    if filename is None or filename.strip() == "":
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    if content_type and not (
        content_type.startswith("video/")
        or content_type in {"application/octet-stream", "application/mp4"}
    ):
        logger.info("Unusual content-type for upload: %s", content_type)

    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Upload exceeds maximum size of {max_bytes} bytes "
                        f"({max_bytes // (1024 * 1024)} MB)."
                    ),
                )
        except ValueError:
            pass

    chunks = []
    total = 0
    while True:
        chunk = await video.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload exceeds maximum size of {max_bytes} bytes "
                    f"({max_bytes // (1024 * 1024)} MB)."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks), filename


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = pipeline_available()
    return HealthResponse(
        status="ok",
        pipeline_available=available,
        demo_mode=DEMO_MODE and not available,
        version=__version__,
    )


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(version=__version__)


@app.get(
    "/metrics",
    summary="Basic JSON metrics counters",
    description=(
        "In-process counters: requests_total, measure_success, measure_failure, "
        "jobs_created. Resets on process restart (stdlib JSON — no prometheus_client)."
    ),
)
def metrics() -> dict:
    return metrics_mod.snapshot()


@app.get(
    "/v1/scale-modes",
    response_model=ScaleModesResponse,
    summary="List available scale modes",
    description=(
        "Catalog of scale_mode values accepted by POST /v1/measure and "
        "POST /v1/measure/jobs and their required form fields."
    ),
)
def list_scale_modes() -> ScaleModesResponse:
    return ScaleModesResponse(modes=SCALE_MODE_CATALOG)


@app.post(
    "/v1/measure",
    response_model=MeasureResponse,
    summary="Measure head biometrics from a video upload",
    description=(
        "Multipart upload of a head-rotation video. Choose scale_mode:\n\n"
        "- magstripe (default): detect magstripe to mm/pixel\n"
        "- card / id1_card: detect full ISO ID-1 card (85.60x53.98 mm)\n"
        "- aruco: require aruco_marker_length_mm; optional aruco_dict (default 4x4_50)\n"
        "- mm_per_pixel: require form mm_per_pixel\n"
        "- reference_mm: require reference_width_mm + reference_width_px\n"
        "- ipd: optional ipd_mm (default 63)\n"
        "- iris: optional iris_mm (default 11.7); approximate eyelid-landmark proxy\n\n"
        "For long videos prefer POST /v1/measure/jobs (async). "
        "Responses include heuristic meta.confidence and meta.warnings "
        "(non-blocking; not medical-grade). Uploads larger than 100 MB → HTTP 413. "
        "When env API_KEY is set, require X-API-Key or Authorization: Bearer."
    ),
    responses={
        401: {"description": "API key required / invalid"},
        413: {"description": "Upload too large"},
        422: {"description": "Detection / scale / validation failure"},
        503: {"description": "CV/TF dependencies unavailable"},
    },
    dependencies=[Depends(require_api_key)],
)
async def measure(
    request: Request,
    video: UploadFile = File(..., description="Video file (e.g. MP4) of the head rotation"),
    clockwise: bool = Form(
        False,
        description="Rotate landscape frames 90 degrees clockwise (iOS); false = counter-clockwise (Android)",
    ),
    scale_mode: str = Form(
        "magstripe",
        description=(
            "Scale source: magstripe | card | id1_card | aruco | mm_per_pixel | "
            "reference_mm | ipd | iris"
        ),
    ),
    mm_per_pixel: Optional[float] = Form(
        None,
        description="Required when scale_mode=mm_per_pixel (millimeters per pixel)",
    ),
    reference_width_mm: Optional[float] = Form(
        None,
        description="Known object width in mm (scale_mode=reference_mm). ISO ID-1 long side = 85.60",
    ),
    reference_width_px: Optional[float] = Form(
        None,
        description="Same object width in pixels (scale_mode=reference_mm)",
    ),
    ipd_mm: Optional[float] = Form(
        63.0,
        description="IPD prior in mm for scale_mode=ipd (default adult mean 63)",
    ),
    iris_mm: Optional[float] = Form(
        11.7,
        description=(
            "Iris-diameter prior in mm for scale_mode=iris "
            "(default adult ≈ 11.7; approximate)"
        ),
    ),
    aruco_dict: Optional[str] = Form(
        "4x4_50",
        description="ArUco dictionary for scale_mode=aruco (default 4x4_50; also 5x5_100)",
    ),
    aruco_marker_length_mm: Optional[float] = Form(
        None,
        description=(
            "Required for scale_mode=aruco — physical side length of the printed "
            "marker in millimeters"
        ),
    ),
) -> MeasureResponse:
    from io import BytesIO

    file_bytes, filename = await _read_upload_capped(video, request)

    scale_options = _parse_scale_options(
        scale_mode=scale_mode,
        mm_per_pixel=mm_per_pixel,
        reference_width_mm=reference_width_mm,
        reference_width_px=reference_width_px,
        ipd_mm=ipd_mm,
        iris_mm=iris_mm,
        aruco_dict=aruco_dict,
        aruco_marker_length_mm=aruco_marker_length_mm,
    )

    try:
        result = measure_upload(
            BytesIO(file_bytes),
            filename=filename,
            clockwise=clockwise,
            scale_options=scale_options,
            max_bytes=MAX_UPLOAD_BYTES,
        )
    except UploadTooLargeError as exc:
        metrics_mod.record_measure_outcome(False)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except DetectionError as exc:
        metrics_mod.record_measure_outcome(False)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DependencyError as exc:
        metrics_mod.record_measure_outcome(False)
        raise HTTPException(
            status_code=503,
            detail=(
                f"{exc}. Install CV dependencies (see README) or set DEMO_MODE=1 "
                "for a clearly labeled mock response."
            ),
        ) from exc
    except PipelineError as exc:
        metrics_mod.record_measure_outcome(False)
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        metrics_mod.record_measure_outcome(False)
        logger.exception("Unexpected failure in /v1/measure")
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {exc}"
        ) from exc

    metrics_mod.record_measure_outcome(True)
    return _build_measure_response(
        result,
        clockwise=clockwise,
        filename=filename,
        scale_mode_fallback=scale_options.scale_mode,
    )


@app.post(
    "/v1/measure/jobs",
    response_model=JobCreateResponse,
    summary="Enqueue an async measurement job",
    description=(
        "Same multipart form fields as POST /v1/measure, but returns immediately "
        "with {job_id, status: queued}. Poll GET /v1/measure/jobs/{job_id} for "
        "queued|running|succeeded|failed.\n\n"
        "**Store:** ``JOB_STORE=memory`` (default) or ``sqlite`` "
        "(``JOB_STORE_PATH``, default ``./data/jobs.sqlite3``). SQLite persists "
        "status/result/error across restarts; workers are still in-process — "
        "**not multi-worker safe**. Use a single uvicorn worker for this MVP.\n\n"
        "When env API_KEY is set, require X-API-Key or Authorization: Bearer."
    ),
    responses={
        401: {"description": "API key required / invalid"},
        413: {"description": "Upload too large"},
    },
    dependencies=[Depends(require_api_key)],
)
async def measure_job_create(
    request: Request,
    video: UploadFile = File(..., description="Video file (e.g. MP4) of the head rotation"),
    clockwise: bool = Form(False),
    scale_mode: str = Form("magstripe"),
    mm_per_pixel: Optional[float] = Form(None),
    reference_width_mm: Optional[float] = Form(None),
    reference_width_px: Optional[float] = Form(None),
    ipd_mm: Optional[float] = Form(63.0),
    iris_mm: Optional[float] = Form(11.7),
    aruco_dict: Optional[str] = Form("4x4_50"),
    aruco_marker_length_mm: Optional[float] = Form(None),
) -> JobCreateResponse:
    file_bytes, filename = await _read_upload_capped(video, request)

    scale_options = _parse_scale_options(
        scale_mode=scale_mode,
        mm_per_pixel=mm_per_pixel,
        reference_width_mm=reference_width_mm,
        reference_width_px=reference_width_px,
        ipd_mm=ipd_mm,
        iris_mm=iris_mm,
        aruco_dict=aruco_dict,
        aruco_marker_length_mm=aruco_marker_length_mm,
    )

    record = submit_measure_job(
        file_bytes=file_bytes,
        filename=filename,
        clockwise=clockwise,
        scale_options=scale_options,
        max_bytes=MAX_UPLOAD_BYTES,
    )
    metrics_mod.record_job_created()
    return JobCreateResponse(job_id=record.job_id, status="queued")


@app.get(
    "/v1/measure/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get async measurement job status / result",
    description=(
        "Poll until status is succeeded (result present) or failed (error present). "
        "Job store is single-process (memory or sqlite); not multi-worker safe. "
        "When env API_KEY is set, require X-API-Key or Authorization: Bearer."
    ),
    responses={
        401: {"description": "API key required / invalid"},
        404: {"description": "Unknown job_id"},
    },
    dependencies=[Depends(require_api_key)],
)
def measure_job_status(job_id: str) -> JobStatusResponse:
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    result_payload = None
    if record.status == "succeeded" and record.result is not None:
        result_payload = _build_measure_response(
            record.result,
            clockwise=record.clockwise,
            filename=record.filename,
            scale_mode_fallback=record.scale_mode or "magstripe",
        )

    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        result=result_payload,
        error=record.error if record.status == "failed" else None,
        filename=record.filename,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app() -> FastAPI:
    """Factory for ASGI servers / tests."""
    return app
