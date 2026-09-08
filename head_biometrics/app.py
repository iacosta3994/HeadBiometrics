"""FastAPI application exposing head biometric measurements over HTTP."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from head_biometrics.models import (
    HealthResponse,
    MeasureMeta,
    MeasureResponse,
    MeasurementsMm,
)
from head_biometrics.pipeline import (
    DEMO_MODE,
    DependencyError,
    DetectionError,
    PipelineError,
    measure_upload,
    pipeline_available,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HeadBiometrics API",
    description=(
        "HTTP service wrapping the legacy OpenCV/TensorFlow head-measurement "
        "pipeline. Upload a video that includes a visible magstripe scale card."
    ),
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    available = pipeline_available()
    return HealthResponse(
        status="ok",
        pipeline_available=available,
        demo_mode=DEMO_MODE and not available,
    )


@app.post("/v1/measure", response_model=MeasureResponse)
async def measure(
    video: UploadFile = File(..., description="Video file (e.g. MP4) of the head rotation"),
    clockwise: bool = Form(
        False,
        description="Rotate landscape frames 90° clockwise (iOS); false = counter-clockwise (Android)",
    ),
) -> MeasureResponse:
    if video is None:
        raise HTTPException(status_code=400, detail="Missing video file upload.")

    filename: Optional[str] = video.filename
    content_type = (video.content_type or "").lower()

    # Reject obviously non-file empties; allow octet-stream / unset types.
    if filename is None or filename.strip() == "":
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    # Soft content-type check — browsers sometimes send application/octet-stream.
    if content_type and not (
        content_type.startswith("video/")
        or content_type in {"application/octet-stream", "application/mp4"}
    ):
        # Still accept; many clients mislabel. Log only.
        logger.info("Unusual content-type for upload: %s", content_type)

    try:
        result = measure_upload(video.file, filename=filename, clockwise=clockwise)
    except DetectionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DependencyError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{exc}. Install CV dependencies (see README) or set DEMO_MODE=1 "
                "for a clearly labeled mock response."
            ),
        ) from exc
    except PipelineError as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure in /v1/measure")
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {exc}"
        ) from exc

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
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app() -> FastAPI:
    """Factory for ASGI servers / tests."""
    return app
