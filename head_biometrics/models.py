"""Pydantic response models for the HeadBiometrics API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MeasurementsMm(BaseModel):
    circumference: int = Field(..., description="Estimated head circumference in millimeters")
    front_to_nape: int = Field(..., description="Front-to-nape measurement in millimeters")
    ear_to_ear: int = Field(..., description="Ear-to-ear measurement in millimeters")
    head_width: int = Field(..., description="Head width (front view) in millimeters")
    length: int = Field(..., description="Head length (side view) in millimeters")


class MeasureMeta(BaseModel):
    clockwise: bool
    filename: Optional[str] = None
    demo_mode: bool = False
    note: Optional[str] = None
    scale_mode: Optional[str] = Field(
        None,
        description=(
            "Scale resolution mode: magstripe | card | aruco | mm_per_pixel | "
            "reference_mm | ipd (alias: id1_card → card)"
        ),
    )
    scale_note: Optional[str] = Field(
        None,
        description="Extra info about scale (e.g. IPD prior caveat, card/aruco detect summary)",
    )
    mm_per_pixel: Optional[float] = Field(
        None,
        description="Millimeters per pixel actually used for quantification (transparency)",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic quality score 0-1 (not calibrated, not medical-grade). "
            "Lower when few scale detections or approximate modes (e.g. ipd)."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking quality / scale caveats for clients",
    )
    scale_frames_used: Optional[int] = Field(
        None,
        description="Number of frame detections contributing to auto scale (card/aruco)",
    )


class MeasureResponse(BaseModel):
    measurements_mm: MeasurementsMm
    meta: MeasureMeta


class HealthResponse(BaseModel):
    status: str = "ok"
    pipeline_available: bool = False
    demo_mode: bool = False
    version: Optional[str] = None


class VersionResponse(BaseModel):
    version: str
    name: str = "HeadBiometrics API"


class ScaleModeInfo(BaseModel):
    id: str
    aliases: List[str] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)
    description: str


class ScaleModesResponse(BaseModel):
    modes: List[ScaleModeInfo]
