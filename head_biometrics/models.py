"""Pydantic response models for the HeadBiometrics API."""

from __future__ import annotations

from typing import Optional

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


class MeasureResponse(BaseModel):
    measurements_mm: MeasurementsMm
    meta: MeasureMeta


class HealthResponse(BaseModel):
    status: str = "ok"
    pipeline_available: bool = False
    demo_mode: bool = False
