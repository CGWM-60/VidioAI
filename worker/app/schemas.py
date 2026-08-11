"""Contrats HTTP internes partagés conceptuellement avec le backend Rust."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelState(StrEnum):
    NOT_INSTALLED = "NOT_INSTALLED"
    DISCOVERED = "DISCOVERED"
    COMPATIBILITY_CHECK = "COMPATIBILITY_CHECK"
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    VALIDATING = "VALIDATING"
    INSTALLED = "INSTALLED"
    LOADING = "LOADING"
    RUNTIME_UNAVAILABLE = "RUNTIME_UNAVAILABLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    READY = "READY"
    FAILED = "FAILED"


class JobState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class InstallModelRequest(BaseModel):
    model_id: str = Field(min_length=2, max_length=128)
    repository: str = Field(min_length=3, max_length=256)
    revision: str = Field(default="main", min_length=1, max_length=128)
    capabilities: list[str] = Field(default_factory=lambda: ["TEXT_TO_IMAGE"])


class ModelRequest(BaseModel):
    model_id: str = Field(min_length=2, max_length=128)


class CompatibilityRequest(BaseModel):
    pipeline_class: str | None = Field(default=None, max_length=256)
    library_name: str | None = Field(default="diffusers", max_length=64)
    pipeline_tag: str | None = Field(default=None, max_length=128)
    tags: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)
    base_models: list[str] = Field(default_factory=list)
    trust_remote_code: bool = False


class GenerateImageRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
    model_id: str = Field(min_length=2, max_length=128)
    prompt: str = Field(min_length=3, max_length=1000)
    negative_prompt: str | None = Field(default=None, max_length=1000)
    output_relative_path: str = Field(min_length=5, max_length=512)
    width: int | None = Field(default=None, ge=64, le=2048)
    height: int | None = Field(default=None, ge=64, le=2048)
    quality: str | None = Field(default=None, max_length=16)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    capability: str | None = Field(default=None, max_length=32)
    seed: int | None = Field(default=None, ge=0)
    input_path: str | None = Field(default=None, max_length=2048)
    mask_path: str | None = Field(default=None, max_length=2048)
    control_path: str | None = Field(default=None, max_length=2048)


class GenerateVideoRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
    model_id: str = Field(min_length=2, max_length=128)
    prompt: str = Field(min_length=3, max_length=1000)
    negative_prompt: str | None = Field(default=None, max_length=1000)
    output_relative_path: str = Field(min_length=5, max_length=512)
    width: int | None = Field(default=None, ge=64, le=2048)
    height: int | None = Field(default=None, ge=64, le=2048)
    quality: Literal["480p", "720p", "1080p", "auto"] = "480p"
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    duration_seconds: int | None = Field(default=None, ge=1, le=20)
    fps: int | None = Field(default=None, ge=1, le=60)
    frames: int | None = Field(default=None, ge=1, le=180)
    capability: str | None = Field(default=None, max_length=32)
    seed: int | None = Field(default=None, ge=0)
    input_path: str | None = Field(default=None, max_length=2048)
    mask_path: str | None = Field(default=None, max_length=2048)
    input_images: list[dict[str, Any]] = Field(default_factory=list)


class UnsupportedGenerationRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
    model_id: str = Field(min_length=2, max_length=128)
    output_relative_path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CancelRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
