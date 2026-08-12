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
    RESOLVING_DEPENDENCIES = "RESOLVING_DEPENDENCIES"
    DOWNLOADING_DEPENDENCY = "DOWNLOADING_DEPENDENCY"
    INSTALLING_DEPENDENCIES = "INSTALLING_DEPENDENCIES"
    INSTALLED = "INSTALLED"
    LOADING = "LOADING"
    UNLOADING = "UNLOADING"
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


class CompatibilityStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class LoraSpec(BaseModel):
    repository: str = Field(min_length=3, max_length=256)
    revision: str = Field(default="main", min_length=1, max_length=128)
    adapter_name: str | None = Field(default=None, min_length=1, max_length=64)
    weight_name: str | None = Field(default=None, min_length=1, max_length=512)
    scale: float = Field(default=1.0, ge=0.0, le=2.0)
    enabled: bool = True


class InferenceRecipe(BaseModel):
    quality_mode: Literal["native", "fast", "balanced", "quality"] = "native"
    width: int | None = Field(default=None, ge=64, le=2048)
    height: int | None = Field(default=None, ge=64, le=2048)
    num_inference_steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    true_cfg_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    max_sequence_length: int | None = Field(default=None, ge=16, le=8192)
    inference_fps: int | None = Field(default=None, ge=1, le=60)


class InstallModelRequest(BaseModel):
    model_id: str = Field(min_length=2, max_length=128)
    repository: str = Field(min_length=3, max_length=256)
    revision: str = Field(default="main", min_length=1, max_length=128)
    capabilities: list[str] = Field(default_factory=lambda: ["TEXT_TO_IMAGE"])
    # None = ne pas modifier le bundle lors d'une installation idempotente.
    # [] = retirer tous les LoRA.
    loras: list[LoraSpec] | None = None
    recipe: InferenceRecipe | None = None


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
    is_modular: bool = False


class GenerateImageRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
    model_id: str = Field(min_length=2, max_length=128)
    prompt: str = Field(min_length=3, max_length=1000)
    negative_prompt: str | None = Field(default=None, max_length=1000)
    output_relative_path: str = Field(min_length=5, max_length=512)
    width: int | None = Field(default=None, ge=64, le=2048)
    height: int | None = Field(default=None, ge=64, le=2048)
    quality: Literal["native", "fast", "balanced", "quality"] | None = None
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    true_cfg_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    max_sequence_length: int | None = Field(default=None, ge=16, le=8192)
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
    true_cfg_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    max_sequence_length: int | None = Field(default=None, ge=16, le=8192)
    inference_fps: int | None = Field(default=None, ge=1, le=60)
    duration_seconds: int | None = Field(default=None, ge=1, le=20)
    fps: int | None = Field(default=None, ge=1, le=60)
    frames: int | None = Field(default=None, ge=1, le=180)
    capability: str | None = Field(default=None, max_length=32)
    seed: int | None = Field(default=None, ge=0)
    input_path: str | None = Field(default=None, max_length=2048)
    mask_path: str | None = Field(default=None, max_length=2048)
    input_images: list[dict[str, Any]] = Field(default_factory=list)
    audio: bool = False


class UnsupportedGenerationRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
    model_id: str = Field(min_length=2, max_length=128)
    output_relative_path: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class CancelRequest(BaseModel):
    job_id: str = Field(min_length=8, max_length=128)
