"""Parametres d'inference derives du snapshot puis normalises pour la pipeline.

2026.08.11-11:
- pas de fallback arbitraire pour steps/guidance;
- metadata puis defaults reels de pipeline.__call__;
- FPS de livraison separe du FPS de conditionnement modele.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


def _positive_int(*values: Any) -> int | None:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _number(*values: Any) -> float | None:
    for value in values:
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _signature_default(pipeline: Any, name: str) -> Any:
    if pipeline is None:
        return None
    try:
        parameter = inspect.signature(pipeline.__call__).parameters.get(name)
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    return parameter.default


@dataclass(slots=True)
class ModelRuntimeProfile:
    fps: int
    steps: int | None
    guidance_scale: float | None
    width: int
    height: int
    num_frames: int | None
    dimension_multiple: int
    temporal_multiple: int
    min_frames: int | None
    max_frames: int | None

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
        pipeline: Any = None,
    ) -> "ModelRuntimeProfile":
        model_index = metadata.get("model_index") or {}
        config = metadata.get("config") or {}
        profile = model_index.get("vidioai_runtime") or config.get("vidioai_runtime") or {}
        pipeline_config = getattr(pipeline, "config", None)
        vae_config = getattr(getattr(pipeline, "vae", None), "config", None)
        transformer_config = getattr(getattr(pipeline, "transformer", None), "config", None)

        def runtime_value(source: Any, name: str) -> Any:
            if isinstance(source, dict):
                return source.get(name)
            return getattr(source, name, None)

        fps = _positive_int(
            profile.get("fps"),
            model_index.get("fps"),
            config.get("fps"),
        ) or 24

        steps = _positive_int(
            profile.get("num_inference_steps"),
            model_index.get("num_inference_steps"),
            config.get("num_inference_steps"),
            config.get("default_num_inference_steps"),
            _signature_default(pipeline, "num_inference_steps"),
        )
        guidance = _number(
            profile.get("guidance_scale"),
            model_index.get("guidance_scale"),
            config.get("guidance_scale"),
            config.get("default_guidance_scale"),
            _signature_default(pipeline, "guidance_scale"),
        )
        width = _positive_int(
            profile.get("width"),
            model_index.get("width"),
            config.get("width"),
            config.get("sample_size"),
        ) or 512
        height = _positive_int(
            profile.get("height"),
            model_index.get("height"),
            config.get("height"),
            config.get("sample_size"),
        ) or 512

        return cls(
            fps=fps,
            steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            num_frames=_positive_int(
                profile.get("num_frames"),
                model_index.get("num_frames"),
                config.get("num_frames"),
            ),
            dimension_multiple=_positive_int(
                profile.get("dimension_multiple"),
                config.get("dimension_multiple"),
                config.get("vae_scale_factor"),
            ) or 8,
            temporal_multiple=_positive_int(
                profile.get("temporal_multiple"),
                config.get("temporal_compression_ratio"),
                runtime_value(vae_config, "temporal_compression_ratio"),
                runtime_value(transformer_config, "temporal_compression_ratio"),
            ) or 1,
            min_frames=_positive_int(
                profile.get("min_frames"),
                config.get("min_frames"),
                runtime_value(pipeline_config, "min_frames"),
            ),
            max_frames=_positive_int(
                profile.get("max_frames"),
                config.get("max_frames"),
                runtime_value(pipeline_config, "max_frames"),
            ),
        )

    @staticmethod
    def _align(value: int, multiple: int) -> int:
        return max(multiple, (value // multiple) * multiple)

    def normalize(self, request: dict[str, Any], *, video: bool) -> dict[str, Any]:
        width = _positive_int(request.get("width")) or self.width
        height = _positive_int(request.get("height")) or self.height
        fps = _positive_int(request.get("fps")) or self.fps
        duration = _positive_int(request.get("duration_seconds")) or 4
        frames = _positive_int(request.get("frames"), self.num_frames)
        if video and frames is not None:
            frames = self._normalize_frames(frames)

        explicit_guidance = (
            _number(request.get("guidance_scale"))
            if request.get("guidance_scale") is not None
            else None
        )
        return {
            "width": self._align(width, self.dimension_multiple),
            "height": self._align(height, self.dimension_multiple),
            "fps": fps,
            "duration_seconds": duration,
            "num_frames": frames if video else None,
            "num_inference_steps": _positive_int(request.get("steps")) or self.steps,
            "guidance_scale": (
                explicit_guidance
                if request.get("guidance_scale") is not None
                else self.guidance_scale
            ),
        }

    def _normalize_frames(self, requested: int) -> int:
        frames = max(1, requested)
        if self.temporal_multiple > 1:
            frames = max(
                1,
                round((frames - 1) / self.temporal_multiple)
                * self.temporal_multiple
                + 1,
            )
        if self.min_frames is not None and frames < self.min_frames:
            frames = self.min_frames
            if self.temporal_multiple > 1:
                frames = (
                    ((frames - 1 + self.temporal_multiple - 1) // self.temporal_multiple)
                    * self.temporal_multiple
                    + 1
                )
        if self.max_frames is not None and frames > self.max_frames:
            frames = self.max_frames
            if self.temporal_multiple > 1:
                frames = ((max(1, frames - 1) // self.temporal_multiple) * self.temporal_multiple) + 1
        return frames
