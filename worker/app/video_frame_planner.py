"""Planification generique des frames d'inference Diffusers."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read(source: Any, name: str) -> Any:
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


@dataclass(frozen=True, slots=True)
class VideoFramePlan:
    requested_duration_seconds: float | None
    requested_fps: int | None
    requested_frames: int | None
    inference_frames: int | None
    min_frames: int | None
    max_frames: int | None
    default_frames: int | None
    temporal_multiple: int
    parameter: str | None
    source: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoFramePlanner:
    """Dissocie la cible produit du nombre de frames accepte par le modele."""

    @staticmethod
    def _signature(pipeline: Any) -> inspect.Signature:
        return inspect.signature(pipeline.__call__)

    @staticmethod
    def _pipeline_sources(pipeline: Any) -> list[tuple[str, Any]]:
        return [
            ("pipeline_config", getattr(pipeline, "config", None)),
            ("transformer_config", getattr(getattr(pipeline, "transformer", None), "config", None)),
            ("unet_config", getattr(getattr(pipeline, "unet", None), "config", None)),
            ("vae_config", getattr(getattr(pipeline, "vae", None), "config", None)),
        ]

    @staticmethod
    def _metadata_sources(metadata: dict[str, Any]) -> list[tuple[str, Any]]:
        model_index = metadata.get("model_index") or {}
        config = metadata.get("config") or {}
        component_configs = metadata.get("component_configs") or {}
        return [
            ("model_index_runtime", model_index.get("vidioai_runtime") or {}),
            ("config_runtime", config.get("vidioai_runtime") or {}),
            ("model_index", model_index),
            ("config", config),
            ("transformer_metadata", component_configs.get("transformer") or {}),
            ("unet_metadata", component_configs.get("unet") or {}),
            ("vae_metadata", component_configs.get("vae") or {}),
        ]

    @staticmethod
    def _first(
        sources: list[tuple[str, Any]], names: tuple[str, ...]
    ) -> tuple[int | None, str | None]:
        for source_name, source in sources:
            for name in names:
                value = _positive_int(_read(source, name))
                if value is not None:
                    return value, f"{source_name}.{name}"
        return None, None

    @staticmethod
    def _align(value: int, multiple: int, *, upward: bool = False) -> int:
        if multiple <= 1:
            return max(1, value)
        offset = max(0, value - 1)
        if upward:
            aligned = ((offset + multiple - 1) // multiple) * multiple + 1
        else:
            aligned = (offset // multiple) * multiple + 1
        return max(1, aligned)

    def plan(
        self,
        *,
        pipeline: Any,
        metadata: dict[str, Any],
        capability: str,
        requested_duration: float | int | None,
        requested_fps: int | None,
        requested_frames: int | None,
        vram_free_bytes: int = 0,
        ram_available_bytes: int = 0,
        width: int = 0,
        height: int = 0,
    ) -> VideoFramePlan:
        del vram_free_bytes, ram_available_bytes, width, height
        duration = float(requested_duration) if requested_duration is not None else None
        fps = _positive_int(requested_fps)
        explicit_frames = _positive_int(requested_frames)
        product_frames = (
            max(1, round(duration * fps))
            if duration is not None and duration > 0 and fps is not None
            else None
        )
        product_request = explicit_frames or product_frames

        signature = self._signature(pipeline)
        parameters = signature.parameters
        parameter = "num_frames" if "num_frames" in parameters else (
            "video_length" if "video_length" in parameters else None
        )

        default_frames = None
        default_source = None
        for name in ("num_frames", "video_length"):
            signature_parameter = parameters.get(name)
            if signature_parameter is None or signature_parameter.default is inspect.Parameter.empty:
                continue
            default_frames = _positive_int(signature_parameter.default)
            if default_frames is not None:
                default_source = f"pipeline_signature.{name}"
                break

        pipeline_sources = self._pipeline_sources(pipeline)
        metadata_sources = self._metadata_sources(metadata)
        if default_frames is None:
            default_frames, default_source = self._first(
                pipeline_sources + metadata_sources,
                ("num_frames", "video_length", "default_num_frames", "default_video_length"),
            )

        min_frames, _ = self._first(
            pipeline_sources + metadata_sources,
            ("min_frames", "min_num_frames", "minimum_frames"),
        )
        max_frames, _ = self._first(
            pipeline_sources + metadata_sources,
            ("max_frames", "max_num_frames", "maximum_frames"),
        )
        temporal_multiple, _ = self._first(
            pipeline_sources + metadata_sources,
            ("temporal_multiple", "temporal_compression_ratio"),
        )
        temporal_multiple = temporal_multiple or 1

        if "VIDEO" not in capability.upper():
            return VideoFramePlan(
                duration, fps, product_request, None, min_frames, max_frames,
                default_frames, temporal_multiple, None, "not_video",
                "La capability ne genere pas de video.",
            )
        if parameter is None:
            return VideoFramePlan(
                duration, fps, product_request, None, min_frames, max_frames,
                default_frames, temporal_multiple, None, "pipeline_signature",
                "La signature n'accepte ni num_frames ni video_length; aucun parametre frame ne sera injecte.",
            )

        if explicit_frames is not None:
            inference = explicit_frames
            source = "requested_frames"
            reason = "Le nombre de frames a ete demande explicitement et la signature l'accepte."
        elif default_frames is not None:
            inference = default_frames
            source = default_source or "pipeline_metadata"
            reason = "La cible duree/FPS reste un objectif produit; le defaut reel de la pipeline est utilise pour l'inference."
        elif min_frames is not None:
            inference = min_frames
            source = "minimum_model_frames"
            reason = "Aucun defaut n'est expose; le minimum modele constitue la valeur d'inference sure sans convertir duree × FPS."
        else:
            return VideoFramePlan(
                duration, fps, product_request, None, min_frames, max_frames,
                None, temporal_multiple, parameter, "pipeline_signature",
                "Aucun nombre de frames explicite ou defaut modele n'est connu; Diffusers conservera son comportement natif.",
            )

        original = inference
        if min_frames is not None and inference < min_frames:
            inference = self._align(min_frames, temporal_multiple, upward=True)
        else:
            inference = self._align(inference, temporal_multiple)
        if max_frames is not None and inference > max_frames:
            inference = self._align(max_frames, temporal_multiple)
        if inference != original:
            reason += f" Valeur alignee/bornee de {original} a {inference}."

        return VideoFramePlan(
            duration, fps, product_request, inference, min_frames, max_frames,
            default_frames, temporal_multiple, parameter, source, reason,
        )
