from __future__ import annotations

import inspect
from typing import Any

from ..capability_resolver import CapabilityResolver
from ..model_profile import ModelRuntimeProfile
from ..normalizers import NormalizationError, assign_alias
from ..pipeline_resolver import PipelineResolver
from .base import RuntimeAdapter, log_diffusers_call


class GenericDiffusersAdapter(RuntimeAdapter):
    """Fallback runtime adapter for Diffusers pipelines discovered at runtime.

    Specialized adapters remain priority in the registry. This adapter is used
    only when a compatible Diffusers pipeline class exists but no dedicated
    adapter matches the model-specific capability profile.
    """

    def capabilities(self) -> list[str]:
        return [
            "TEXT_TO_IMAGE",
            "IMAGE_TO_IMAGE",
            "INPAINTING",
            "OUTPAINTING",
            "IMAGE_VARIATION",
            "IMAGE_UPSCALE",
            "CONTROLLED_IMAGE_GENERATION",
            "TEXT_TO_VIDEO",
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
            "VIDEO_TO_VIDEO",
            "VIDEO_INPAINTING",
            "VIDEO_UPSCALE",
        ]

    def supported_capabilities(self, metadata: dict[str, Any]) -> list[str]:
        resolution = PipelineResolver().resolve_class(metadata)
        return CapabilityResolver().resolve(metadata, resolution.pipeline_cls)

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        try:
            resolver = PipelineResolver()
            resolution = resolver.resolve_class(metadata)
            if resolution.runtime_supported:
                return True
            return bool(
                resolution.class_name is None
                and metadata.get("model_index")
                and str(metadata.get("library_name") or "diffusers").lower()
                in {"", "diffusers"}
                and not resolver.requires_remote_code(metadata)
            )
        except Exception:
            return False

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        class_name = str(metadata.get("class_name") or "").lower()
        if "video" in class_name or "wan" in class_name:
            return {"vram_bytes": 20 * 1024 * 1024 * 1024, "ram_bytes": 20 * 1024 * 1024 * 1024}
        return {"vram_bytes": 10 * 1024 * 1024 * 1024, "ram_bytes": 10 * 1024 * 1024 * 1024}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        pipeline, resolution = PipelineResolver().load(
            snapshot,
            runtime.get("metadata") or {},
            str(runtime.get("capability") or "TEXT_TO_IMAGE"),
            settings,
        )
        runtime["pipeline_resolution"] = resolution
        return pipeline

    def unload(self, pipeline: Any, runtime: Any) -> None:
        del pipeline

    @staticmethod
    def _has_output_items(value: Any) -> bool:
        if value is None:
            return False
        try:
            return len(value) > 0
        except TypeError:
            return True

    @classmethod
    def _normalize_output(
        cls, output: Any, capability: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        images = getattr(output, "images", None)
        if cls._has_output_items(images):
            return {"images": images, **values}

        frames = getattr(output, "frames", None)
        if cls._has_output_items(frames):
            return {"frames": frames, **values}

        if isinstance(output, (tuple, list)) and cls._has_output_items(output):
            candidate = output[0] if isinstance(output, tuple) and len(output) == 1 else output
            if cls._has_output_items(candidate):
                key = "frames" if "VIDEO" in capability else "images"
                return {key: candidate, **values}

        raise NormalizationError(
            "Le pipeline Diffusers n'a produit aucune image ou frame exploitable.",
            code="OUTPUT_NORMALIZATION_FAILED",
        )

    @staticmethod
    def _prepare_pipeline_inputs(request: dict[str, Any]) -> tuple[list[Any], list[str]]:
        resolved_images = request.get("resolved_input_images") or []
        if resolved_images:
            roles = [
                str(item.get("role") or "reference").lower()
                for item in (request.get("input_images") or [])
                if isinstance(item, dict)
            ]
            return list(resolved_images), roles

        if request.get("input_images"):
            raise NormalizationError(
                "Les asset IDs doivent être résolus vers des images avant l'appel pipeline.",
                code="INVALID_INPUT_ASSET",
            )
        return [], []

    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        capability = str(request.get("capability") or "TEXT_TO_IMAGE").upper()
        images, roles = self._prepare_pipeline_inputs(request)

        accepted = set(inspect.signature(pipeline.__call__).parameters)
        profile = ModelRuntimeProfile.from_metadata(runtime.get("metadata") or {}, pipeline)
        values = profile.normalize(request, video="VIDEO" in capability)
        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "width": values["width"],
            "height": values["height"],
            "num_inference_steps": values["num_inference_steps"],
            "guidance_scale": values["guidance_scale"],
            "generator": runtime.get("generator"),
            "fps": values["fps"],
            "decode_chunk_size": request.get("decode_chunk_size"),
            "strength": request.get("strength"),
            "callback_on_step_end": runtime.get("callback"),
        }

        assign_alias(kwargs, accepted, values["num_frames"], "num_frames", "video_length")
        assign_alias(kwargs, accepted, request.get("input_image"), "image")
        input_video = request.get("input_video")
        if input_video is None:
            input_video = request.get("input_frames")
        assign_alias(kwargs, accepted, input_video, "video", "frames")
        assign_alias(kwargs, accepted, request.get("mask_image"), "mask_image", "mask")
        assign_alias(kwargs, accepted, request.get("control_image"), "control_image")

        if capability in {
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        } and images:
            assign_alias(kwargs, accepted, images[0], "image")
            assign_alias(kwargs, accepted, images, "images", "reference_images", "conditioning_images")
            assign_alias(kwargs, accepted, images, "keyframes")
            assign_alias(kwargs, accepted, roles, "image_roles")
            end_index = None
            for index, role in enumerate(roles):
                if role in {"end", "end_frame"}:
                    end_index = index
                    break
            if end_index is not None:
                assign_alias(kwargs, accepted, images[end_index], "last_image", "end_image")
            elif len(images) > 1:
                assign_alias(kwargs, accepted, images[-1], "last_image", "end_image")

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        log_diffusers_call(pipeline, filtered, capability)
        output = pipeline(**filtered)
        return self._normalize_output(output, capability, values)
