from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter


class ImageToVideoAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return [
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        ]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return bool(
            capabilities.intersection(set(self.capabilities()))
            or "img2vid" in str(metadata.get("pipeline_tag") or "").lower()
        )

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 18 * 1024 * 1024 * 1024, "ram_bytes": 18 * 1024 * 1024 * 1024}

    def input_profile(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        model_name = str((metadata or {}).get("class_name") or "").lower()
        if "ltx" in model_name:
            return {
                "min_input_images": 1,
                "max_input_images": 2,
                "supported_image_roles": ["start", "end", "start_frame", "end_frame"],
                "supports_start_end_frames": True,
                "supports_reference_images": False,
                "supports_keyframes": False,
            }
        if "cogvideox" in model_name or "cog" in model_name:
            return {
                "min_input_images": 1,
                "max_input_images": 8,
                "supported_image_roles": ["reference", "keyframe", "start", "end"],
                "supports_start_end_frames": False,
                "supports_reference_images": True,
                "supports_keyframes": True,
            }
        return {
            "min_input_images": 1,
            "max_input_images": 1,
            "supported_image_roles": [],
            "supports_start_end_frames": False,
            "supports_reference_images": False,
            "supports_keyframes": False,
        }

    def prepare_pipeline_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_images = request.get("input_images") or []
        images = []
        roles = []
        for item in raw_images:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id")
            if asset_id is None:
                continue
            images.append(asset_id)
            roles.append(str(item.get("role") or "reference").lower())
        return {"images": images, "roles": roles}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForImage2Video, DiffusionPipeline

        try:
            return AutoPipelineForImage2Video.from_pretrained(
                snapshot,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=settings.get("torch_dtype"),
            )
        except Exception:
            return DiffusionPipeline.from_pretrained(
                snapshot,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=settings.get("torch_dtype"),
            )

    def unload(self, pipeline: Any, runtime: Any) -> None:
        del pipeline

    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare_pipeline_inputs(request)
        resolved_images = request.get("resolved_input_images") or []
        if resolved_images:
            prepared = {
                "images": resolved_images,
                "roles": [
                    str(item.get("role") or "reference").lower()
                    for item in (request.get("input_images") or [])
                ],
            }
        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "height": request.get("height", 320),
            "width": request.get("width", 512),
            "num_frames": request.get("frames"),
            "fps": request.get("fps"),
            "num_inference_steps": request.get("steps", 4),
            "guidance_scale": request.get("guidance_scale", 0.0),
            "generator": runtime.get("generator"),
        }
        if prepared["images"]:
            kwargs["image"] = prepared["images"][0]
            kwargs["images"] = prepared["images"]
            kwargs["keyframes"] = prepared["images"]
            kwargs["image_roles"] = prepared["roles"]
            end_index = None
            for index, role in enumerate(prepared["roles"]):
                if role in {"end", "end_frame"}:
                    end_index = index
                    break
            if end_index is not None:
                kwargs["end_image"] = prepared["images"][end_index]
            elif len(prepared["images"]) > 1:
                kwargs["end_image"] = prepared["images"][-1]
        else:
            kwargs["image"] = request.get("input_image")
        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        output = pipeline(**filtered)
        return {"frames": getattr(output, "frames", [])}
