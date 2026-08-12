from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter, log_diffusers_call


class ImageToImageAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return [
            "IMAGE_TO_IMAGE", "INPAINTING", "OUTPAINTING",
            "IMAGE_VARIATION", "IMAGE_UPSCALE", "CONTROLLED_IMAGE_GENERATION",
        ]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return bool(
            capabilities.intersection(set(self.capabilities()))
            or "img2img" in str(metadata.get("pipeline_tag") or "").lower()
            or "inpaint" in str(metadata.get("class_name") or "").lower()
        )

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "vram_bytes": 10 * 1024 * 1024 * 1024,
            "ram_bytes": 10 * 1024 * 1024 * 1024,
        }

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForImage2Image, DiffusionPipeline

        try:
            return AutoPipelineForImage2Image.from_pretrained(
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
        capability = str(request.get("capability") or "IMAGE_TO_IMAGE").upper()
        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "image": request.get("input_image"),
            "generator": runtime.get("generator"),
            "mask_image": request.get("mask_image"),
            "control_image": request.get("control_image"),
        }

        if request.get("strength") is not None:
            kwargs["strength"] = request.get("strength")
        if request.get("steps") is not None:
            kwargs["num_inference_steps"] = request.get("steps")
        if request.get("guidance_scale") is not None:
            kwargs["guidance_scale"] = request.get("guidance_scale")
        if request.get("width") is not None:
            kwargs["width"] = request.get("width")
        if request.get("height") is not None:
            kwargs["height"] = request.get("height")

        if capability == "IMAGE_VARIATION":
            kwargs.pop("strength", None)

        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        log_diffusers_call(pipeline, filtered, capability)
        output = pipeline(**filtered)
        return {"images": getattr(output, "images", [])}
