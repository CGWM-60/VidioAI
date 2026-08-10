from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter


class ImageToImageAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return [
            "IMAGE_TO_IMAGE",
            "INPAINTING",
            "OUTPAINTING",
            "IMAGE_VARIATION",
            "IMAGE_UPSCALE",
            "CONTROLLED_IMAGE_GENERATION",
        ]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return bool(
            capabilities.intersection(set(self.capabilities()))
            or "img2img" in str(metadata.get("pipeline_tag") or "").lower()
            or "inpaint" in str(metadata.get("class_name") or "").lower()
        )

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 10 * 1024 * 1024 * 1024, "ram_bytes": 10 * 1024 * 1024 * 1024}

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
        image = request.get("input_image")
        capability = str(request.get("capability") or "IMAGE_TO_IMAGE").upper()
        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "image": image,
            "strength": request.get("strength", 0.8),
            "num_inference_steps": request.get("steps", 4),
            "guidance_scale": request.get("guidance_scale", 0.0),
            "generator": runtime.get("generator"),
            "width": request.get("width"),
            "height": request.get("height"),
            "mask_image": request.get("mask_image"),
            "control_image": request.get("control_image"),
        }
        if capability == "IMAGE_VARIATION":
            kwargs.pop("strength", None)
        if capability in {"IMAGE_UPSCALE", "OUTPAINTING"}:
            kwargs["width"] = request.get("width", 1024)
            kwargs["height"] = request.get("height", 1024)

        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        output = pipeline(**filtered)
        return {"images": getattr(output, "images", [])}
