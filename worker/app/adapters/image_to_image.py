from __future__ import annotations

from typing import Any

from .base import RuntimeAdapter


class ImageToImageAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return ["IMAGE_TO_IMAGE"]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return "IMAGE_TO_IMAGE" in capabilities or "img2img" in str(metadata.get("pipeline_tag") or "").lower()

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 10 * 1024 * 1024 * 1024, "ram_bytes": 10 * 1024 * 1024 * 1024}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForImage2Image

        return AutoPipelineForImage2Image.from_pretrained(
            snapshot,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=settings.get("torch_dtype"),
        )

    def unload(self, pipeline: Any, runtime: Any) -> None:
        del pipeline

    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        image = request["input_image"]
        output = pipeline(
            prompt=request["prompt"],
            image=image,
            negative_prompt=request.get("negative_prompt"),
            strength=request.get("strength", 0.8),
            num_inference_steps=request.get("steps", 4),
            guidance_scale=request.get("guidance_scale", 0.0),
            generator=runtime.get("generator"),
        )
        return {"images": getattr(output, "images", [])}
