from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter, log_diffusers_call


class TextToImageAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return ["TEXT_TO_IMAGE"]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return "TEXT_TO_IMAGE" in capabilities or "stable-diffusion" in str(metadata.get("model_type") or "").lower()

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 8 * 1024 * 1024 * 1024, "ram_bytes": 8 * 1024 * 1024 * 1024}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForText2Image, DiffusionPipeline

        try:
            return AutoPipelineForText2Image.from_pretrained(
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
        kwargs = {
            "prompt": request["prompt"],
            "negative_prompt": request.get("negative_prompt"),
            "width": request.get("width", 512),
            "height": request.get("height", 512),
            "num_inference_steps": request.get("steps", 4),
            "guidance_scale": request.get("guidance_scale", 0.0),
            "generator": runtime.get("generator"),
        }
        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        log_diffusers_call(pipeline, filtered, str(request.get("capability") or "TEXT_TO_IMAGE"))
        output = pipeline(**filtered)
        return {"images": getattr(output, "images", [])}
