from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter


class VideoToVideoAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return ["VIDEO_TO_VIDEO", "VIDEO_INPAINTING", "VIDEO_UPSCALE"]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return bool(
            capabilities.intersection(set(self.capabilities()))
            or "vid2vid" in str(metadata.get("pipeline_tag") or "").lower()
        )

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 20 * 1024 * 1024 * 1024, "ram_bytes": 20 * 1024 * 1024 * 1024}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForVideoToVideo, DiffusionPipeline

        try:
            return AutoPipelineForVideoToVideo.from_pretrained(
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
        video = request.get("input_video")
        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "video": video,
            "frames": request.get("input_frames"),
            "height": request.get("height", 320),
            "width": request.get("width", 512),
            "num_frames": request.get("frames"),
            "fps": request.get("fps"),
            "num_inference_steps": request.get("steps", 4),
            "guidance_scale": request.get("guidance_scale", 0.0),
            "generator": runtime.get("generator"),
            "mask_image": request.get("mask_image"),
            "strength": request.get("strength"),
        }
        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        output = pipeline(**filtered)
        return {"frames": getattr(output, "frames", [])}
