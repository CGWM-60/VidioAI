from __future__ import annotations

from typing import Any

from .base import RuntimeAdapter


class TextToVideoAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return ["TEXT_TO_VIDEO"]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return "TEXT_TO_VIDEO" in capabilities or "video" in str(metadata.get("pipeline_tag") or "").lower()

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 16 * 1024 * 1024 * 1024, "ram_bytes": 16 * 1024 * 1024 * 1024}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForText2Video

        return AutoPipelineForText2Video.from_pretrained(
            snapshot,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=settings.get("torch_dtype"),
        )

    def unload(self, pipeline: Any, runtime: Any) -> None:
        del pipeline

    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        output = pipeline(
            prompt=request["prompt"],
            negative_prompt=request.get("negative_prompt"),
            height=request.get("height", 320),
            width=request.get("width", 512),
            num_inference_steps=request.get("steps", 4),
            guidance_scale=request.get("guidance_scale", 0.0),
            generator=runtime.get("generator"),
        )
        return {"frames": getattr(output, "frames", [])}
