from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter


class TextToVideoAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return ["TEXT_TO_VIDEO"]

    def supports_model(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        capabilities = set(
            metadata.get("capabilities", [])
        )

        return (
            "TEXT_TO_VIDEO" in capabilities
            or "video" in str(
                metadata.get("pipeline_tag") or ""
            ).lower()
        )

    def estimate_resources(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "vram_bytes": 16 * 1024 * 1024 * 1024,
            "ram_bytes": 16 * 1024 * 1024 * 1024,
        }

    @staticmethod
    def _is_wan(runtime: dict[str, Any]) -> bool:
        metadata = runtime.get("metadata") or {}

        class_name = str(
            metadata.get("class_name")
            or runtime.get("class_name")
            or ""
        ).lower()

        architectures = [
            str(value).lower()
            for value in metadata.get("architectures", [])
        ]

        return (
            "wanpipeline" in class_name
            or any(
                "wantransformer3dmodel" in value
                for value in architectures
            )
        )

    def load(
        self,
        snapshot: str,
        settings: dict[str, Any],
        runtime: Any,
    ) -> Any:
        if self._is_wan(runtime):
            import torch

            from diffusers import (
                AutoencoderKLWan,
                WanPipeline,
            )

            vae = AutoencoderKLWan.from_pretrained(
                snapshot,
                subfolder="vae",
                local_files_only=True,
                torch_dtype=torch.float32,
            )

            return WanPipeline.from_pretrained(
                snapshot,
                vae=vae,
                local_files_only=True,
                torch_dtype=settings.get("torch_dtype"),
            )

        from diffusers import (
            AutoPipelineForText2Video,
            DiffusionPipeline,
        )

        try:
            return AutoPipelineForText2Video.from_pretrained(
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

    def unload(
        self,
        pipeline: Any,
        runtime: Any,
    ) -> None:
        del pipeline

    def generate(
        self,
        pipeline: Any,
        runtime: Any,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        is_wan = self._is_wan(runtime)

        width = int(request.get("width") or 512)
        height = int(request.get("height") or 320)

        fps = int(request.get("fps") or 24)
        duration = int(
            request.get("duration_seconds") or 4
        )

        frames = request.get("frames")
        steps = request.get("steps")
        guidance = request.get("guidance_scale")

        if is_wan:
            # Wan2.2 TI2V-5B travaille en 720p avec une
            # zone de 1280x704 ou 704x1280.
            if width >= height:
                width = 1280
                height = 704
            else:
                width = 704
                height = 1280

            fps = 24

            # 4*N+1 est la forme attendue par les pipelines Wan.
            if not frames or int(frames) <= 8:
                frames = duration * fps + 1

            frames = int(frames)

            remainder = (frames - 1) % 4

            if remainder:
                frames -= remainder

            frames = max(5, frames)

            # Le "4" actuel de VidioAI est un ancien défaut
            # trop faible pour Wan.
            if not steps or int(steps) <= 4:
                steps = 50

            if guidance is None or float(guidance) <= 0.0:
                guidance = 5.0

        kwargs = {
            "prompt": request["prompt"],
            "negative_prompt": request.get(
                "negative_prompt"
            ),
            "height": height,
            "width": width,
            "num_frames": frames,
            "num_inference_steps": steps or 4,
            "guidance_scale": (
                guidance
                if guidance is not None
                else 0.0
            ),
            "generator": runtime.get("generator"),
        }

        accepted = set(
            inspect.signature(
                pipeline.__call__
            ).parameters
        )

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }

        output = pipeline(**filtered)

        return {
            "frames": getattr(output, "frames", []),
            "width": width,
            "height": height,
            "fps": fps,
        }