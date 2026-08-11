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

    def supported_capabilities(
        self,
        metadata: dict[str, Any],
    ) -> list[str]:
        detected = {
            str(value).upper()
            for value in metadata.get(
                "capabilities",
                [],
            )
        }

        return [
            capability
            for capability in self.capabilities()
            if capability in detected
        ]

    def supports_model(
        self,
        metadata: dict[str, Any],
    ) -> bool:
        return bool(
            self.supported_capabilities(metadata)
        )

    def estimate_resources(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "vram_bytes": 18 * 1024 * 1024 * 1024,
            "ram_bytes": 18 * 1024 * 1024 * 1024,
        }

    @staticmethod
    def _is_wan(
        metadata: dict[str, Any],
    ) -> bool:
        class_name = str(
            metadata.get("class_name") or ""
        ).lower()

        architectures = [
            str(value).lower()
            for value in metadata.get(
                "architectures",
                [],
            )
        ]

        return (
            "wanpipeline" in class_name
            or "wanimagetovideopipeline" in class_name
            or any(
                "wantransformer3dmodel" in value
                for value in architectures
            )
        )

    def input_profile(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}

        if self._is_wan(metadata):
            return {
                "min_input_images": 1,
                "max_input_images": 1,
                "supported_image_roles": [
                    "start",
                    "start_frame",
                ],
                "supports_start_end_frames": False,
                "supports_reference_images": False,
                "supports_keyframes": False,
            }

        model_name = str(
            metadata.get("class_name") or ""
        ).lower()

        if "ltx" in model_name:
            return {
                "min_input_images": 1,
                "max_input_images": 2,
                "supported_image_roles": [
                    "start",
                    "end",
                    "start_frame",
                    "end_frame",
                ],
                "supports_start_end_frames": True,
                "supports_reference_images": False,
                "supports_keyframes": False,
            }

        if (
            "cogvideox" in model_name
            or "cog" in model_name
        ):
            return {
                "min_input_images": 1,
                "max_input_images": 8,
                "supported_image_roles": [
                    "reference",
                    "keyframe",
                    "start",
                    "end",
                ],
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

    def prepare_pipeline_inputs(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        raw_images = request.get(
            "input_images"
        ) or []

        images = []
        roles = []

        for item in raw_images:
            if not isinstance(item, dict):
                continue

            asset_id = item.get("asset_id")

            if asset_id is None:
                continue

            images.append(asset_id)

            roles.append(
                str(
                    item.get("role")
                    or "reference"
                ).lower()
            )

        return {
            "images": images,
            "roles": roles,
        }

    def load(
        self,
        snapshot: str,
        settings: dict[str, Any],
        runtime: Any,
    ) -> Any:
        metadata = runtime.get("metadata") or {}

        if self._is_wan(metadata):
            import torch

            from diffusers import (
                AutoencoderKLWan,
                WanImageToVideoPipeline,
            )

            vae = AutoencoderKLWan.from_pretrained(
                snapshot,
                subfolder="vae",
                local_files_only=True,
                torch_dtype=torch.float32,
            )

            return (
                WanImageToVideoPipeline
                .from_pretrained(
                    snapshot,
                    vae=vae,
                    local_files_only=True,
                    torch_dtype=settings.get(
                        "torch_dtype"
                    ),
                )
            )

        from diffusers import (
            AutoPipelineForImage2Video,
            DiffusionPipeline,
        )

        try:
            return (
                AutoPipelineForImage2Video
                .from_pretrained(
                    snapshot,
                    local_files_only=True,
                    use_safetensors=True,
                    torch_dtype=settings.get(
                        "torch_dtype"
                    ),
                )
            )
        except Exception:
            return (
                DiffusionPipeline
                .from_pretrained(
                    snapshot,
                    local_files_only=True,
                    use_safetensors=True,
                    torch_dtype=settings.get(
                        "torch_dtype"
                    ),
                )
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
        metadata = runtime.get("metadata") or {}

        is_wan = self._is_wan(metadata)

        resolved_images = (
            request.get(
                "resolved_input_images"
            )
            or []
        )

        if not resolved_images:
            raise RuntimeError(
                "Une image réelle est obligatoire "
                "pour IMAGE_TO_VIDEO."
            )

        image = resolved_images[0]

        width = int(
            request.get("width") or 512
        )

        height = int(
            request.get("height") or 320
        )

        fps = int(
            request.get("fps") or 24
        )

        duration = int(
            request.get(
                "duration_seconds"
            )
            or 4
        )

        frames = request.get("frames")
        steps = request.get("steps")
        guidance = request.get(
            "guidance_scale"
        )

        if is_wan:
            image_width, image_height = image.size

            if image_width >= image_height:
                width = 1280
                height = 704
            else:
                width = 704
                height = 1280

            fps = 24

            if not frames or int(frames) <= 8:
                frames = duration * fps + 1

            frames = int(frames)

            remainder = (frames - 1) % 4

            if remainder:
                frames -= remainder

            frames = max(5, frames)

            if not steps or int(steps) <= 4:
                steps = 50

            if (
                guidance is None
                or float(guidance) <= 0.0
            ):
                guidance = 5.0

        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get(
                "negative_prompt"
            ),
            "image": image,
            "height": height,
            "width": width,
            "num_frames": frames,
            "num_inference_steps": (
                steps or 4
            ),
            "guidance_scale": (
                guidance
                if guidance is not None
                else 0.0
            ),
            "generator": runtime.get(
                "generator"
            ),
        }

        accepted = set(
            inspect.signature(
                pipeline.__call__
            ).parameters
        )

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted
            and value is not None
        }

        output = pipeline(**filtered)

        return {
            "frames": getattr(
                output,
                "frames",
                [],
            ),
            "width": width,
            "height": height,
            "fps": fps,
        }