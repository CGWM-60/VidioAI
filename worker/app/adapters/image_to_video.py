from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter, log_diffusers_call


class ImageToVideoAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return [
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        ]

    def supported_capabilities(self, metadata: dict[str, Any]) -> list[str]:
        detected = {
            str(value).upper()
            for value in metadata.get("capabilities", [])
        }
        return [
            capability
            for capability in self.capabilities()
            if capability in detected
        ]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        return bool(self.supported_capabilities(metadata))

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "vram_bytes": 18 * 1024 * 1024 * 1024,
            "ram_bytes": 18 * 1024 * 1024 * 1024,
        }

    @staticmethod
    def _is_wan(metadata: dict[str, Any]) -> bool:
        class_name = str(metadata.get("class_name") or "").lower()
        architectures = [
            str(value).lower()
            for value in metadata.get("architectures", [])
        ]
        return (
            "wanpipeline" in class_name
            or "wanimagetovideopipeline" in class_name
            or any("wantransformer3dmodel" in value for value in architectures)
        )

    def input_profile(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        if self._is_wan(metadata):
            return {
                "min_input_images": 1,
                "max_input_images": 1,
                "supported_image_roles": ["start", "start_frame"],
                "supports_start_end_frames": False,
                "supports_reference_images": False,
                "supports_keyframes": False,
            }

        model_name = str(metadata.get("class_name") or "").lower()
        if "ltx" in model_name:
            return {
                "min_input_images": 1,
                "max_input_images": 2,
                "supported_image_roles": [
                    "start", "end", "start_frame", "end_frame"
                ],
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
        raw_images = sorted(
            [item for item in (request.get("input_images") or []) if isinstance(item, dict)],
            key=lambda item: int(item.get("order") or 0),
        )
        images: list[Any] = []
        roles: list[str] = []
        for item in raw_images:
            asset_id = item.get("asset_id")
            if asset_id is None:
                continue
            images.append(asset_id)
            roles.append(str(item.get("role") or "reference").lower())
        return {"images": images, "roles": roles}

    @staticmethod
    def _resolved_images_and_roles(request: dict[str, Any]) -> tuple[list[Any], list[str]]:
        raw_items = sorted(
            [item for item in (request.get("input_images") or []) if isinstance(item, dict)],
            key=lambda item: int(item.get("order") or 0),
        )
        roles = [str(item.get("role") or "reference").lower() for item in raw_items]

        resolved = list(request.get("resolved_input_images") or [])
        if resolved:
            # Les images résolues sont construites dans le même ordre par le runtime.
            if len(roles) < len(resolved):
                roles.extend(["reference"] * (len(resolved) - len(roles)))
            return resolved, roles[: len(resolved)]

        # Fallback uniquement pour les tests unitaires / appels directs de l'adapter.
        # En production, RuntimeManager résout les asset_id en vraies images avant ici.
        images = [item.get("asset_id") for item in raw_items if item.get("asset_id") is not None]
        return images, roles[: len(images)]

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        metadata = runtime.get("metadata") or {}
        if self._is_wan(metadata):
            import torch
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

            vae = AutoencoderKLWan.from_pretrained(
                snapshot,
                subfolder="vae",
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            return WanImageToVideoPipeline.from_pretrained(
                snapshot,
                vae=vae,
                local_files_only=True,
                torch_dtype=settings.get("torch_dtype"),
            )

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

    def generate(
        self,
        pipeline: Any,
        runtime: Any,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = runtime.get("metadata") or {}
        is_wan = self._is_wan(metadata)

        images, roles = self._resolved_images_and_roles(request)
        if not images:
            raise RuntimeError("Une image est obligatoire pour IMAGE_TO_VIDEO.")

        image = images[0]
        end_image = None
        for index, role in enumerate(roles):
            if role in {"end", "end_frame"} and index < len(images):
                end_image = images[index]
                break
        if end_image is None and len(images) > 1:
            end_image = images[-1]

        width = int(request.get("width") or 512)
        height = int(request.get("height") or 320)
        fps = int(request.get("fps") or 24)
        frames = request.get("frames")
        steps = request.get("steps")
        guidance = request.get("guidance_scale")

        if is_wan:
            if not steps or int(steps) <= 4:
                steps = 50
            if guidance is None or float(guidance) <= 0.0:
                guidance = 5.0

        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "image": image,
            "images": images,
            "end_image": end_image,
            "image_roles": roles,
            "keyframes": images,
            "height": height,
            "width": width,
            "num_frames": frames,
            "video_length": frames,
            "decode_chunk_size": request.get("decode_chunk_size"),
            "fps": fps,
            "num_inference_steps": steps or 4,
            "guidance_scale": guidance if guidance is not None else 0.0,
            "generator": runtime.get("generator"),
        }

        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        log_diffusers_call(pipeline, filtered, str(request.get("capability") or "IMAGE_TO_VIDEO"))
        output = pipeline(**filtered)
        return {
            "frames": getattr(output, "frames", []),
            "width": width,
            "height": height,
            "fps": fps,
        }
