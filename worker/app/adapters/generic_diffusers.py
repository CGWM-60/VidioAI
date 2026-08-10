from __future__ import annotations

import inspect
from typing import Any

from .base import RuntimeAdapter


class GenericDiffusersAdapter(RuntimeAdapter):
    """Fallback runtime adapter for Diffusers pipelines discovered at runtime.

    Specialized adapters remain priority in the registry. This adapter is used
    only when a compatible Diffusers pipeline class exists but no dedicated
    adapter matches the model-specific capability profile.
    """

    def capabilities(self) -> list[str]:
        return [
            "TEXT_TO_IMAGE",
            "IMAGE_TO_IMAGE",
            "INPAINTING",
            "OUTPAINTING",
            "IMAGE_VARIATION",
            "IMAGE_UPSCALE",
            "CONTROLLED_IMAGE_GENERATION",
            "TEXT_TO_VIDEO",
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
            "VIDEO_TO_VIDEO",
            "VIDEO_INPAINTING",
            "VIDEO_UPSCALE",
        ]

    def supported_capabilities(self, metadata: dict[str, Any]) -> list[str]:
        capabilities = [
            str(value).upper()
            for value in (metadata.get("capabilities") or [])
            if isinstance(value, str)
        ]
        if capabilities:
            return capabilities

        pipeline_tag = str(metadata.get("pipeline_tag") or "").lower()
        class_name = str(metadata.get("class_name") or "").lower()
        if "video" in pipeline_tag or "video" in class_name:
            return ["TEXT_TO_VIDEO"]
        return ["TEXT_TO_IMAGE"]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        if str(metadata.get("library_name") or "").lower() != "diffusers":
            return False
        class_name = str(metadata.get("class_name") or "").strip()
        if not class_name:
            return False
        try:
            import diffusers
        except Exception:
            return False
        return getattr(diffusers, class_name, None) is not None

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        class_name = str(metadata.get("class_name") or "").lower()
        if "video" in class_name or "wan" in class_name:
            return {"vram_bytes": 20 * 1024 * 1024 * 1024, "ram_bytes": 20 * 1024 * 1024 * 1024}
        return {"vram_bytes": 10 * 1024 * 1024 * 1024, "ram_bytes": 10 * 1024 * 1024 * 1024}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import DiffusionPipeline
        import diffusers

        class_name = str(runtime.get("class_name") or "")
        pipeline_cls = getattr(diffusers, class_name, None) if class_name else None
        if pipeline_cls is not None:
            try:
                return pipeline_cls.from_pretrained(
                    snapshot,
                    local_files_only=True,
                    use_safetensors=True,
                    torch_dtype=settings.get("torch_dtype"),
                )
            except Exception:
                pass

        return DiffusionPipeline.from_pretrained(
            snapshot,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=settings.get("torch_dtype"),
        )

    def unload(self, pipeline: Any, runtime: Any) -> None:
        del pipeline

    @staticmethod
    def _prepare_pipeline_inputs(request: dict[str, Any]) -> tuple[list[Any], list[str]]:
        resolved_images = request.get("resolved_input_images") or []
        if resolved_images:
            roles = [
                str(item.get("role") or "reference").lower()
                for item in (request.get("input_images") or [])
                if isinstance(item, dict)
            ]
            return list(resolved_images), roles

        images = []
        roles = []
        for item in request.get("input_images") or []:
            if not isinstance(item, dict):
                continue
            source = item.get("asset_id")
            if source is None:
                continue
            images.append(source)
            roles.append(str(item.get("role") or "reference").lower())
        return images, roles

    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        capability = str(request.get("capability") or "TEXT_TO_IMAGE").upper()
        images, roles = self._prepare_pipeline_inputs(request)

        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "negative_prompt": request.get("negative_prompt"),
            "width": request.get("width", 512),
            "height": request.get("height", 512),
            "num_inference_steps": request.get("steps", 4),
            "guidance_scale": request.get("guidance_scale", 0.0),
            "generator": runtime.get("generator"),
            "num_frames": request.get("frames"),
            "fps": request.get("fps"),
            "image": request.get("input_image"),
            "video": request.get("input_video"),
            "frames": request.get("input_frames"),
            "mask_image": request.get("mask_image"),
            "control_image": request.get("control_image"),
            "strength": request.get("strength"),
        }

        if capability in {
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        } and images:
            kwargs["image"] = images[0]
            kwargs["images"] = images
            kwargs["keyframes"] = images
            kwargs["image_roles"] = roles
            end_index = None
            for index, role in enumerate(roles):
                if role in {"end", "end_frame"}:
                    end_index = index
                    break
            if end_index is not None:
                kwargs["end_image"] = images[end_index]
            elif len(images) > 1:
                kwargs["end_image"] = images[-1]

        accepted = set(inspect.signature(pipeline.__call__).parameters)
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted and value is not None
        }
        output = pipeline(**filtered)
        frames = getattr(output, "frames", [])
        if frames:
            return {"frames": frames}
        return {"images": getattr(output, "images", [])}
