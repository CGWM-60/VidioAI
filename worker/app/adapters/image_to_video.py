from __future__ import annotations

from typing import Any

from .base import RuntimeAdapter


class ImageToVideoAdapter(RuntimeAdapter):
    def capabilities(self) -> list[str]:
        return ["IMAGE_TO_VIDEO"]

    def supports_model(self, metadata: dict[str, Any]) -> bool:
        capabilities = set(metadata.get("capabilities", []))
        return "IMAGE_TO_VIDEO" in capabilities or "img2vid" in str(metadata.get("pipeline_tag") or "").lower()

    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {"vram_bytes": 18 * 1024 * 1024 * 1024, "ram_bytes": 18 * 1024 * 1024 * 1024}

    def input_profile(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        model_name = str((metadata or {}).get("class_name") or "").lower()
        if "ltx" in model_name:
            return {
                "min_input_images": 1,
                "max_input_images": 2,
                "supported_image_roles": ["start_frame", "end_frame"],
                "supports_start_end_frames": True,
                "supports_reference_images": False,
                "supports_keyframes": False,
            }
        if "cogvideox" in model_name or "cog" in model_name:
            return {
                "min_input_images": 1,
                "max_input_images": 8,
                "supported_image_roles": ["reference", "keyframe"],
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
        raw_images = request.get("input_images") or []
        images = []
        roles = []
        for item in raw_images:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("asset_id")
            if asset_id is None:
                continue
            images.append(asset_id)
            roles.append(item.get("role") or "reference")
        return {"images": images, "roles": roles}

    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        from diffusers import AutoPipelineForImage2Video

        return AutoPipelineForImage2Video.from_pretrained(
            snapshot,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=settings.get("torch_dtype"),
        )

    def unload(self, pipeline: Any, runtime: Any) -> None:
        del pipeline

    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        prepared = self.prepare_pipeline_inputs(request)
        kwargs: dict[str, Any] = {
            "prompt": request.get("prompt"),
            "height": request.get("height", 320),
            "width": request.get("width", 512),
            "num_inference_steps": request.get("steps", 4),
            "guidance_scale": request.get("guidance_scale", 0.0),
            "generator": runtime.get("generator"),
        }
        if prepared["images"]:
            if len(prepared["images"]) == 1:
                kwargs["image"] = prepared["images"][0]
            else:
                kwargs["image"] = prepared["images"][0]
                kwargs["end_image"] = prepared["images"][-1]
                kwargs["images"] = prepared["images"]
                kwargs["image_roles"] = prepared["roles"]
        else:
            kwargs["image"] = request["input_image"]
        output = pipeline(**kwargs)
        return {"frames": getattr(output, "frames", [])}
