"""Détection conservative des capacités depuis métadonnées, signatures et blocks Modular."""

from __future__ import annotations

import inspect
from typing import Any


CAPABILITY_ORDER = [
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


class CapabilityResolver:
    TAG_MAP = {
        "text-to-image": "TEXT_TO_IMAGE",
        "image-to-image": "IMAGE_TO_IMAGE",
        "image-inpainting": "INPAINTING",
        "inpainting": "INPAINTING",
        "outpainting": "OUTPAINTING",
        "image-variation": "IMAGE_VARIATION",
        "image-upscale": "IMAGE_UPSCALE",
        "super-resolution": "IMAGE_UPSCALE",
        "controlled-image-generation": "CONTROLLED_IMAGE_GENERATION",
        "text-to-video": "TEXT_TO_VIDEO",
        "video-generation": "TEXT_TO_VIDEO",
        "image-to-video": "IMAGE_TO_VIDEO",
        "multi-image-to-video": "MULTI_IMAGE_TO_VIDEO",
        "reference-to-video": "MULTI_IMAGE_TO_VIDEO",
        "reference-image-to-video": "MULTI_IMAGE_TO_VIDEO",
        "start-end-image-to-video": "START_END_IMAGE_TO_VIDEO",
        "first-last-frame-to-video": "START_END_IMAGE_TO_VIDEO",
        "keyframes-to-video": "KEYFRAMES_TO_VIDEO",
        "video-to-video": "VIDEO_TO_VIDEO",
        "video-inpainting": "VIDEO_INPAINTING",
        "video-upscale": "VIDEO_UPSCALE",
    }

    @staticmethod
    def _modular_input_names(
        pipeline_or_class: Any | None,
    ) -> set[str]:
        if pipeline_or_class is None:
            return set()
        blocks = getattr(
            pipeline_or_class,
            "blocks",
            None,
        )
        if blocks is None:
            return set()
        values: set[str] = set()
        for item in getattr(blocks, "inputs", []) or []:
            name = getattr(item, "name", None)
            if isinstance(name, str) and name:
                values.add(name)
            elif isinstance(item, str) and item:
                values.add(item)
        return values

    @staticmethod
    def signature_parameters(
        pipeline_or_class: Any | None,
    ) -> set[str]:
        modular = CapabilityResolver._modular_input_names(
            pipeline_or_class
        )
        if modular:
            return modular

        if pipeline_or_class is None:
            return set()
        target = getattr(
            pipeline_or_class,
            "__call__",
            pipeline_or_class,
        )
        try:
            return {
                name
                for name, parameter in inspect.signature(
                    target
                ).parameters.items()
                if name
                not in {"self", "args", "kwargs"}
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            }
        except (TypeError, ValueError):
            return set()

    @staticmethod
    def _metadata_tokens(
        metadata: dict[str, Any],
    ) -> set[str]:
        values: list[Any] = [
            metadata.get("pipeline_tag"),
            metadata.get("class_name"),
            metadata.get("library_name"),
            *(
                metadata.get("raw_tags")
                or metadata.get("tags")
                or []
            ),
            *(metadata.get("architectures") or []),
            *(metadata.get("base_models") or []),
        ]
        return {
            str(value).strip().lower()
            for value in values
            if str(value or "").strip()
        }

    def resolve(
        self,
        metadata: dict[str, Any],
        pipeline_or_class: Any | None = None,
    ) -> list[str]:
        runtime = self.runtime_capabilities(
            pipeline_or_class
        )
        if runtime:
            return runtime
        declared = self.declared_capabilities(metadata)
        if declared:
            return declared
        return self._display_hints(metadata)

    def declared_capabilities(
        self,
        metadata: dict[str, Any],
    ) -> list[str]:
        tokens = self._metadata_tokens(metadata)
        capabilities = {
            self.TAG_MAP[token]
            for token in tokens
            if token in self.TAG_MAP
        }
        return [
            value
            for value in CAPABILITY_ORDER
            if value in capabilities
        ]

    def runtime_capabilities(
        self,
        pipeline_or_class: Any | None,
    ) -> list[str]:
        params = self.signature_parameters(
            pipeline_or_class
        )
        capabilities: set[str] = set()

        has_prompt = (
            "prompt" in params
            or "prompt_embeds" in params
        )
        has_frames = bool(
            {
                "num_frames",
                "video_length",
                "frames",
            }
            & params
        )
        has_start_image = bool(
            {
                "image",
                "start_image",
                "first_image",
                "first_frame",
            }
            & params
        )
        has_end_image = bool(
            {
                "last_image",
                "end_image",
                "last_frame",
            }
            & params
        )
        has_images = bool(
            {
                "images",
                "reference_images",
                "conditioning_images",
            }
            & params
        )
        has_keyframes = "keyframes" in params
        has_video = bool(
            {
                "video",
                "input_video",
                "reference_video",
            }
            & params
        )
        has_mask = (
            "mask_image" in params
            or "mask" in params
        )

        # Pour les blocks Modular, num_frames peut être calculé en interne. Une
        # sortie vidéo est aussi prouvée par des noms de contexte vidéo dans les
        # inputs. On reste conservateur pour les pipelines standards.
        modular = bool(
            self._modular_input_names(
                pipeline_or_class
            )
        )
        video_context = (
            has_frames
            or modular
            and bool(
                {
                    "fps",
                    "duration",
                    "duration_seconds",
                    "video",
                    "reference_video",
                    "start_image",
                    "first_image",
                    "first_frame",
                    "reference_images",
                    "last_image",
                    "end_image",
                }
                & params
            )
        )

        if has_prompt and video_context:
            capabilities.add("TEXT_TO_VIDEO")
        if has_start_image and video_context:
            capabilities.add("IMAGE_TO_VIDEO")
        if has_images and video_context:
            capabilities.add("MULTI_IMAGE_TO_VIDEO")
        if (
            has_start_image
            and has_end_image
            and video_context
        ):
            capabilities.add(
                "START_END_IMAGE_TO_VIDEO"
            )
        if has_keyframes and video_context:
            capabilities.add(
                "KEYFRAMES_TO_VIDEO"
            )
        if has_video:
            capabilities.add("VIDEO_TO_VIDEO")
        if has_video and has_mask:
            capabilities.add("VIDEO_INPAINTING")
        if has_start_image and has_mask and not video_context:
            capabilities.update(
                {
                    "IMAGE_TO_IMAGE",
                    "INPAINTING",
                }
            )
        if "control_image" in params:
            capabilities.add(
                "CONTROLLED_IMAGE_GENERATION"
            )
        if has_start_image and not video_context:
            capabilities.add("IMAGE_TO_IMAGE")
        if (
            has_prompt
            and not video_context
            and not has_start_image
            and not has_video
        ):
            capabilities.add("TEXT_TO_IMAGE")

        return [
            value
            for value in CAPABILITY_ORDER
            if value in capabilities
        ]

    def _display_hints(
        self,
        metadata: dict[str, Any],
    ) -> list[str]:
        hint = " ".join(
            str(value).lower()
            for value in (
                metadata.get("class_name"),
                *(
                    metadata.get("architectures")
                    or []
                ),
            )
            if value
        )
        capabilities: set[str] = set()
        if "video" in hint:
            capabilities.add("TEXT_TO_VIDEO")
        elif "image" in hint:
            capabilities.add("TEXT_TO_IMAGE")
        return [
            value
            for value in CAPABILITY_ORDER
            if value in capabilities
        ]

    def describe(
        self,
        metadata: dict[str, Any],
        pipeline_or_class: Any | None = None,
    ) -> dict[str, list[str]]:
        declared = self.declared_capabilities(metadata)
        runtime = self.runtime_capabilities(
            pipeline_or_class
        )
        display = (
            runtime
            or declared
            or self._display_hints(metadata)
        )
        return {
            "declared_capabilities": declared,
            "runtime_capabilities": runtime,
            "display_capabilities": display,
        }
