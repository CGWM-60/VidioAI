from __future__ import annotations

from typing import Any

from .text_to_image import TextToImageAdapter
from .image_to_image import ImageToImageAdapter
from .text_to_video import TextToVideoAdapter
from .image_to_video import ImageToVideoAdapter
from .video_to_video import VideoToVideoAdapter
from .minimax_h3 import MiniMaxH3Adapter
from .modular_diffusers import ModularDiffusersAdapter
from .generic_diffusers import GenericDiffusersAdapter


class PipelineRegistry:
    def __init__(self) -> None:
        # Les runtimes d'architecture passent avant le runtime Modular générique.
        # Le choix est fait sur les métadonnées/classes, jamais sur repo_id.
        self._architecture_adapters = [
            MiniMaxH3Adapter(),
        ]
        self._modular = ModularDiffusersAdapter()
        self._generic = GenericDiffusersAdapter()
        self._specialized = [
            TextToImageAdapter(),
            ImageToImageAdapter(),
            TextToVideoAdapter(),
            ImageToVideoAdapter(),
            VideoToVideoAdapter(),
        ]
        self._adapters = [
            *self._architecture_adapters,
            self._modular,
            *self._specialized,
            self._generic,
        ]

    def capabilities(self) -> list[str]:
        return [
            capability
            for adapter in self._adapters
            for capability in adapter.capabilities()
        ]

    def _architecture_adapter(
        self,
        metadata: dict[str, Any],
    ):
        for adapter in self._architecture_adapters:
            if adapter.supports_model(metadata):
                return adapter
        return None

    def select_for_capability(
        self,
        metadata: dict[str, Any],
        capability: str,
    ):
        architecture = self._architecture_adapter(metadata)
        if architecture is not None:
            return (
                architecture
                if capability
                in architecture.supported_capabilities(metadata)
                else None
            )

        if self._modular.supports_model(metadata):
            discovered = self._modular.supported_capabilities(
                metadata
            )
            if not discovered or capability in discovered:
                return self._modular
            return None

        if self._generic.supports_model(metadata):
            discovered = self._generic.supported_capabilities(
                metadata
            )
            if not discovered or capability in discovered:
                return self._generic

        for adapter in self._specialized:
            if (
                capability
                in adapter.supported_capabilities(metadata)
                and adapter.supports_model(metadata)
            ):
                return adapter
        return None

    def select_for_model(
        self,
        metadata: dict[str, Any],
    ):
        architecture = self._architecture_adapter(metadata)
        if architecture is not None:
            return architecture
        if self._modular.supports_model(metadata):
            return self._modular
        for adapter in [
            self._generic,
            *self._specialized,
        ]:
            if adapter.supports_model(metadata):
                return adapter
        return None
