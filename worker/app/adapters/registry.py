from __future__ import annotations

from typing import Any

from .text_to_image import TextToImageAdapter
from .image_to_image import ImageToImageAdapter
from .text_to_video import TextToVideoAdapter
from .image_to_video import ImageToVideoAdapter
from .video_to_video import VideoToVideoAdapter
from .generic_diffusers import GenericDiffusersAdapter


class PipelineRegistry:
    def __init__(self) -> None:
        self._adapters = [
            TextToImageAdapter(),
            ImageToImageAdapter(),
            TextToVideoAdapter(),
            ImageToVideoAdapter(),
            VideoToVideoAdapter(),
            # Fallback en dernier: les adapters spécialisés restent prioritaires.
            GenericDiffusersAdapter(),
        ]

    def capabilities(self) -> list[str]:
        return [capability for adapter in self._adapters for capability in adapter.capabilities()]

    def select_for_capability(self, metadata: dict[str, Any], capability: str):
        # La pipeline Diffusers reelle est le chemin principal. Les adapters
        # specialises ne sont que des fallbacks pour les pipelines qui ne sont
        # pas exposées par la version installee de Diffusers.
        generic = self._adapters[-1]
        if generic.supports_model(metadata):
            discovered = generic.supported_capabilities(metadata)
            # Une signature opaque est chargee de facon provisoire afin de
            # pouvoir introspecter l'instance. Cela n'ajoute pas la capability
            # aux metadonnees et ne la publie pas dans le catalogue.
            if not discovered or capability in discovered:
                return generic
        for adapter in self._adapters[:-1]:
            if capability in adapter.supported_capabilities(metadata) and adapter.supports_model(metadata):
                return adapter
        return None

    def select_for_model(self, metadata: dict[str, Any]):
        for adapter in self._adapters:
            if adapter.supports_model(metadata):
                return adapter
        return None
