from __future__ import annotations

from typing import Any

from .text_to_image import TextToImageAdapter
from .image_to_image import ImageToImageAdapter
from .text_to_video import TextToVideoAdapter
from .image_to_video import ImageToVideoAdapter
from .video_to_video import VideoToVideoAdapter


class PipelineRegistry:
    def __init__(self) -> None:
        self._adapters = [
            TextToImageAdapter(),
            ImageToImageAdapter(),
            TextToVideoAdapter(),
            ImageToVideoAdapter(),
            VideoToVideoAdapter(),
        ]

    def capabilities(self) -> list[str]:
        return [capability for adapter in self._adapters for capability in adapter.capabilities()]

    def select_for_capability(self, metadata: dict[str, Any], capability: str):
        for adapter in self._adapters:
            if capability in adapter.capabilities() and adapter.supports_model(metadata):
                return adapter
        return None

    def select_for_model(self, metadata: dict[str, Any]):
        for adapter in self._adapters:
            if adapter.supports_model(metadata):
                return adapter
        return None
