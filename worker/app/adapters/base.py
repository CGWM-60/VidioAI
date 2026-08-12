from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


def log_diffusers_call(pipeline: Any, kwargs: dict[str, Any], capability: str) -> None:
    """Journalise le contrat d'appel sans contenu utilisateur ni asset privé."""
    public_scalars = {
        "width", "height", "num_frames", "video_length", "fps",
        "num_inference_steps", "guidance_scale", "true_cfg_scale",
        "strength", "max_sequence_length", "decode_chunk_size",
    }
    parameters = {
        key: value if key in public_scalars and isinstance(value, (int, float, bool, str))
        else type(value).__name__
        for key, value in sorted(kwargs.items())
    }
    logger.info(
        "DIFFUSERS_CALL %s",
        json.dumps(
            {
                "pipeline": type(pipeline).__name__,
                "capability": capability,
                "parameters": parameters,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


class RuntimeAdapter(ABC):
    @abstractmethod
    def capabilities(self) -> list[str]:
        raise NotImplementedError

    def supported_capabilities(self, metadata: dict[str, Any]) -> list[str]:
        del metadata
        return self.capabilities()

    @abstractmethod
    def supports_model(self, metadata: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def estimate_resources(self, metadata: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load(self, snapshot: str, settings: dict[str, Any], runtime: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def unload(self, pipeline: Any, runtime: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, pipeline: Any, runtime: Any, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def input_profile(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "min_input_images": 1,
            "max_input_images": 1,
            "supported_image_roles": [],
            "supports_start_end_frames": False,
            "supports_reference_images": False,
            "supports_keyframes": False,
        }

    def prepare_pipeline_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"images": [], "roles": []}
