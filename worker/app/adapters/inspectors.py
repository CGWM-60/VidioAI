from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def inspect_model_metadata(snapshot: str | Path) -> dict[str, Any]:
    root = Path(snapshot)
    metadata: dict[str, Any] = {
        "capabilities": [],
        "pipeline_tag": None,
        "library_name": None,
        "architectures": [],
        "model_type": None,
        "files": [],
        "model_index": None,
        "config": None,
        "raw_tags": [],
    }
    model_index_path = root / "model_index.json"
    if model_index_path.is_file():
        try:
            metadata["model_index"] = json.loads(model_index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata["model_index"] = {}
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            metadata["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata["config"] = {}
    model_index = metadata["model_index"] or {}
    config = metadata["config"] or {}
    metadata["pipeline_tag"] = model_index.get("pipeline_tag") or config.get("pipeline_tag")
    metadata["library_name"] = model_index.get("library_name") or config.get("library_name")
    metadata["architectures"] = list(config.get("architectures") or [])
    metadata["model_type"] = config.get("model_type") or model_index.get("model_type")
    metadata["raw_tags"] = [str(tag).lower() for tag in (model_index.get("tags") or [])]
    metadata["files"] = [p.name for p in sorted(root.rglob("*")) if p.is_file()]

    tags = set(metadata["raw_tags"])
    tag_values = {tag.lower() for tag in tags}
    pipeline_tag = str(metadata["pipeline_tag"] or "").lower()
    if pipeline_tag:
        tag_values.add(pipeline_tag)

    capabilities: list[str] = []
    if any(token in tag_values for token in {"text-to-image", "stable-diffusion", "diffusion"}):
        capabilities.append("TEXT_TO_IMAGE")
    if any(token in tag_values for token in {"image-to-image", "img2img", "inpainting"}):
        capabilities.append("IMAGE_TO_IMAGE")
        capabilities.append("TEXT_TO_IMAGE")
    if any(token in tag_values for token in {"text-to-video", "video-generation", "video"}):
        capabilities.append("TEXT_TO_VIDEO")
    if any(token in tag_values for token in {"image-to-video", "img2vid"}):
        capabilities.append("IMAGE_TO_VIDEO")
    if any(token in tag_values for token in {"video-to-video", "vid2vid"}):
        capabilities.append("VIDEO_TO_VIDEO")

    if not capabilities:
        if any(token in {"stable-diffusion", "sdxl"} for token in {str(metadata["model_type"] or "").lower()}):
            capabilities.append("TEXT_TO_IMAGE")
        if any(token in {"ltx", "wan", "skyreels", "cogvideox"} for token in {str(metadata["model_type"] or "").lower()}):
            capabilities.append("TEXT_TO_VIDEO")

    metadata["capabilities"] = capabilities
    return metadata
