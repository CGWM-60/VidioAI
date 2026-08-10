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
        "class_name": None,
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
    metadata["class_name"] = model_index.get("_class_name") or config.get("_class_name")
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
    class_name = str(metadata.get("class_name") or "").lower()
    architecture_tokens = [str(value).lower() for value in (metadata.get("architectures") or [])]
    file_tokens = {str(name).lower() for name in metadata["files"]}

    if any(token in tag_values for token in {"text-to-image", "stable-diffusion", "diffusion"}):
        capabilities.append("TEXT_TO_IMAGE")
    if any(token in tag_values for token in {"image-to-image", "img2img", "inpainting"}):
        capabilities.append("IMAGE_TO_IMAGE")
        capabilities.append("TEXT_TO_IMAGE")
    if any(token in tag_values for token in {"inpainting", "image-inpainting"}) or "inpaint" in class_name:
        capabilities.extend(["INPAINTING", "IMAGE_TO_IMAGE"])
    if any(token in tag_values for token in {"outpainting", "image-outpainting"}) or "outpaint" in class_name:
        capabilities.extend(["OUTPAINTING", "IMAGE_TO_IMAGE"])
    if any(token in tag_values for token in {"image-variation", "variation"}) or "variation" in class_name:
        capabilities.extend(["IMAGE_VARIATION", "IMAGE_TO_IMAGE"])
    if any(token in tag_values for token in {"super-resolution", "upscale", "image-upscale"}) or "upscale" in class_name:
        capabilities.extend(["IMAGE_UPSCALE", "IMAGE_TO_IMAGE"])
    if any(token in tag_values for token in {"controlnet", "controlled-image-generation"}) or "control" in class_name:
        capabilities.extend(["CONTROLLED_IMAGE_GENERATION", "IMAGE_TO_IMAGE"])
    if any(token in tag_values for token in {"text-to-video", "video-generation", "video"}):
        capabilities.append("TEXT_TO_VIDEO")
    if any(token in tag_values for token in {"image-to-video", "img2vid"}):
        capabilities.append("IMAGE_TO_VIDEO")
        capabilities.append("MULTI_IMAGE_TO_VIDEO")
    if any(token in tag_values for token in {"video-to-video", "vid2vid"}):
        capabilities.append("VIDEO_TO_VIDEO")
    if any(token in tag_values for token in {"video-inpainting", "inpainting-video"}) or (
        "inpaint" in class_name and "video" in class_name
    ):
        capabilities.extend(["VIDEO_INPAINTING", "VIDEO_TO_VIDEO"])
    if any(token in tag_values for token in {"video-upscale", "video-super-resolution"}) or (
        "upscale" in class_name and "video" in class_name
    ):
        capabilities.extend(["VIDEO_UPSCALE", "VIDEO_TO_VIDEO"])

    if "ltx" in class_name or any("ltx" in token for token in architecture_tokens):
        capabilities.extend(["IMAGE_TO_VIDEO", "START_END_IMAGE_TO_VIDEO"])
    if "cogvideo" in class_name or any("cogvideo" in token for token in architecture_tokens):
        capabilities.extend(["IMAGE_TO_VIDEO", "MULTI_IMAGE_TO_VIDEO", "KEYFRAMES_TO_VIDEO"])
    if any("mask" in token for token in file_tokens):
        capabilities.append("INPAINTING")

    if not capabilities:
        if any(token in {"stable-diffusion", "sdxl"} for token in {str(metadata["model_type"] or "").lower()}):
            capabilities.append("TEXT_TO_IMAGE")
        if any(token in {"ltx", "wan", "skyreels", "cogvideox"} for token in {str(metadata["model_type"] or "").lower()}):
            capabilities.append("TEXT_TO_VIDEO")

    # Dédoublonnage stable
    seen = set()
    deduplicated: list[str] = []
    for capability in capabilities:
        if capability in seen:
            continue
        seen.add(capability)
        deduplicated.append(capability)

    metadata["capabilities"] = deduplicated
    return metadata
