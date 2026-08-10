from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_base_models(model_index: dict[str, Any], config: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for source in (model_index, config):
        for key in ("base_model", "base_models", "base", "base_repo", "base_repository"):
            candidate = source.get(key)
            if isinstance(candidate, str) and candidate.strip():
                values.append(candidate.strip())
            elif isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item.strip():
                        values.append(item.strip())

    seen = set()
    deduplicated: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(value)
    return deduplicated


def _collect_architectures(root: Path, model_index: dict[str, Any], config: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for architecture in config.get("architectures") or []:
        if isinstance(architecture, str) and architecture:
            values.append(architecture)

    for key, component in model_index.items():
        if key.startswith("_"):
            continue
        if isinstance(component, list) and len(component) >= 2:
            module_name, class_name = component[0], component[1]
            if isinstance(module_name, str) and module_name:
                values.append(module_name)
            if isinstance(class_name, str) and class_name:
                values.append(class_name)

    for config_path in sorted(root.rglob("config.json")):
        payload = _read_json(config_path)
        for architecture in payload.get("architectures") or []:
            if isinstance(architecture, str) and architecture:
                values.append(architecture)

    seen = set()
    deduplicated: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(value)
    return deduplicated


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
        "base_models": [],
    }
    model_index_path = root / "model_index.json"
    if model_index_path.is_file():
        metadata["model_index"] = _read_json(model_index_path)
    config_path = root / "config.json"
    if config_path.is_file():
        metadata["config"] = _read_json(config_path)
    model_index = metadata["model_index"] or {}
    config = metadata["config"] or {}
    metadata["pipeline_tag"] = model_index.get("pipeline_tag") or config.get("pipeline_tag")
    metadata["library_name"] = model_index.get("library_name") or config.get("library_name")
    metadata["class_name"] = model_index.get("_class_name") or config.get("_class_name")
    metadata["architectures"] = _collect_architectures(root, model_index, config)
    metadata["model_type"] = config.get("model_type") or model_index.get("model_type")
    metadata["raw_tags"] = [str(tag).lower() for tag in (model_index.get("tags") or []) if str(tag).strip()]
    metadata["base_models"] = _extract_base_models(model_index, config)
    metadata["files"] = [str(p.relative_to(root)).lower() for p in sorted(root.rglob("*")) if p.is_file()]

    tags = set(metadata["raw_tags"])
    tag_values = {tag.lower() for tag in tags}
    pipeline_tag = str(metadata["pipeline_tag"] or "").lower()
    if pipeline_tag:
        tag_values.add(pipeline_tag)

    library_name = str(metadata.get("library_name") or "").lower()
    class_name = str(metadata.get("class_name") or "").lower()
    architecture_tokens = [str(value).lower() for value in (metadata.get("architectures") or [])]
    base_model_tokens = [str(value).lower() for value in (metadata.get("base_models") or [])]
    file_tokens = {str(name).lower() for name in metadata["files"]}

    for token in (library_name, class_name):
        if token:
            tag_values.add(token)
    tag_values.update(architecture_tokens)
    tag_values.update(base_model_tokens)
    if any("wan" in token for token in file_tokens):
        tag_values.add("wan")

    capabilities: list[str] = []

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

    if "diffusers" in library_name:
        # Priorité 1-2 : class_name / library_name
        if "wanpipeline" in class_name:
            capabilities.extend(
                [
                    "TEXT_TO_VIDEO",
                    "IMAGE_TO_VIDEO",
                    "MULTI_IMAGE_TO_VIDEO",
                    "START_END_IMAGE_TO_VIDEO",
                    "KEYFRAMES_TO_VIDEO",
                ]
            )

        # Priorité 3 : architectures (y compris sous-configs transformeur)
        if any("wantransformer3dmodel" in token for token in architecture_tokens):
            capabilities.extend(
                [
                    "TEXT_TO_VIDEO",
                    "IMAGE_TO_VIDEO",
                    "MULTI_IMAGE_TO_VIDEO",
                    "START_END_IMAGE_TO_VIDEO",
                    "KEYFRAMES_TO_VIDEO",
                ]
            )

        # Priorité 4-5 : pipeline_tag + tags
        if any(token in tag_values for token in {"text-to-video", "image-to-video", "img2vid", "wan"}):
            capabilities.extend(["TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"])

        # Priorité 6 : base_model metadata
        if any("wan" in token for token in base_model_tokens):
            capabilities.extend(["TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"])

        # Priorité 7 : fichiers présents
        if "model_index.json" in file_tokens and any("transformer" in token for token in file_tokens):
            capabilities.append("TEXT_TO_VIDEO")

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
