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


def _extract_base_models(
    model_index: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    values: list[str] = []

    for source in (model_index, config):
        for key in (
            "base_model",
            "base_models",
            "base",
            "base_repo",
            "base_repository",
        ):
            candidate = source.get(key)
            if isinstance(candidate, str) and candidate.strip():
                values.append(candidate.strip())
            elif isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item.strip():
                        values.append(item.strip())

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _collect_architectures(
    root: Path,
    model_index: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    values: list[str] = []

    for architecture in config.get("architectures") or []:
        if isinstance(architecture, str) and architecture:
            values.append(architecture)

    for key, component in model_index.items():
        if key.startswith("_"):
            continue
        if isinstance(component, list) and len(component) >= 2:
            module_name = component[0]
            class_name = component[1]
            if isinstance(module_name, str) and module_name:
                values.append(module_name)
            if isinstance(class_name, str) and class_name:
                values.append(class_name)

    for config_path in sorted(root.rglob("config.json")):
        payload = _read_json(config_path)
        for architecture in payload.get("architectures") or []:
            if isinstance(architecture, str) and architecture:
                values.append(architecture)

    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _infer_library_name(
    model_index: dict[str, Any],
    config: dict[str, Any],
) -> str | None:
    direct = model_index.get("library_name") or config.get("library_name")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    component_libraries: set[str] = set()
    for key, component in model_index.items():
        if key.startswith("_"):
            continue
        if (
            isinstance(component, list)
            and len(component) >= 2
            and isinstance(component[0], str)
        ):
            component_libraries.add(component[0].strip().lower())

    if "diffusers" in component_libraries:
        return "diffusers"
    return None


def _is_wan_ti2v(
    model_index: dict[str, Any],
    class_name: str,
    architectures: list[str],
    base_models: list[str],
) -> bool:
    class_lower = class_name.lower()
    architecture_tokens = {str(value).lower() for value in architectures}

    is_wan = (
        "wanpipeline" in class_lower
        or "wantransformer3dmodel" in architecture_tokens
    )
    if not is_wan:
        return False

    # Les dérivés/quantifiés peuvent ne pas recopier expand_timesteps dans leur
    # model_index mais déclarer explicitement le modèle TI2V-5B comme base.
    if any(
        "wan2.2-ti2v-5b" in str(base_model).lower()
        or "ti2v-5b" in str(base_model).lower()
        for base_model in base_models
    ):
        return True

    expand_timesteps = model_index.get("expand_timesteps") is True
    transformer_2 = model_index.get("transformer_2")
    no_second_transformer = (
        transformer_2 is None
        or (
            isinstance(transformer_2, list)
            and all(item is None for item in transformer_2)
        )
    )
    return expand_timesteps and no_second_transformer


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
    metadata["library_name"] = _infer_library_name(model_index, config)
    metadata["class_name"] = model_index.get("_class_name") or config.get("_class_name")
    metadata["architectures"] = _collect_architectures(root, model_index, config)
    metadata["model_type"] = config.get("model_type") or model_index.get("model_type")

    tags_source = model_index.get("tags") or config.get("tags") or []
    metadata["raw_tags"] = [str(tag).lower() for tag in tags_source if str(tag).strip()]
    metadata["base_models"] = _extract_base_models(model_index, config)
    metadata["files"] = [
        str(path.relative_to(root)).lower()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]

    tag_values = set(metadata["raw_tags"])
    pipeline_tag = str(metadata["pipeline_tag"] or "").lower()
    library_name = str(metadata["library_name"] or "").lower()
    class_name = str(metadata["class_name"] or "").lower()
    architecture_tokens = [str(value).lower() for value in metadata["architectures"]]
    base_model_tokens = [str(value).lower() for value in metadata["base_models"]]

    if pipeline_tag:
        tag_values.add(pipeline_tag)
    if library_name:
        tag_values.add(library_name)
    if class_name:
        tag_values.add(class_name)
    tag_values.update(architecture_tokens)
    tag_values.update(base_model_tokens)

    capabilities: list[str] = []

    # Images
    if any(token in tag_values for token in {"text-to-image", "stable-diffusion", "diffusion"}):
        capabilities.append("TEXT_TO_IMAGE")
    if any(token in tag_values for token in {"image-to-image", "img2img", "inpainting"}):
        capabilities.extend(["IMAGE_TO_IMAGE", "TEXT_TO_IMAGE"])
    if "inpainting" in tag_values or "image-inpainting" in tag_values or "inpaint" in class_name:
        capabilities.extend(["INPAINTING", "IMAGE_TO_IMAGE"])
    if "outpainting" in tag_values or "image-outpainting" in tag_values or "outpaint" in class_name:
        capabilities.extend(["OUTPAINTING", "IMAGE_TO_IMAGE"])
    if "image-variation" in tag_values or "variation" in tag_values or "variation" in class_name:
        capabilities.extend(["IMAGE_VARIATION", "IMAGE_TO_IMAGE"])
    if (
        "super-resolution" in tag_values
        or "upscale" in tag_values
        or "image-upscale" in tag_values
        or "upscale" in class_name
    ):
        capabilities.extend(["IMAGE_UPSCALE", "IMAGE_TO_IMAGE"])
    if "controlnet" in tag_values or "controlled-image-generation" in tag_values or "control" in class_name:
        capabilities.extend(["CONTROLLED_IMAGE_GENERATION", "IMAGE_TO_IMAGE"])

    # Vidéos génériques
    if any(token in tag_values for token in {"text-to-video", "video-generation"}):
        capabilities.append("TEXT_TO_VIDEO")
    if any(token in tag_values for token in {"image-to-video", "img2vid"}):
        capabilities.append("IMAGE_TO_VIDEO")
    if "multi-image-to-video" in tag_values:
        capabilities.append("MULTI_IMAGE_TO_VIDEO")
    if "start-end-image-to-video" in tag_values:
        capabilities.append("START_END_IMAGE_TO_VIDEO")
    if "keyframes-to-video" in tag_values:
        capabilities.append("KEYFRAMES_TO_VIDEO")
    if any(token in tag_values for token in {"video-to-video", "vid2vid"}):
        capabilities.append("VIDEO_TO_VIDEO")

    # Diffusers / Wan
    if library_name == "diffusers":
        wan_ti2v = _is_wan_ti2v(
            model_index,
            class_name,
            metadata["architectures"],
            metadata["base_models"],
        )

        if class_name == "wanpipeline":
            capabilities.append("TEXT_TO_VIDEO")
            if wan_ti2v:
                capabilities.append("IMAGE_TO_VIDEO")
        elif class_name == "wanimagetovideopipeline":
            capabilities.append("IMAGE_TO_VIDEO")
        elif any("wantransformer3dmodel" in value for value in architecture_tokens):
            capabilities.append("TEXT_TO_VIDEO")
            if wan_ti2v:
                capabilities.append("IMAGE_TO_VIDEO")

        if any("ltx" in value for value in [class_name, *architecture_tokens]):
            if "image-to-video" in tag_values:
                capabilities.append("IMAGE_TO_VIDEO")
        if any("cogvideo" in value for value in [class_name, *architecture_tokens]):
            if "image-to-video" in tag_values:
                capabilities.append("IMAGE_TO_VIDEO")

    if (
        "video-inpainting" in tag_values
        or "inpainting-video" in tag_values
        or ("video" in class_name and "inpaint" in class_name)
    ):
        capabilities.extend(["VIDEO_INPAINTING", "VIDEO_TO_VIDEO"])

    if (
        "video-upscale" in tag_values
        or "video-super-resolution" in tag_values
        or ("video" in class_name and "upscale" in class_name)
    ):
        capabilities.extend(["VIDEO_UPSCALE", "VIDEO_TO_VIDEO"])

    if not capabilities:
        model_type = str(metadata["model_type"] or "").lower()
        if model_type in {"stable-diffusion", "sdxl"}:
            capabilities.append("TEXT_TO_IMAGE")
        if model_type in {"ltx", "wan", "skyreels", "cogvideox"}:
            capabilities.append("TEXT_TO_VIDEO")

    seen: set[str] = set()
    deduplicated: list[str] = []
    for capability in capabilities:
        if capability in seen:
            continue
        seen.add(capability)
        deduplicated.append(capability)

    metadata["capabilities"] = deduplicated
    return metadata
