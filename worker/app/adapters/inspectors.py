from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..capability_resolver import CapabilityResolver
from ..modular_runtime import ModularManifestResolver
from ..pipeline_resolver import PipelineResolver


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_base_models(
    model_index: dict[str, Any],
    modular_index: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    values: list[str] = []

    for source in (model_index, modular_index, config):
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

    for component in ModularManifestResolver.components(modular_index):
        if component.repository:
            values.append(component.repository)

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
    modular_index: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    values: list[str] = []

    for architecture in config.get("architectures") or []:
        if isinstance(architecture, str) and architecture:
            values.append(architecture)

    for source in (model_index,):
        for key, component in source.items():
            if key.startswith("_"):
                continue
            if isinstance(component, list) and len(component) >= 2:
                module_name = component[0]
                class_name = component[1]
                if isinstance(module_name, str) and module_name:
                    values.append(module_name)
                if isinstance(class_name, str) and class_name:
                    values.append(class_name)

    values.extend(ModularManifestResolver.type_hints(modular_index))

    for config_path in sorted(root.rglob("config.json")):
        # Les fichiers de cache/metadata HF ne sont pas des configs runtime.
        if ".cache" in config_path.parts:
            continue
        payload = _read_json(config_path)
        for architecture in payload.get("architectures") or []:
            if isinstance(architecture, str) and architecture:
                values.append(architecture)
        class_name = payload.get("_class_name")
        if isinstance(class_name, str) and class_name:
            values.append(class_name)

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
    modular_index: dict[str, Any],
    config: dict[str, Any],
) -> str | None:
    direct = (
        modular_index.get("library_name")
        or model_index.get("library_name")
        or config.get("library_name")
    )
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    if modular_index:
        # modular_model_index.json est un contrat Diffusers.
        return "diffusers"

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
        "modular_model_index": None,
        "is_modular": False,
        "config": None,
        "raw_tags": [],
        "base_models": [],
        "component_configs": {},
    }

    model_index_path = root / "model_index.json"
    if model_index_path.is_file():
        metadata["model_index"] = _read_json(model_index_path)

    modular_index_path = root / "modular_model_index.json"
    if modular_index_path.is_file():
        metadata["modular_model_index"] = _read_json(
            modular_index_path
        )
        metadata["is_modular"] = True

    config_path = root / "config.json"
    if config_path.is_file():
        metadata["config"] = _read_json(config_path)

    model_index = metadata["model_index"] or {}
    modular_index = metadata["modular_model_index"] or {}
    config = metadata["config"] or {}

    for component_name in (
        "unet",
        "transformer",
        "vae",
        "video_vae",
        "audio_vae",
        "image_encoder",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "audio_encoder",
        "conditioner",
    ):
        component_path = root / component_name / "config.json"
        if component_path.is_file():
            metadata["component_configs"][component_name] = _read_json(
                component_path
            )

    # Les composants ModularPipeline externes sont materialisés dans le
    # snapshot. Leur configuration doit participer à la résolution dtype /
    # quantification exactement comme un composant Diffusers local.
    for (
        component_name,
        component_config,
    ) in ModularManifestResolver.component_config_paths(root).items():
        metadata["component_configs"][component_name] = _read_json(
            component_config
        )

    metadata["pipeline_tag"] = (
        modular_index.get("pipeline_tag")
        or model_index.get("pipeline_tag")
        or config.get("pipeline_tag")
    )
    metadata["library_name"] = _infer_library_name(
        model_index,
        modular_index,
        config,
    )
    metadata["class_name"] = (
        modular_index.get("_class_name")
        or model_index.get("_class_name")
        or config.get("_class_name")
    )
    metadata["architectures"] = _collect_architectures(
        root,
        model_index,
        modular_index,
        config,
    )
    metadata["model_type"] = (
        config.get("model_type")
        or modular_index.get("model_type")
        or model_index.get("model_type")
    )

    tags_source = (
        modular_index.get("tags")
        or model_index.get("tags")
        or config.get("tags")
        or []
    )
    metadata["raw_tags"] = [
        str(tag).lower()
        for tag in tags_source
        if str(tag).strip()
    ]
    metadata["base_models"] = _extract_base_models(
        model_index,
        modular_index,
        config,
    )
    metadata["files"] = [
        str(path.relative_to(root)).lower()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]

    if not metadata["library_name"] and (
        modular_index_path.is_file()
        or (
            model_index_path.is_file()
            and str(metadata.get("class_name") or "").endswith(
                "Pipeline"
            )
        )
    ):
        metadata["library_name"] = "diffusers"

    pipeline_cls = None
    try:
        resolution = PipelineResolver().resolve_class(metadata)
        pipeline_cls = resolution.pipeline_cls
        metadata["runtime_supported"] = resolution.runtime_supported
        metadata["runtime_reason"] = resolution.runtime_reason
        metadata["compatibility_status"] = (
            "SUPPORTED"
            if resolution.runtime_supported
            else "UNSUPPORTED"
            if resolution.class_name
            else "UNKNOWN"
        )
    except Exception as error:
        metadata["runtime_supported"] = False
        metadata["runtime_reason"] = (
            f"{type(error).__name__}: {error}"
        )
        metadata["compatibility_status"] = "UNSUPPORTED"

    capability_sets = CapabilityResolver().describe(
        metadata,
        pipeline_cls,
    )
    metadata.update(capability_sets)
    metadata["capabilities"] = capability_sets[
        "display_capabilities"
    ]
    return metadata
