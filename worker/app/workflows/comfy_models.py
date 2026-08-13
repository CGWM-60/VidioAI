"""Persistent and safe bridge from Diffusers snapshots to ComfyUI model paths."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..packs.schema import ModelPack


def _synchronized(method):
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class ComfyModelError(ValueError):
    def __init__(self, message: str, *, code: str = "MODEL_FILE_MISSING") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class MaterializedComfyModels:
    components: dict[str, Any]
    links: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {"components": self.components, "links": list(self.links)}


class ComfyModelMaterializer:
    """Create namespaced relative links in the /models volume shared by ComfyUI."""

    EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
    CATEGORIES = {"diffusion_models", "text_encoders", "vae"}
    CATEGORY_BY_COMPONENT = {
        "checkpoint": "diffusion_models",
        "vae": "vae",
        "text_encoder": "text_encoders",
    }

    def __init__(self, models_root: Path | str) -> None:
        self.models_root = Path(models_root)
        self.registry_dir = self.models_root / ".vidioai-comfy"
        self.registry_path = self.registry_dir / "registry.json"
        self._lock = threading.RLock()

    @staticmethod
    def _safe(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
        if not normalized:
            raise ComfyModelError("Identifiant de lien ComfyUI invalide.")
        return normalized

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        return path == root or root in path.parents

    def _component_file(self, snapshot: Path, declared: str) -> Path:
        root = snapshot.resolve(strict=True)
        candidate = (snapshot / declared).resolve(strict=False)
        if not self._inside(candidate, root) or not candidate.exists():
            raise ComfyModelError(f"Composant ModelPack absent ou hors snapshot: {declared}")
        if candidate.is_file():
            if candidate.suffix.lower() not in self.EXTENSIONS:
                raise ComfyModelError(f"Format ComfyUI non supporté: {declared}")
            return candidate
        files = sorted(
            path.resolve()
            for path in candidate.rglob("*")
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS
        )
        safe_tensors = [path for path in files if path.suffix.lower() == ".safetensors"]
        preferred = safe_tensors or files
        if len(preferred) != 1 or any("-of-" in path.name for path in preferred):
            detail = ", ".join(path.name for path in preferred[:4]) or "aucun fichier"
            raise ComfyModelError(
                f"Composant ComfyUI ambigu ou shardé ({declared}): {detail}"
            )
        if not self._inside(preferred[0], root):
            raise ComfyModelError(f"Composant résolu hors snapshot: {declared}")
        return preferred[0]

    def _planned_components(
        self,
        snapshot: Path,
        pack: ModelPack,
    ) -> list[tuple[str, str, Path]]:
        planned: list[tuple[str, str, Path]] = []
        for component_name in ("checkpoint", "vae"):
            declared = pack.components.get(component_name)
            if declared:
                planned.append(
                    (
                        component_name,
                        self.CATEGORY_BY_COMPONENT[component_name],
                        self._component_file(snapshot, str(declared)),
                    )
                )
        for index, declared in enumerate(pack.components.get("text_encoders") or [], 1):
            planned.append(
                (
                    f"text_encoder_{index}",
                    self.CATEGORY_BY_COMPONENT["text_encoder"],
                    self._component_file(snapshot, str(declared)),
                )
            )
        return planned

    def _destination(
        self,
        *,
        category: str,
        model_id: str,
        pack_id: str,
        component_name: str,
        suffix: str,
    ) -> Path:
        if category not in self.CATEGORIES:
            raise ComfyModelError(f"Catégorie ComfyUI interdite: {category}")
        name = (
            f"vidioai-{self._safe(model_id)}-{self._safe(pack_id)}-"
            f"{self._safe(component_name)}{suffix.lower()}"
        )
        return self.models_root / category / name

    @_synchronized
    def materialize(
        self,
        *,
        snapshot: Path,
        model_id: str,
        pack: ModelPack,
    ) -> MaterializedComfyModels:
        if pack.engine != "comfyui":
            return MaterializedComfyModels(dict(pack.components), ())
        # Resolve every source before mutating the persistent shared tree.
        planned = self._planned_components(snapshot, pack)
        mutations: list[tuple[Path, str | None]] = []
        links: list[dict[str, str]] = []
        component_names: dict[str, str] = {}
        try:
            for component_name, category, source in planned:
                destination = self._destination(
                    category=category,
                    model_id=model_id,
                    pack_id=pack.id,
                    component_name=component_name,
                    suffix=source.suffix,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                relative_target = os.path.relpath(source, start=destination.parent)
                previous: str | None = None
                if os.path.lexists(destination):
                    if not destination.is_symlink():
                        raise ComfyModelError(
                            f"Le chemin ComfyUI géré est occupé: {destination.name}"
                        )
                    previous = os.readlink(destination)
                    if previous == relative_target:
                        component_names[component_name] = destination.name
                        links.append(
                            self._record(category, destination, source, relative_target)
                        )
                        continue
                temporary = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    temporary.symlink_to(relative_target)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                mutations.append((destination, previous))
                component_names[component_name] = destination.name
                links.append(self._record(category, destination, source, relative_target))
            components = {
                "checkpoint": component_names.get("checkpoint"),
                "vae": component_names.get("vae"),
                "text_encoders": [
                    component_names[key]
                    for key in sorted(component_names)
                    if key.startswith("text_encoder_")
                ],
                "loras": [],
            }
            result = MaterializedComfyModels(components, tuple(links))
            self._persist(model_id, pack.id, result)
            return result
        except Exception:
            for destination, previous in reversed(mutations):
                destination.unlink(missing_ok=True)
                if previous is not None:
                    destination.symlink_to(previous)
            raise

    @staticmethod
    def _record(
        category: str,
        destination: Path,
        source: Path,
        target: str,
    ) -> dict[str, str]:
        return {
            "category": category,
            "name": destination.name,
            "source": str(source),
            "target": target,
        }

    def _read_registry(self) -> dict[str, Any]:
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
            return {"schema_version": 1, "models": {}}
        return value

    def _write_registry(self, value: dict[str, Any]) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_name(
            f".{self.registry_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(value, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.registry_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _persist(
        self,
        model_id: str,
        pack_id: str,
        result: MaterializedComfyModels,
    ) -> None:
        registry = self._read_registry()
        registry["models"][model_id] = {
            "model_pack_id": pack_id,
            **result.as_dict(),
        }
        self._write_registry(registry)

    @_synchronized
    def remove(self, model_id: str) -> dict[str, Any]:
        """Remove only namespaced symlinks recorded for one uninstalled model."""
        registry = self._read_registry()
        record = registry["models"].pop(model_id, None)
        removed: list[str] = []
        if isinstance(record, dict):
            for link in record.get("links") or []:
                if not isinstance(link, dict):
                    continue
                category = str(link.get("category") or "")
                name = str(link.get("name") or "")
                if (
                    category not in self.CATEGORIES
                    or not name.startswith("vidioai-")
                    or Path(name).name != name
                ):
                    continue
                destination = self.models_root / category / name
                if destination.is_symlink():
                    destination.unlink()
                    removed.append(f"{category}/{name}")
        self._write_registry(registry)
        return {"model_id": model_id, "removed": removed}

    @_synchronized
    def prune(self) -> dict[str, Any]:
        """Drop links whose recorded snapshot source was uninstalled."""
        registry = self._read_registry()
        stale = [
            model_id
            for model_id, record in registry["models"].items()
            if isinstance(record, dict)
            and any(
                isinstance(link, dict)
                and not Path(str(link.get("source") or "")).is_file()
                for link in record.get("links") or []
            )
        ]
        removed: list[str] = []
        for model_id in stale:
            removed.extend(self.remove(model_id)["removed"])
        return {"models": stale, "removed": removed}
