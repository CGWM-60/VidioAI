"""Support générique des manifests Diffusers ModularPipeline.

Aucun identifiant de modèle n'est codé en dur. Le module lit
``modular_model_index.json``, décrit ses composants et aide le runtime à
matérialiser localement les dépendances externes afin que l'inférence ne dépende
pas du réseau.

Ce module ne permet jamais ``trust_remote_code=True``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ModularRuntimeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "MODULAR_RUNTIME_INVALID",
        status_code: int = 422,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class ModularComponentSpec:
    name: str
    repository: str | None
    revision: str | None
    subfolder: str | None
    variant: str | None
    type_library: str | None
    type_class: str | None

    @property
    def external(self) -> bool:
        return bool(
            self.repository
            and REPOSITORY_RE.fullmatch(self.repository)
        )

    def materialization_key(self) -> str:
        source = "|".join(
            [
                self.repository or "local",
                self.revision or "main",
                self.subfolder or "",
            ]
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModularManifestResolver:
    """Parse un manifest Modular Diffusers sans charger de poids."""

    @staticmethod
    def read(snapshot: str | Path) -> dict[str, Any]:
        path = Path(snapshot) / "modular_model_index.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModularRuntimeError(
                "modular_model_index.json est invalide.",
                code="MODULAR_MANIFEST_INVALID",
            ) from error
        if not isinstance(payload, dict):
            raise ModularRuntimeError(
                "modular_model_index.json doit être un objet JSON.",
                code="MODULAR_MANIFEST_INVALID",
            )
        return payload

    @staticmethod
    def from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        value = metadata.get("modular_model_index")
        return value if isinstance(value, dict) else {}

    @staticmethod
    def is_modular(metadata: dict[str, Any]) -> bool:
        return bool(
            metadata.get("is_modular")
            or isinstance(metadata.get("modular_model_index"), dict)
        )

    @staticmethod
    def requires_remote_code(
        modular_index: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> bool:
        config = config or {}
        for source in (modular_index, config):
            if source.get("trust_remote_code") is True:
                return True
            if source.get("auto_map"):
                return True
            if source.get("custom_pipeline") or source.get("custom_revision"):
                return True
        return False

    @staticmethod
    def components(
        modular_index: dict[str, Any],
    ) -> list[ModularComponentSpec]:
        result: list[ModularComponentSpec] = []

        for name, raw in modular_index.items():
            if str(name).startswith("_"):
                continue

            loading: dict[str, Any] = {}
            type_hint: Any = None

            if isinstance(raw, list):
                if len(raw) >= 3 and isinstance(raw[2], dict):
                    loading = raw[2]
                if len(raw) >= 2 and raw[0] and raw[1]:
                    type_hint = [raw[0], raw[1]]
                if loading.get("type_hint") is not None:
                    type_hint = loading.get("type_hint")
            elif isinstance(raw, dict):
                loading = (
                    raw.get("loading_specs_dict")
                    if isinstance(raw.get("loading_specs_dict"), dict)
                    else raw
                )
                type_hint = raw.get("type_hint") or loading.get("type_hint")
            else:
                continue

            type_library = None
            type_class = None
            if (
                isinstance(type_hint, (list, tuple))
                and len(type_hint) >= 2
            ):
                if isinstance(type_hint[0], str):
                    type_library = type_hint[0]
                if isinstance(type_hint[1], str):
                    type_class = type_hint[1]

            repository = loading.get("pretrained_model_name_or_path")
            if repository is not None:
                repository = str(repository).strip() or None

            revision = loading.get("revision")
            if revision is not None:
                revision = str(revision).strip() or None

            subfolder = loading.get("subfolder")
            if subfolder is not None:
                subfolder = str(subfolder).strip().strip("/") or None

            variant = loading.get("variant")
            if variant is not None:
                variant = str(variant).strip() or None

            result.append(
                ModularComponentSpec(
                    name=str(name),
                    repository=repository,
                    revision=revision,
                    subfolder=subfolder,
                    variant=variant,
                    type_library=type_library,
                    type_class=type_class,
                )
            )

        return result

    @classmethod
    def external_components(
        cls,
        modular_index: dict[str, Any],
        *,
        base_repository: str | None = None,
    ) -> list[ModularComponentSpec]:
        result: list[ModularComponentSpec] = []
        for item in cls.components(modular_index):
            if not item.external:
                continue
            if (
                base_repository
                and item.repository
                and item.repository.lower() == base_repository.lower()
            ):
                # Le composant est dans le repository principal. Le snapshot de
                # base doit déjà le contenir.
                continue
            result.append(item)
        return result

    @classmethod
    def type_hints(
        cls,
        modular_index: dict[str, Any],
    ) -> list[str]:
        values: list[str] = []
        for item in cls.components(modular_index):
            if item.type_library:
                values.append(item.type_library)
            if item.type_class:
                values.append(item.type_class)
        return list(dict.fromkeys(values))

    @staticmethod
    def write_materialization(
        snapshot: Path,
        records: list[dict[str, Any]],
    ) -> Path:
        directory = snapshot / "vidioai"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "modular-components.json"
        temporary = path.with_suffix(".json.tmp")
        payload = {
            "schema_version": 1,
            "components": records,
        }
        temporary.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    @staticmethod
    def read_materialization(
        snapshot: str | Path,
    ) -> list[dict[str, Any]]:
        path = Path(snapshot) / "vidioai" / "modular-components.json"
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        values = payload.get("components") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return []
        return [dict(value) for value in values if isinstance(value, dict)]

    @classmethod
    def component_config_paths(
        cls,
        snapshot: str | Path,
    ) -> dict[str, Path]:
        root = Path(snapshot)
        result: dict[str, Path] = {}
        for record in cls.read_materialization(root):
            name = str(record.get("name") or "")
            local_root = record.get("local_root")
            if not name or not isinstance(local_root, str):
                continue
            candidate = root / local_root
            subfolder = record.get("subfolder")
            if isinstance(subfolder, str) and subfolder:
                candidate = candidate / subfolder
            config = candidate / "config.json"
            if config.is_file():
                result[name] = config
        return result
