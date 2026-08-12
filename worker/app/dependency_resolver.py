"""Résolution générique et strictement allowlistée des imports optionnels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .pipeline_resolver import PipelineResolutionError


MAX_DEPENDENCY_RESOLUTION_PASSES = 8
T = TypeVar("T")


class DependencyResolutionError(RuntimeError):
    def __init__(self, message: str, *, code: str, dependency: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.dependency = dependency


@dataclass(frozen=True, slots=True)
class DependencySpec:
    import_name: str
    package: str
    version: str


class DependencyRegistry:
    """Mapping interne : aucune chaîne provenant d'un repository n'atteint pip."""

    _SPECS = {
        "bitsandbytes": DependencySpec("bitsandbytes", "bitsandbytes", "0.49.2"),
        "torchao": DependencySpec("torchao", "torchao", "0.17.0"),
        "sentencepiece": DependencySpec("sentencepiece", "sentencepiece", "0.2.2"),
        "einops": DependencySpec("einops", "einops", "0.8.2"),
        "peft": DependencySpec("peft", "peft", "0.20.0"),
        "ftfy": DependencySpec("ftfy", "ftfy", "6.3.1"),
        "timm": DependencySpec("timm", "timm", "1.0.28"),
        "av": DependencySpec("av", "av", "18.0.0"),
        "cv2": DependencySpec("cv2", "opencv-python-headless", "5.0.0.93"),
        "yaml": DependencySpec("yaml", "PyYAML", "6.0.3"),
        "skimage": DependencySpec("skimage", "scikit-image", "0.26.0"),
    }

    @classmethod
    def resolve(cls, missing_import: str) -> DependencySpec:
        root_import = str(missing_import or "").strip().split(".", 1)[0]
        spec = cls._SPECS.get(root_import)
        if spec is None:
            raise DependencyResolutionError(
                f"La dépendance Python {root_import or 'inconnue'} n'est pas autorisée.",
                code="DEPENDENCY_NOT_ALLOWED",
                dependency=root_import or None,
            )
        return spec


def _walk(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower(), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


class DependencyResolver:
    """Détecte les dépendances depuis les configs puis répare les imports lazy."""

    BNB_KEYS = {
        "quantization_config",
        "load_in_4bit",
        "load_in_8bit",
        "bnb_4bit_compute_dtype",
        "bnb_4bit_quant_storage",
        "bnb_4bit_quant_type",
        "bnb_4bit_use_double_quant",
        "quant_method",
    }

    @classmethod
    def requirements_from_payloads(cls, *payloads: Any) -> list[DependencySpec]:
        requires_bnb = False
        for payload in payloads:
            for key, value in _walk(payload):
                normalized = str(value).lower()
                if key.startswith("bnb_4bit"):
                    requires_bnb = True
                elif key in {"load_in_4bit", "load_in_8bit"} and value is True:
                    requires_bnb = True
                elif key in cls.BNB_KEYS and any(
                    marker in normalized for marker in ("bitsandbytes", "nf4", "fp4")
                ):
                    requires_bnb = True
        return [DependencyRegistry.resolve("bitsandbytes")] if requires_bnb else []

    @classmethod
    def requirements_from_snapshot(
        cls, snapshot: Path, metadata: dict[str, Any] | None = None
    ) -> list[DependencySpec]:
        # Seuls les JSON de configuration sont interprétés comme données. Les
        # requirements.txt, modules Python et custom pipelines sont ignorés.
        payloads: list[Any] = [metadata or {}]
        for path in sorted(snapshot.rglob("*.json")):
            if path.name not in {
                "config.json",
                "model_index.json",
                "modular_model_index.json",
                "quantization_config.json",
            }:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payloads.append(payload)
        return cls.requirements_from_payloads(*payloads)

    @staticmethod
    def missing_dependency(error: BaseException) -> str | None:
        if isinstance(error, (PipelineResolutionError, DependencyResolutionError)):
            return error.dependency
        if isinstance(error, ModuleNotFoundError):
            return error.name
        return None

    def load_with_repair(
        self,
        loader: Callable[[], T],
        installer: Any,
        *,
        required_by: str = "pipeline",
    ) -> tuple[T, list[dict[str, Any]]]:
        attempted: set[str] = set()
        records: list[dict[str, Any]] = []
        for _pass in range(MAX_DEPENDENCY_RESOLUTION_PASSES + 1):
            try:
                return loader(), records
            except Exception as error:
                missing = self.missing_dependency(error)
                if not missing:
                    raise
                spec = DependencyRegistry.resolve(missing)
                if spec.import_name in attempted:
                    raise DependencyResolutionError(
                        f"La dépendance {spec.import_name} reste absente après installation.",
                        code="DEPENDENCY_INSTALL_FAILED",
                        dependency=spec.import_name,
                    ) from error
                if len(attempted) >= MAX_DEPENDENCY_RESOLUTION_PASSES:
                    raise DependencyResolutionError(
                        "Nombre maximal de résolutions de dépendances atteint.",
                        code="DEPENDENCY_INSTALL_FAILED",
                        dependency=spec.import_name,
                    ) from error
                attempted.add(spec.import_name)
                records.append(installer.ensure(spec.import_name, required_by=required_by))
        raise DependencyResolutionError(
            "Nombre maximal de résolutions de dépendances atteint.",
            code="DEPENDENCY_INSTALL_FAILED",
        )
