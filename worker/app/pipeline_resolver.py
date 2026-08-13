"""Résolution générique des pipelines disponibles dans le runtime Diffusers."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any

from .modular_runtime import ModularManifestResolver


@dataclass(slots=True)
class PipelineResolution:
    class_name: str | None
    pipeline_cls: Any | None
    runtime_supported: bool
    strategy: str | None = None
    runtime_reason: str = ""
    attempted: list[dict[str, str]] = field(default_factory=list)


class PipelineResolutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        dependency: str | None = None,
        attempts: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.dependency = dependency
        self.attempts = attempts or []


class PipelineResolver:
    """Diffusers est la source de vérité, jamais l'identifiant du repository."""

    AUTO_PIPELINES = {
        "TEXT_TO_IMAGE": "AutoPipelineForText2Image",
        "IMAGE_TO_IMAGE": "AutoPipelineForImage2Image",
        "INPAINTING": "AutoPipelineForInpainting",
        "TEXT_TO_VIDEO": "AutoPipelineForText2Video",
        "IMAGE_TO_VIDEO": "AutoPipelineForImage2Video",
    }

    @staticmethod
    def _clean_class_name(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and value:
            for item in reversed(value):
                if (
                    isinstance(item, str)
                    and item.endswith("Pipeline")
                ):
                    return item
        return None

    @staticmethod
    def is_modular(metadata: dict[str, Any]) -> bool:
        if metadata.get("is_modular") is True:
            return True
        modular_index = metadata.get("modular_model_index")
        if isinstance(modular_index, dict) and bool(modular_index):
            return True
        if isinstance(metadata.get("modular_manifest"), dict):
            return True
        if str(metadata.get("manifest_type") or "").strip().lower() == "modular":
            return True
        if str(metadata.get("model_index_kind") or "").strip().lower() == "modular":
            return True
        return any(
            str(value).replace("\\", "/").lower().endswith(
                "modular_model_index.json"
            )
            for value in metadata.get("files") or []
        )

    def class_candidates(
        self,
        metadata: dict[str, Any],
    ) -> list[str]:
        modular_index = metadata.get(
            "modular_model_index"
        ) or {}
        model_index = metadata.get("model_index") or {}
        config = metadata.get("config") or {}
        candidates: list[str] = []

        # Un manifest modular est plus spécifique qu'un model_index standard.
        for value in (
            modular_index.get("_class_name"),
            model_index.get("_class_name"),
            config.get("_class_name"),
            metadata.get("class_name"),
        ):
            class_name = self._clean_class_name(value)
            if class_name:
                candidates.append(class_name)

        architectures = list(
            metadata.get("architectures") or []
        )
        architectures.extend(
            config.get("architectures") or []
        )
        for value in architectures:
            class_name = self._clean_class_name(value)
            if (
                class_name
                and class_name.endswith("Pipeline")
            ):
                candidates.append(class_name)

        for source in (modular_index, model_index):
            for key, component in source.items():
                if str(key).startswith("_"):
                    continue
                class_name = self._clean_class_name(component)
                if (
                    class_name
                    and class_name.endswith("Pipeline")
                ):
                    candidates.append(class_name)

        seen: set[str] = set()
        return [
            value
            for value in candidates
            if not (value in seen or seen.add(value))
        ]

    @staticmethod
    def requires_remote_code(
        metadata: dict[str, Any],
    ) -> bool:
        for source in (
            metadata,
            metadata.get("modular_model_index") or {},
            metadata.get("model_index") or {},
            metadata.get("config") or {},
        ):
            if source.get("trust_remote_code") is True:
                return True
            if source.get("auto_map"):
                return True
            if (
                source.get("custom_pipeline")
                or source.get("custom_revision")
            ):
                return True
        return False

    @staticmethod
    def _diffusers(module: Any | None = None) -> Any:
        if module is not None:
            return module
        try:
            return importlib.import_module("diffusers")
        except (ImportError, ModuleNotFoundError) as error:
            dependency = (
                PipelineResolver._missing_dependency(error)
                or getattr(error, "name", None)
                or "diffusers"
            )
            raise PipelineResolutionError(
                f"La dépendance {dependency} n'est pas installée.",
                code="MISSING_DEPENDENCY",
                dependency=dependency,
            ) from error

    def resolve_class(
        self,
        metadata: dict[str, Any],
        *,
        diffusers_module: Any | None = None,
    ) -> PipelineResolution:
        library = str(
            metadata.get("library_name") or "diffusers"
        ).lower()
        if library not in {"", "diffusers"}:
            return PipelineResolution(
                class_name=None,
                pipeline_cls=None,
                runtime_supported=False,
                runtime_reason=(
                    f"Bibliothèque non prise en charge: {library}"
                ),
            )
        if self.requires_remote_code(metadata):
            return PipelineResolution(
                class_name=metadata.get("class_name"),
                pipeline_cls=None,
                runtime_supported=False,
                runtime_reason="REMOTE_CODE_REQUIRED",
            )

        diffusers = self._diffusers(diffusers_module)

        if self.is_modular(metadata):
            modular_cls = getattr(
                diffusers,
                "ModularPipeline",
                None,
            )
            class_name = (
                self._clean_class_name(
                    (metadata.get("modular_model_index") or {}).get(
                        "_class_name"
                    )
                )
                or "ModularPipeline"
            )
            if modular_cls is None:
                return PipelineResolution(
                    class_name=class_name,
                    pipeline_cls=None,
                    runtime_supported=False,
                    strategy="modular-pipeline",
                    runtime_reason=(
                        "DIFFUSERS_VERSION_TOO_OLD: "
                        "ModularPipeline absent de Diffusers."
                    ),
                )

            # H3 exige sa classe exacte. Les autres manifests modular restent
            # chargeables par le conteneur ModularPipeline générique.
            exact_modular_cls = None
            if class_name != "ModularPipeline":
                exact_modular_cls = getattr(
                    diffusers,
                    class_name,
                    None,
                )
                if (
                    exact_modular_cls is None
                    and class_name == "MiniMaxH3ModularPipeline"
                ):
                    return PipelineResolution(
                        class_name=class_name,
                        pipeline_cls=None,
                        runtime_supported=False,
                        strategy="modular-pipeline",
                        runtime_reason=(
                            "DIFFUSERS_VERSION_TOO_OLD: architecture modular "
                            f"{class_name} absente de Diffusers."
                        ),
                    )

            return PipelineResolution(
                class_name=class_name,
                pipeline_cls=exact_modular_cls or modular_cls,
                runtime_supported=True,
                strategy="modular-pipeline",
                runtime_reason=(
                    "Manifest modular_model_index.json reconnu et "
                    f"architecture {class_name} disponible dans Diffusers."
                ),
            )

        candidates = self.class_candidates(metadata)
        explicit_class = self._clean_class_name(
            metadata.get("pipeline_class") or metadata.get("class_name")
        )
        if explicit_class and explicit_class not in candidates:
            candidates.insert(0, explicit_class)
        for class_name in candidates:
            try:
                pipeline_cls = getattr(
                    diffusers,
                    class_name,
                    None,
                )
            except (ImportError, ModuleNotFoundError) as error:
                dependency = (
                    self._missing_dependency(error)
                    or getattr(error, "name", None)
                )
                raise PipelineResolutionError(
                    "Dépendance optionnelle manquante: "
                    f"{dependency or 'inconnue'}",
                    code="MISSING_DEPENDENCY",
                    dependency=dependency,
                ) from error
            if pipeline_cls is not None:
                return PipelineResolution(
                    class_name=class_name,
                    pipeline_cls=pipeline_cls,
                    runtime_supported=True,
                    strategy="exact-class",
                    runtime_reason=(
                        f"{class_name} est disponible dans Diffusers."
                    ),
                )

        class_name = candidates[0] if candidates else None
        reason = (
            "DIFFUSERS_VERSION_TOO_OLD: classe "
            f"{class_name} absente de Diffusers."
            if class_name
            else (
                "PIPELINE_CLASS_NOT_AVAILABLE: "
                "aucune classe pipeline déclarée."
            )
        )
        return PipelineResolution(
            class_name=class_name,
            pipeline_cls=None,
            runtime_supported=False,
            runtime_reason=reason,
        )

    @staticmethod
    def _missing_dependency(
        error: BaseException,
    ) -> str | None:
        if isinstance(error, ModuleNotFoundError):
            return error.name or "unknown"
        message = str(error)
        match = re.search(
            r"No module named ['\"]([^'\"]+)",
            message,
        )
        return match.group(1) if match else None

    @staticmethod
    def _remote_code_error(
        error: BaseException,
    ) -> bool:
        message = str(error).lower()
        return (
            "trust_remote_code" in message
            or "execute the configuration file" in message
        )

    def load(
        self,
        snapshot: str,
        metadata: dict[str, Any],
        capability: str,
        settings: dict[str, Any],
        *,
        diffusers_module: Any | None = None,
    ) -> tuple[Any, PipelineResolution]:
        if self.requires_remote_code(metadata):
            raise PipelineResolutionError(
                "Le snapshot exige l'exécution de code distant.",
                code="REMOTE_CODE_REQUIRED",
            )

        if self.is_modular(metadata):
            raise PipelineResolutionError(
                "Un repository modular doit être chargé par "
                "ModularDiffusersAdapter.",
                code="MODULAR_ADAPTER_REQUIRED",
            )

        diffusers = self._diffusers(diffusers_module)
        exact = self.resolve_class(
            metadata,
            diffusers_module=diffusers,
        )
        loaders: list[tuple[str, Any]] = []
        if exact.pipeline_cls is not None:
            loaders.append(
                ("exact-class", exact.pipeline_cls)
            )

        auto_name = self.AUTO_PIPELINES.get(capability)
        auto_cls = (
            getattr(diffusers, auto_name, None)
            if auto_name
            else None
        )
        if auto_cls is not None:
            loaders.append((auto_name, auto_cls))

        generic_cls = getattr(
            diffusers,
            "DiffusionPipeline",
            None,
        )
        if generic_cls is not None:
            loaders.append(
                ("DiffusionPipeline", generic_cls)
            )

        attempts: list[dict[str, str]] = []
        for strategy, loader_cls in loaders:
            try:
                pipeline = loader_cls.from_pretrained(
                    snapshot,
                    local_files_only=True,
                    use_safetensors=None,
                    torch_dtype=settings.get("torch_dtype"),
                    trust_remote_code=False,
                )
                from .capability_resolver import CapabilityResolver

                resolved_capabilities = (
                    CapabilityResolver().resolve(
                        metadata,
                        pipeline,
                    )
                )
                if (
                    resolved_capabilities
                    and capability
                    not in resolved_capabilities
                ):
                    attempts.append(
                        {
                            "strategy": strategy,
                            "result": "FAILED",
                            "error": (
                                f"capability {capability} "
                                "absente de la signature"
                            ),
                        }
                    )
                    print(
                        "PIPELINE_RESOLVE "
                        f"strategy={strategy} result=FAILED"
                    )
                    del pipeline
                    continue
                attempts.append(
                    {
                        "strategy": strategy,
                        "result": "SUCCESS",
                    }
                )
                print(
                    "PIPELINE_RESOLVE "
                    f"strategy={strategy} result=SUCCESS"
                )
                return pipeline, PipelineResolution(
                    class_name=(
                        exact.class_name
                        or getattr(
                            loader_cls,
                            "__name__",
                            None,
                        )
                    ),
                    pipeline_cls=loader_cls,
                    runtime_supported=True,
                    strategy=strategy,
                    runtime_reason=(
                        f"Chargement réussi via {strategy}."
                    ),
                    attempted=attempts,
                )
            except Exception as error:
                dependency = self._missing_dependency(error)
                attempts.append(
                    {
                        "strategy": strategy,
                        "result": "FAILED",
                        "error": (
                            f"{type(error).__name__}: {error}"
                        ),
                    }
                )
                print(
                    "PIPELINE_RESOLVE "
                    f"strategy={strategy} result=FAILED"
                )
                if dependency:
                    raise PipelineResolutionError(
                        "Dépendance optionnelle manquante: "
                        f"{dependency}",
                        code="MISSING_DEPENDENCY",
                        dependency=dependency,
                        attempts=attempts,
                    ) from error
                if self._remote_code_error(error):
                    raise PipelineResolutionError(
                        "Le pipeline exige trust_remote_code.",
                        code="REMOTE_CODE_REQUIRED",
                        attempts=attempts,
                    ) from error

        if (
            exact.class_name
            and exact.pipeline_cls is None
        ):
            code = "DIFFUSERS_VERSION_TOO_OLD"
            message = exact.runtime_reason
        elif not loaders:
            code = "PIPELINE_CLASS_NOT_AVAILABLE"
            message = (
                "Aucun loader Diffusers exploitable "
                "n'est installé."
            )
        else:
            code = "INVALID_MODEL_SNAPSHOT"
            message = (
                "Tous les loaders Diffusers ont refusé "
                "le snapshot."
            )
        raise PipelineResolutionError(
            message,
            code=code,
            attempts=attempts,
        )
