"""Resolution generique des pipelines disponibles dans le runtime Diffusers."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any


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
    """Diffusers est la source de verite, jamais l'identifiant du repository."""

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
                if isinstance(item, str) and item.endswith("Pipeline"):
                    return item
        return None

    def class_candidates(self, metadata: dict[str, Any]) -> list[str]:
        model_index = metadata.get("model_index") or {}
        config = metadata.get("config") or {}
        candidates: list[str] = []

        # Ordre contractuel: model_index, config, architectures, puis indices
        # moins precis. Les tags ne deviennent jamais des noms de classes.
        for value in (
            model_index.get("_class_name"),
            config.get("_class_name"),
            metadata.get("class_name"),
        ):
            class_name = self._clean_class_name(value)
            if class_name:
                candidates.append(class_name)

        architectures = list(metadata.get("architectures") or [])
        architectures.extend(config.get("architectures") or [])
        for value in architectures:
            class_name = self._clean_class_name(value)
            if class_name and class_name.endswith("Pipeline"):
                candidates.append(class_name)

        # Les composants du model_index peuvent exceptionnellement declarer une
        # pipeline reutilisable. Les modeles/VAEs/schedulers sont ignores.
        for key, component in model_index.items():
            if str(key).startswith("_"):
                continue
            class_name = self._clean_class_name(component)
            if class_name and class_name.endswith("Pipeline"):
                candidates.append(class_name)

        seen: set[str] = set()
        return [
            value
            for value in candidates
            if not (value in seen or seen.add(value))
        ]

    @staticmethod
    def requires_remote_code(metadata: dict[str, Any]) -> bool:
        for source in (metadata, metadata.get("model_index") or {}, metadata.get("config") or {}):
            if source.get("trust_remote_code") is True or source.get("auto_map"):
                return True
            if source.get("custom_pipeline") or source.get("custom_revision"):
                return True
        return False

    @staticmethod
    def _diffusers(module: Any | None = None) -> Any:
        if module is not None:
            return module
        try:
            return importlib.import_module("diffusers")
        except (ImportError, ModuleNotFoundError) as error:
            dependency = PipelineResolver._missing_dependency(error) or getattr(error, "name", None) or "diffusers"
            raise PipelineResolutionError(
                f"La dependance {dependency} n'est pas installee.",
                code="MISSING_DEPENDENCY",
                dependency=dependency,
            ) from error

    def resolve_class(
        self,
        metadata: dict[str, Any],
        *,
        diffusers_module: Any | None = None,
    ) -> PipelineResolution:
        library = str(metadata.get("library_name") or "diffusers").lower()
        if library not in {"", "diffusers"}:
            return PipelineResolution(
                class_name=None,
                pipeline_cls=None,
                runtime_supported=False,
                runtime_reason=f"Bibliotheque non prise en charge: {library}",
            )
        if self.requires_remote_code(metadata):
            return PipelineResolution(
                class_name=None,
                pipeline_cls=None,
                runtime_supported=False,
                runtime_reason="REMOTE_CODE_REQUIRED",
            )

        diffusers = self._diffusers(diffusers_module)
        candidates = self.class_candidates(metadata)
        for class_name in candidates:
            try:
                pipeline_cls = getattr(diffusers, class_name, None)
            except (ImportError, ModuleNotFoundError) as error:
                dependency = self._missing_dependency(error) or getattr(error, "name", None)
                raise PipelineResolutionError(
                    f"Dependance optionnelle manquante: {dependency or 'inconnue'}",
                    code="MISSING_DEPENDENCY",
                    dependency=dependency,
                ) from error
            if pipeline_cls is not None:
                return PipelineResolution(
                    class_name=class_name,
                    pipeline_cls=pipeline_cls,
                    runtime_supported=True,
                    strategy="exact-class",
                    runtime_reason=f"{class_name} est disponible dans Diffusers.",
                )

        class_name = candidates[0] if candidates else None
        reason = (
            f"DIFFUSERS_VERSION_TOO_OLD: classe {class_name} absente de Diffusers."
            if class_name
            else "PIPELINE_CLASS_NOT_AVAILABLE: aucune classe pipeline declaree."
        )
        return PipelineResolution(
            class_name=class_name,
            pipeline_cls=None,
            runtime_supported=False,
            runtime_reason=reason,
        )

    @staticmethod
    def _missing_dependency(error: BaseException) -> str | None:
        if isinstance(error, ModuleNotFoundError):
            return error.name or "unknown"
        message = str(error)
        match = re.search(r"No module named ['\"]([^'\"]+)", message)
        return match.group(1) if match else None

    @staticmethod
    def _remote_code_error(error: BaseException) -> bool:
        message = str(error).lower()
        return "trust_remote_code" in message or "execute the configuration file" in message

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
                "Le snapshot exige l'execution de code distant.",
                code="REMOTE_CODE_REQUIRED",
            )

        diffusers = self._diffusers(diffusers_module)
        exact = self.resolve_class(metadata, diffusers_module=diffusers)
        loaders: list[tuple[str, Any]] = []
        if exact.pipeline_cls is not None:
            loaders.append(("exact-class", exact.pipeline_cls))

        auto_name = self.AUTO_PIPELINES.get(capability)
        auto_cls = getattr(diffusers, auto_name, None) if auto_name else None
        if auto_cls is not None:
            loaders.append((auto_name, auto_cls))

        generic_cls = getattr(diffusers, "DiffusionPipeline", None)
        if generic_cls is not None:
            loaders.append(("DiffusionPipeline", generic_cls))

        attempts: list[dict[str, str]] = []
        for strategy, loader_cls in loaders:
            try:
                pipeline = loader_cls.from_pretrained(
                    snapshot,
                    local_files_only=True,
                    # None laisse Diffusers preferer Safetensors tout en gardant
                    # les snapshots standards qui ne publient que des .bin.
                    use_safetensors=None,
                    torch_dtype=settings.get("torch_dtype"),
                    trust_remote_code=False,
                )
                from .capability_resolver import CapabilityResolver

                resolved_capabilities = CapabilityResolver().resolve(metadata, pipeline)
                if resolved_capabilities and capability not in resolved_capabilities:
                    attempts.append(
                        {
                            "strategy": strategy,
                            "result": "FAILED",
                            "error": f"capability {capability} absente de la signature",
                        }
                    )
                    print(f"PIPELINE_RESOLVE strategy={strategy} result=FAILED")
                    del pipeline
                    continue
                attempts.append({"strategy": strategy, "result": "SUCCESS"})
                print(f"PIPELINE_RESOLVE strategy={strategy} result=SUCCESS")
                return pipeline, PipelineResolution(
                    class_name=exact.class_name or getattr(loader_cls, "__name__", None),
                    pipeline_cls=loader_cls,
                    runtime_supported=True,
                    strategy=strategy,
                    runtime_reason=f"Chargement reussi via {strategy}.",
                    attempted=attempts,
                )
            except Exception as error:
                dependency = self._missing_dependency(error)
                attempts.append(
                    {
                        "strategy": strategy,
                        "result": "FAILED",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"PIPELINE_RESOLVE strategy={strategy} result=FAILED")
                if dependency:
                    raise PipelineResolutionError(
                        f"Dependance optionnelle manquante: {dependency}",
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

        if exact.class_name and exact.pipeline_cls is None:
            code = "DIFFUSERS_VERSION_TOO_OLD"
            message = exact.runtime_reason
        elif not loaders:
            code = "PIPELINE_CLASS_NOT_AVAILABLE"
            message = "Aucun loader Diffusers exploitable n'est installe."
        else:
            code = "INVALID_MODEL_SNAPSHOT"
            message = "Tous les loaders Diffusers ont refuse le snapshot."
        raise PipelineResolutionError(message, code=code, attempts=attempts)
