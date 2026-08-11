"""Runtime Diffusers réel et gestion sûre des snapshots de modèles.

Version corrigée pour :
- fusion des capacités catalogue/worker ;
- distinction DOWNLOADED / INSTALLED / READY ;
- revalidation des snapshots déjà présents ;
- chargement/rechargement du pipeline selon la capacité demandée ;
- export vidéo MP4 réel à partir de toutes les frames ;
- maintien de la compatibilité runtime après unload ;
- retries/fallback HF-XET existants conservés.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.inspectors import inspect_model_metadata
from .adapters.registry import PipelineRegistry
from .capability_resolver import CapabilityResolver
from .normalizers import InputNormalizer, NormalizationError, OutputNormalizer, VIDEO_CAPABILITIES
from .pipeline_resolver import PipelineResolutionError, PipelineResolver
from .dependency_installer import DependencyInstaller
from .dependency_resolver import DependencyResolutionError, DependencyResolver
from .model_profile import ModelRuntimeProfile
from .resolution_resolver import ResolutionResolver
from .config import Settings
from .dtype_resolver import DTypeResolver, PrecisionPlan
from .generation_progress import GenerationProgressReporter
from .schemas import CompatibilityStatus, JobState, ModelState


class WorkerError(RuntimeError):
    """Erreur métier exposable au backend sans traceback ni secret."""

    def __init__(
        self,
        message: str,
        status_code: int = 409,
        *,
        code: str = "WORKER_ERROR",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


@dataclass(slots=True)
class LoadedModel:
    model_id: str
    repository: str
    revision: str
    device: str
    loaded_at: float
    validation_test: bool
    precision: str
    load_benchmark: dict[str, Any]
    pipeline: Any
    capability: str | None = None
    metadata: dict[str, Any] | None = None
    precision_plan: PrecisionPlan | None = None


@dataclass(frozen=True, slots=True)
class RuntimeImports:
    """Dépendances paresseuses du runtime, avec un contrat stable et lisible."""

    torch: Any
    hf_api: Any
    snapshot_download: Any


class RuntimeManager:
    """Propriétaire unique des pipelines et des états de jobs du worker."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self._lock = threading.RLock()
        self._loaded: dict[str, LoadedModel] = {}
        self._model_states: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._runtime_modules: RuntimeImports | None = None
        self._runtime_error: str | None = None
        self._registry = PipelineRegistry()
        self._pipeline_resolver = PipelineResolver()
        self._dependency_resolver = DependencyResolver()
        self._dependency_installer = DependencyInstaller(
            self.settings.runtime_dependencies_path
        )
        self._dtype_resolver = DTypeResolver()
        self._resolution_resolver = ResolutionResolver()
        self._input_normalizer = InputNormalizer(self.settings.work_dir)
        self._output_normalizer = OutputNormalizer(self.settings.work_dir)

    @staticmethod
    def _safe_segment(value: str) -> str:
        cleaned = "".join(
            character
            for character in value
            if character.isalnum() or character in {"-", "_", "."}
        )
        if not cleaned or cleaned != value:
            raise WorkerError("Identifiant de modèle ou révision invalide.", 422)
        return cleaned

    def _model_root(self, model_id: str) -> Path:
        return self.settings.models_dir / self._safe_segment(model_id)

    @staticmethod
    def _capability_order() -> list[str]:
        return [
            "TEXT_TO_IMAGE",
            "IMAGE_TO_IMAGE",
            "INPAINTING",
            "OUTPAINTING",
            "IMAGE_VARIATION",
            "IMAGE_UPSCALE",
            "CONTROLLED_IMAGE_GENERATION",
            "TEXT_TO_VIDEO",
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
            "VIDEO_TO_VIDEO",
            "VIDEO_INPAINTING",
            "VIDEO_UPSCALE",
        ]

    def _active_pointer(self, model_id: str) -> Path:
        return self._model_root(model_id) / "active.json"

    @staticmethod
    def _log_model_state(
        model_id: str,
        from_state: str,
        to_state: str,
        reason: str | None = None,
    ) -> None:
        suffix = f" reason={reason}" if reason else ""
        print(f"MODEL_STATE {model_id} {from_state} -> {to_state}{suffix}")

    def _resolve_generation_inputs(self, request: dict[str, Any], pipeline: Any) -> dict[str, Any]:
        accepted = set(inspect.signature(pipeline.__call__).parameters)
        try:
            return self._input_normalizer.normalize(request, accepted)
        except NormalizationError as error:
            raise WorkerError(str(error), 422, code=error.code) from error

    def _active_snapshot(self, model_id: str) -> tuple[Path, dict[str, Any]]:
        pointer = self._active_pointer(model_id)
        if not pointer.is_file():
            raise WorkerError("Le modèle n'est pas installé.", 404)
        metadata = json.loads(pointer.read_text(encoding="utf-8"))
        revision = self._safe_segment(str(metadata["revision"]))
        snapshot = self._model_root(model_id) / revision
        if not snapshot.is_dir():
            raise WorkerError("Le snapshot actif est absent du cache.", 409)
        return snapshot, metadata

    def check_compatibility(self, request: dict[str, Any]) -> dict[str, Any]:
        class_name = request.get("pipeline_class")
        metadata = {
            "class_name": class_name,
            "library_name": request.get("library_name"),
            "pipeline_tag": request.get("pipeline_tag"),
            "raw_tags": request.get("tags") or [],
            "architectures": request.get("architectures") or [],
            "base_models": request.get("base_models") or [],
            "trust_remote_code": request.get("trust_remote_code") is True,
            "model_index": {"_class_name": class_name} if class_name else {},
            "config": {},
        }
        if self._pipeline_resolver.requires_remote_code(metadata):
            return {
                "compatibility_status": CompatibilityStatus.UNSUPPORTED,
                "runtime_supported": False,
                "runtime_capabilities": [],
                "pipeline_class": class_name,
                "runtime_reason": "REMOTE_CODE_REQUIRED",
                "error_code": "REMOTE_CODE_REQUIRED",
            }
        library = str(metadata.get("library_name") or "").lower()
        if library not in {"", "diffusers"}:
            return {
                "compatibility_status": CompatibilityStatus.UNSUPPORTED,
                "runtime_supported": False,
                "runtime_capabilities": [],
                "pipeline_class": class_name,
                "runtime_reason": f"Bibliotheque non prise en charge: {library}",
                "error_code": "UNSUPPORTED_LIBRARY",
            }
        try:
            resolution = self._pipeline_resolver.resolve_class(metadata)
        except PipelineResolutionError as error:
            return {
                "compatibility_status": CompatibilityStatus.UNSUPPORTED,
                "runtime_supported": False,
                "runtime_capabilities": [],
                "pipeline_class": class_name,
                "runtime_reason": str(error),
                "error_code": error.code,
                "dependency": error.dependency,
            }
        capability_sets = CapabilityResolver().describe(metadata, resolution.pipeline_cls)
        capabilities = capability_sets["runtime_capabilities"]
        status = CompatibilityStatus.SUPPORTED
        error_code = None
        if not resolution.runtime_supported:
            if resolution.class_name:
                status = CompatibilityStatus.UNSUPPORTED
                error_code = "DIFFUSERS_VERSION_TOO_OLD"
            else:
                status = CompatibilityStatus.UNKNOWN
        return {
            "compatibility_status": status,
            "runtime_supported": resolution.runtime_supported,
            "runtime_capabilities": capabilities,
            "declared_capabilities": capability_sets["declared_capabilities"],
            "display_capabilities": capability_sets["display_capabilities"],
            "pipeline_class": resolution.class_name,
            "runtime_reason": resolution.runtime_reason,
            "error_code": error_code,
        }

    @staticmethod
    def _metadata_with_capabilities(
        metadata: dict[str, Any],
        fallback_capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        # IMPORTANT : les capacités du backend ne doivent jamais être injectées
        # dans les métadonnées détectées. Sinon un modèle incompatible peut
        # devenir artificiellement TEXT_TO_VIDEO/IMAGE_TO_VIDEO et un adapter
        # spécialisé l'accepte à tort.
        del fallback_capabilities
        result = dict(metadata)
        merged: list[str] = []
        seen: set[str] = set()
        for capability in metadata.get("capabilities") or []:
            if not isinstance(capability, str):
                continue
            normalized = capability.strip().upper()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        result["capabilities"] = merged
        return result

    def _resolve_supported_capability(
        self,
        metadata: dict[str, Any],
        fallback_capabilities: list[str] | None = None,
    ) -> str | None:
        effective_metadata = self._metadata_with_capabilities(metadata)
        detected = set(effective_metadata.get("capabilities") or [])
        requested = {
            str(capability).strip().upper()
            for capability in (fallback_capabilities or [])
            if isinstance(capability, str) and capability.strip()
        }

        if detected and requested:
            candidates = detected & requested
        elif detected:
            candidates = detected
        elif requested:
            # Seulement pour permettre au GenericDiffusersAdapter de valider
            # une classe Diffusers réellement existante quand les métadonnées
            # n'annoncent aucune capability. Les capacités ne sont PAS ajoutées
            # à effective_metadata.
            candidates = requested
        else:
            candidates = set(self._capability_order())

        for candidate in self._capability_order():
            if candidate not in candidates:
                continue
            adapter = self._registry.select_for_capability(
                effective_metadata,
                candidate,
            )
            if adapter is not None:
                return candidate
        return None

    def _unsupported_pipeline_error(
        self,
        metadata: dict[str, Any],
        capability: str | None = None,
    ) -> WorkerError:
        if self._pipeline_resolver.requires_remote_code(metadata):
            return WorkerError(
                "Le snapshot exige trust_remote_code.",
                422,
                code="REMOTE_CODE_REQUIRED",
            )
        library = str(metadata.get("library_name") or "diffusers").lower()
        if library not in {"", "diffusers"}:
            return WorkerError(
                f"Bibliotheque runtime non prise en charge: {library}",
                422,
                code="UNSUPPORTED_LIBRARY",
            )
        try:
            resolution = self._pipeline_resolver.resolve_class(metadata)
        except PipelineResolutionError as error:
            return WorkerError(str(error), 422, code=error.code)
        if not resolution.runtime_supported:
            code = (
                "DIFFUSERS_VERSION_TOO_OLD"
                if resolution.class_name
                else "PIPELINE_CLASS_NOT_AVAILABLE"
            )
            return WorkerError(resolution.runtime_reason, 422, code=code)
        if capability:
            return WorkerError(
                f"La pipeline {resolution.class_name} ne declare pas {capability}.",
                422,
                code="PIPELINE_CAPABILITY_NOT_AVAILABLE",
            )
        return WorkerError(
            "La classe Diffusers existe, mais aucun loader generique ou specialise n'a fonctionne.",
            422,
            code="PIPELINE_UNSUPPORTED",
        )

    @staticmethod
    def _read_manifest(snapshot: Path) -> dict[str, Any]:
        path = snapshot / "vidioai-model.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _effective_snapshot_metadata(
        self,
        snapshot: Path,
        fallback_capabilities: list[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        manifest = self._read_manifest(snapshot)
        metadata = inspect_model_metadata(snapshot)
        metadata = self._metadata_with_capabilities(metadata)
        return metadata, manifest

    def _prepare_dependencies(
        self,
        snapshot: Path,
        metadata: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for spec in self._dependency_resolver.requirements_from_snapshot(
            snapshot, metadata
        ):
            before = self._dependency_installer.status(
                spec.import_name, required_by="quantization_config"
            )
            if model_id is not None:
                pending = {
                    **before,
                    "status": before["status"],
                }
                with self._lock:
                    self._model_states[model_id].update(
                        state=ModelState.RESOLVING_DEPENDENCIES,
                        runtime_dependencies=[pending],
                    )

            def report_dependency(status: str) -> None:
                if model_id is None:
                    return
                state = {
                    "DOWNLOADING": ModelState.DOWNLOADING_DEPENDENCY,
                    "INSTALLING": ModelState.INSTALLING_DEPENDENCIES,
                }.get(status, ModelState.RESOLVING_DEPENDENCIES)
                with self._lock:
                    self._model_states[model_id].update(
                        state=state,
                        runtime_dependencies=[{**before, "status": status}],
                    )

            installed = self._dependency_installer.ensure(
                spec.import_name,
                required_by="quantization_config",
                progress=report_dependency,
            )
            records.append(installed)
            if model_id is not None:
                with self._lock:
                    self._model_states[model_id].update(
                        state=ModelState.RESOLVING_DEPENDENCIES,
                        runtime_dependencies=list(records),
                    )
        return records

    @staticmethod
    def _merge_dependency_records(
        *groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for group in groups:
            for record in group:
                import_name = str(record.get("import_name") or "")
                if import_name:
                    merged[import_name] = record
        return [merged[key] for key in sorted(merged)]

    @staticmethod
    def _persist_runtime_dependencies(
        snapshot: Path, manifest: dict[str, Any], records: list[dict[str, Any]]
    ) -> None:
        if not records:
            return
        manifest = {**manifest, "runtime_dependencies": records}
        path = snapshot / "vidioai-model.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _metadata_runtime_supported(metadata: dict[str, Any]) -> bool:
        return bool(
            metadata.get("runtime_supported") is True
            or metadata.get("compatibility_status")
            == CompatibilityStatus.SUPPORTED
        )

    def _imports(self) -> RuntimeImports:
        """Import paresseux : /health reste disponible même si CUDA est cassé."""
        if self._runtime_modules is not None:
            return self._runtime_modules
        if self._runtime_error is not None:
            raise WorkerError(self._runtime_error, 503)
        try:
            import torch
            from huggingface_hub import HfApi, snapshot_download

            self._runtime_modules = RuntimeImports(
                torch=torch,
                hf_api=HfApi,
                snapshot_download=snapshot_download,
            )
            return self._runtime_modules
        except Exception as error:
            self._runtime_error = (
                f"Runtime IA indisponible: {type(error).__name__}: {error}"
            )
            raise WorkerError(self._runtime_error, 503) from error

    @staticmethod
    def _hf_token() -> str | None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            return None
        cleaned = token.strip()
        return cleaned or None

    @staticmethod
    def _hf_exception_type_names(error: Exception) -> set[str]:
        """Retourne les types HF sans importer huggingface_hub au démarrage."""
        return {error_type.__name__ for error_type in type(error).__mro__}

    @classmethod
    def _classify_hf_error(cls, error: Exception) -> str | None:
        """Classe une erreur HF par son type, jamais par son message traduit."""
        names = cls._hf_exception_type_names(error)
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if "GatedRepoError" in names or status_code in {401, 403}:
            return "HF_ACCESS_DENIED"
        if "RevisionNotFoundError" in names:
            return "HF_REVISION_NOT_FOUND"
        if "RepositoryNotFoundError" in names:
            return "HF_MODEL_NOT_FOUND"
        if names.intersection({"RemoteEntryNotFoundError", "EntryNotFoundError"}):
            return "HF_FILE_NOT_FOUND"
        return None

    @staticmethod
    def _looks_like_timeout(error: Exception) -> bool:
        name = type(error).__name__.lower()
        message = str(error).lower()
        return "timeout" in name or "timed out" in message

    @staticmethod
    def _looks_like_xet_reconstruction_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "background writer channel closed" in message
            or "file reconstruction error" in message
            or "internal writer error" in message
            or "hf-xet" in message
            or ("xet" in message and "reconstruct" in message)
        )

    @staticmethod
    def _is_writable_directory(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".vidioai-write-{uuid.uuid4()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    @staticmethod
    def _available_disk_bytes(path: Path) -> int:
        stats = os.statvfs(path)
        return int(stats.f_frsize) * int(stats.f_bavail)

    @staticmethod
    def _available_inodes(path: Path) -> int:
        stats = os.statvfs(path)
        return int(stats.f_favail)

    def _precheck_download_environment(self, required_bytes: int) -> None:
        models_dir = self.settings.models_dir
        cache_dir = Path(
            os.getenv("HF_HUB_CACHE")
            or os.getenv("HUGGINGFACE_HUB_CACHE")
            or (self.settings.hf_home / "hub")
        )
        xet_dir = Path(os.getenv("HF_XET_CACHE") or (self.settings.hf_home / "xet"))
        tmp_dir = Path(os.getenv("TMPDIR") or self.settings.work_dir)

        for path, code, message in [
            (
                models_dir,
                "SCRATCH_NOT_WRITABLE",
                "Le dossier Scratch des modèles n'est pas inscriptible.",
            ),
            (
                cache_dir,
                "CACHE_NOT_WRITABLE",
                "Le cache Hugging Face n'est pas inscriptible.",
            ),
            (
                xet_dir,
                "CACHE_NOT_WRITABLE",
                "Le cache HF-XET n'est pas inscriptible.",
            ),
            (
                tmp_dir,
                "SCRATCH_NOT_WRITABLE",
                "Le dossier temporaire du worker n'est pas inscriptible.",
            ),
        ]:
            if not self._is_writable_directory(path):
                raise WorkerError(message, 422, code=code)

        disk_required = max(required_bytes, self.settings.minimum_weights_bytes)
        for path, code in [
            (models_dir, "INSUFFICIENT_DISK_SPACE"),
            (cache_dir, "INSUFFICIENT_DISK_SPACE"),
            (tmp_dir, "INSUFFICIENT_DISK_SPACE"),
        ]:
            if self._available_disk_bytes(path) < disk_required:
                raise WorkerError(
                    f"Espace disque insuffisant sur {path}.",
                    422,
                    code=code,
                    retryable=False,
                )

        for path in (models_dir, cache_dir, tmp_dir):
            if self._available_inodes(path) < 64:
                raise WorkerError(
                    f"Inodes insuffisants sur {path}.",
                    422,
                    code="INSUFFICIENT_INODES",
                    retryable=False,
                )

    @staticmethod
    def _estimate_required_download_bytes(model_info: Any) -> int:
        total = 0
        siblings = getattr(model_info, "siblings", []) or []
        for sibling in siblings:
            size = getattr(sibling, "size", None)
            if isinstance(size, int) and size > 0:
                total += size
        if total <= 0:
            return 2 * 1024 * 1024 * 1024
        return int(total * 2.2)

    @staticmethod
    def _clear_partial_directory(path: Path) -> None:
        if not path.exists():
            return
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    @staticmethod
    def _directory_has_files(path: Path) -> bool:
        return path.is_dir() and any(item.is_file() for item in path.rglob("*"))

    @staticmethod
    def _snapshot_download_with_env(
        snapshot_download: Any,
        *,
        repo_id: str,
        revision: str,
        local_dir: Path,
        cache_dir: Path,
        token: str | None,
        disable_xet: bool,
        sequential_reconstruct: bool,
    ) -> None:
        previous_disable = os.getenv("HF_HUB_DISABLE_XET")
        previous_seq = os.getenv("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY")
        try:
            if disable_xet:
                os.environ["HF_HUB_DISABLE_XET"] = "1"
            else:
                os.environ.pop("HF_HUB_DISABLE_XET", None)

            if sequential_reconstruct:
                os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
            else:
                os.environ.pop("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", None)

            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=local_dir,
                cache_dir=cache_dir,
                token=token,
                ignore_patterns=[
                    "*.ckpt",
                    "*.onnx",
                    "*.msgpack",
                    "*.h5",
                    "*.tflite",
                ],
            )
        finally:
            if previous_disable is None:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
            else:
                os.environ["HF_HUB_DISABLE_XET"] = previous_disable

            if previous_seq is None:
                os.environ.pop("HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY", None)
            else:
                os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = previous_seq

    def _preflight_remote_metadata(
        self,
        repository: str,
        revision: str,
        local_dir: Path,
        token: str | None,
        model_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Ne recupere que les JSON legers avant les poids du snapshot."""
        try:
            from huggingface_hub import hf_hub_download
        except (ImportError, ModuleNotFoundError):
            return None

        downloaded = False
        for filename in ("model_index.json", "config.json"):
            try:
                hf_hub_download(
                    repo_id=repository,
                    filename=filename,
                    revision=revision,
                    local_dir=local_dir,
                    token=token,
                )
                downloaded = True
            except Exception as error:
                # Ces deux fichiers sont optionnels et indépendants. Un 404 de
                # fichier ne dit rien sur l'existence du repository, déjà
                # confirmée par HfApi.model_info(). Toute autre erreur remonte.
                if self._classify_hf_error(error) == "HF_FILE_NOT_FOUND":
                    continue
                raise
        if not downloaded:
            return {
                "repository": repository,
                "revision": revision,
                "capabilities": [],
                "library_name": "diffusers",
                "class_name": None,
                "compatibility_status": CompatibilityStatus.UNKNOWN,
                "runtime_supported": False,
                "runtime_reason": (
                    "Métadonnées publiques optionnelles absentes; "
                    "validation du snapshot requise."
                ),
            }

        metadata = inspect_model_metadata(local_dir)
        if self._pipeline_resolver.requires_remote_code(metadata):
            raise WorkerError(
                "Le modele exige trust_remote_code; execution distante refusee.",
                422,
                code="REMOTE_CODE_REQUIRED",
            )
        library = str(metadata.get("library_name") or "diffusers").lower()
        if library not in {"", "diffusers"}:
            raise WorkerError(
                f"Bibliotheque runtime non prise en charge: {library}",
                422,
                code="UNSUPPORTED_LIBRARY",
            )
        resolution, repaired_dependencies = self._dependency_resolver.load_with_repair(
            lambda: self._pipeline_resolver.resolve_class(metadata),
            self._dependency_installer,
        )
        metadata["runtime_dependencies"] = repaired_dependencies
        if model_id is not None and repaired_dependencies:
            with self._lock:
                self._model_states[model_id].update(
                    state=ModelState.RESOLVING_DEPENDENCIES,
                    runtime_dependencies=repaired_dependencies,
                )
        if not resolution.runtime_supported:
            if resolution.class_name:
                raise WorkerError(
                    resolution.runtime_reason,
                    422,
                    code="DIFFUSERS_VERSION_TOO_OLD",
                )
            metadata["compatibility_status"] = CompatibilityStatus.UNKNOWN
            metadata["runtime_supported"] = False
            metadata["runtime_reason"] = (
                "Métadonnées publiques incomplètes; validation du snapshot requise."
            )
        capability_sets = CapabilityResolver().describe(metadata, resolution.pipeline_cls)
        metadata.update(capability_sets)
        metadata["capabilities"] = capability_sets["display_capabilities"]
        return metadata

    def runtime_status(self) -> dict[str, Any]:
        configuration_errors = self.settings.configuration_errors()
        scratch = self._scratch_status()
        if self.settings.gpu_required and not scratch["scratch_mount_ok"]:
            configuration_errors.append(
                "SCRATCH_FILESYSTEM_INVALID: /models, /cache, /work et /worker-work "
                "doivent utiliser le Scratch dédié."
            )
        try:
            torch = self._imports().torch
            runtime_available = True
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = getattr(torch.version, "cuda", None)
            torch_version = getattr(torch, "__version__", None)
        except Exception as error:
            runtime_available = False
            cuda_available = False
            cuda_version = None
            torch_version = None
            message = (
                str(error)
                if isinstance(error, WorkerError)
                else f"Runtime IA indisponible: {type(error).__name__}: {error}"
            )
            if message not in configuration_errors:
                configuration_errors.append(message)

        ready = (
            not configuration_errors
            and runtime_available
            and (cuda_available or not self.settings.gpu_required)
        )
        return {
            "ready": ready,
            "profile": self.settings.app_env,
            "gpu_required": self.settings.gpu_required,
            "runtime_available": runtime_available,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
            "versions": self.runtime_versions(),
            **scratch,
            "errors": configuration_errors,
        }

    def _scratch_status(self) -> dict[str, Any]:
        cache_dir = self.settings.hf_home.parent
        paths = {
            "models": self.settings.models_dir,
            "cache": cache_dir,
            "work": self.settings.outputs_dir,
            "worker_work": self.settings.work_dir,
        }
        try:
            devices = {name: path.stat().st_dev for name, path in paths.items()}
            root_device = Path("/").stat().st_dev
            writable = all(os.access(path, os.W_OK | os.X_OK) for path in paths.values())
            usage = shutil.disk_usage(self.settings.models_dir)
            minimum = int(os.getenv("VIDIOAI_MIN_SCRATCH_TOTAL_BYTES", "214748364800"))
            expected_layout = True
            if self.settings.gpu_required:
                expected_layout = {
                    name: str(path)
                    for name, path in paths.items()
                } == {
                    "models": "/models",
                    "cache": "/cache",
                    "work": "/work",
                    "worker_work": "/worker-work",
                }
            mount_ok = (
                writable
                and len(set(devices.values())) == 1
                and devices["models"] != root_device
                and usage.total >= minimum
                and expected_layout
            )
            return {
                "scratch_mount_ok": mount_ok,
                "scratch_filesystem": f"device:{devices['models']}",
                "scratch_total_bytes": usage.total,
                "scratch_available_bytes": usage.free,
            }
        except (OSError, ValueError) as error:
            return {
                "scratch_mount_ok": False,
                "scratch_filesystem": f"unavailable:{type(error).__name__}",
                "scratch_total_bytes": 0,
                "scratch_available_bytes": 0,
            }

    @staticmethod
    def runtime_versions() -> dict[str, str | None]:
        versions: dict[str, str | None] = {}
        for package in (
            "torch",
            "diffusers",
            "transformers",
            "huggingface_hub",
            "accelerate",
            "bitsandbytes",
        ):
            distribution = "huggingface-hub" if package == "huggingface_hub" else package
            try:
                versions[package] = importlib.metadata.version(distribution)
            except Exception:
                versions[package] = None
        return versions

    def log_runtime_versions(self) -> None:
        versions = self.runtime_versions()
        values = " ".join(f"{name}={value or 'missing'}" for name, value in versions.items())
        print(f"RUNTIME_VERSIONS {values}")

    @staticmethod
    def _nvidia_metrics() -> dict[str, Any] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            line = result.stdout.strip().splitlines()[0]
            name, utilization, total, used, temperature = [
                value.strip() for value in line.split(",", maxsplit=4)
            ]
            return {
                "name": name,
                "backend": "CUDA",
                "utilization_percent": float(utilization),
                "vram_total_bytes": int(total) * 1024 * 1024,
                "vram_used_bytes": int(used) * 1024 * 1024,
                "temperature_celsius": float(temperature),
            }
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _ram_peak_bytes() -> int:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    def resources(self) -> dict[str, Any]:
        gpu = self._nvidia_metrics()
        with self._lock:
            loaded = [
                {
                    "model_id": model.model_id,
                    "repository": model.repository,
                    "revision": model.revision,
                    "device": model.device,
                    "validation_test": model.validation_test,
                    "capability": model.capability,
                }
                for model in self._loaded.values()
            ]
            active_jobs = sum(
                1
                for job in self._jobs.values()
                if job["state"] in {JobState.QUEUED, JobState.RUNNING}
            )
        return {
            "gpu": gpu,
            "gpu_status": "available" if gpu else "unavailable",
            "worker_status": "ready" if self.runtime_status()["ready"] else "not_ready",
            "active_jobs": active_jobs,
            "loaded_models": loaded,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_snapshot(self, snapshot: Path) -> dict[str, Any]:
        """Valide structure, présence réelle des poids, tailles et empreintes."""
        model_index = snapshot / "model_index.json"
        if not model_index.is_file():
            raise WorkerError("model_index.json est absent du snapshot.", 422)
        try:
            parsed_index = json.loads(model_index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkerError("model_index.json est invalide.", 422) from error
        has_diffusers_component = any(
            isinstance(component, list)
            and component
            and str(component[0]).strip().lower() == "diffusers"
            for key, component in parsed_index.items()
            if not str(key).startswith("_")
        )
        if "_class_name" not in parsed_index and not has_diffusers_component:
            raise WorkerError(
                "Le manifest ne déclare ni pipeline ni composant Diffusers.",
                422,
            )

        weights = sorted(snapshot.rglob("*.safetensors"))
        if not weights:
            weights = sorted(snapshot.rglob("*.bin"))
        total_weights = sum(path.stat().st_size for path in weights)
        if not weights or total_weights < self.settings.minimum_weights_bytes:
            raise WorkerError(
                "Le snapshot ne contient pas de poids Diffusers cohérents.",
                422,
            )

        files = []
        for path in sorted(file for file in snapshot.rglob("*") if file.is_file()):
            if ".cache" in path.parts:
                continue
            files.append(
                {
                    "path": str(path.relative_to(snapshot)),
                    "size": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
            )
        return {
            "weights_valid": True,
            "weights_bytes": total_weights,
            "weights_files": len(weights),
            "files": files,
        }

    def _mark_failed(
        self,
        model_id: str,
        *,
        error: WorkerError,
        downloaded: bool,
        validated: bool = False,
        installed: bool = False,
    ) -> None:
        with self._lock:
            self._model_states[model_id] = {
                "model_id": model_id,
                "state": ModelState.FAILED,
                "downloaded": downloaded,
                "validated": validated,
                "installed": installed,
                "loaded": False,
                "ready": False,
                "runtime_compatible": False,
                "validation_test": False,
                "error_code": error.code,
                "retryable": error.retryable,
                "error": str(error),
            }

    def _cached_install_status(
        self,
        model_id: str,
        snapshot: Path,
        pointer: dict[str, Any],
        capabilities: list[str],
    ) -> dict[str, Any]:
        validation = self.validate_snapshot(snapshot)
        runtime_dependencies = self._prepare_dependencies(snapshot)
        metadata, manifest = self._effective_snapshot_metadata(snapshot, capabilities)
        resolved_capability = self._resolve_supported_capability(
            metadata,
            manifest.get("requested_capabilities")
            or manifest.get("capabilities")
            or capabilities,
        )
        if resolved_capability is None:
            raise self._unsupported_pipeline_error(metadata)

        runtime_capabilities = list(metadata.get("capabilities") or [])
        cached = {
            **manifest,
            "model_id": model_id,
            "repository": pointer["repository"],
            "revision": pointer["revision"],
            "capabilities": runtime_capabilities,
            "requested_capabilities": list(capabilities),
            "downloaded": True,
            "validated": True,
            "installed": True,
            "weights_valid": validation["weights_valid"],
            "runtime_available": self.runtime_status()["runtime_available"],
            "runtime_compatible": self._metadata_runtime_supported(metadata),
            "validation_test": False,
            "loaded": False,
            "ready": False,
            "state": ModelState.INSTALLED,
            "runtime_dependencies": self._merge_dependency_records(
                list(manifest.get("runtime_dependencies") or []),
                runtime_dependencies,
            ),
            **validation,
        }
        self._persist_runtime_dependencies(
            snapshot, cached, cached["runtime_dependencies"]
        )
        with self._lock:
            self._model_states[model_id] = cached
        return cached

    def install_model(
        self,
        model_id: str,
        repository: str,
        revision: str,
        capabilities: list[str],
    ) -> dict[str, Any]:
        model_id = self._safe_segment(model_id)
        self._log_model_state(model_id, "DISCOVERED", "COMPATIBILITY_CHECK")

        # Réutilisation du snapshot actif seulement s'il correspond exactement à
        # la révision demandée ET repasse la validation runtime actuelle.
        try:
            snapshot, pointer = self._active_snapshot(model_id)
        except WorkerError:
            snapshot = None
            pointer = None

        if snapshot is not None and pointer is not None and pointer.get("revision") == revision:
            try:
                return self._cached_install_status(
                    model_id,
                    snapshot,
                    pointer,
                    capabilities,
                )
            except WorkerError as error:
                # Un ancien snapshot valide mais incompatible ne doit pas être
                # présenté comme installé. On ne retélécharge pas la même révision
                # uniquement pour masquer une erreur de pipeline.
                if error.code in {
                    "PIPELINE_UNSUPPORTED",
                    "DIFFUSERS_VERSION_TOO_OLD",
                    "PIPELINE_CLASS_NOT_AVAILABLE",
                    "REMOTE_CODE_REQUIRED",
                    "UNSUPPORTED_LIBRARY",
                }:
                    self._mark_failed(
                        model_id,
                        error=error,
                        downloaded=True,
                    )
                    raise
                # Snapshot incomplet/corrompu : téléchargement propre autorisé.

        with self._lock:
            self._model_states[model_id] = {
                "model_id": model_id,
                "state": ModelState.COMPATIBILITY_CHECK,
                "downloaded": False,
                "validated": False,
                "installed": False,
                "loaded": False,
                "ready": False,
                "error": None,
            }

        temporary = (
            self.settings.models_dir
            / ".downloads"
            / f"download-{model_id}-{uuid.uuid4()}"
        )
        temporary.mkdir(parents=True, exist_ok=True)
        weights_download_started = False

        try:
            runtime_imports = self._imports()
            HfApi = runtime_imports.hf_api
            snapshot_download = runtime_imports.snapshot_download
            hf_token = self._hf_token()
            info = HfApi(token=hf_token).model_info(repository, revision=revision)
            resolved_revision = self._safe_segment(info.sha)

            preflight_metadata = self._preflight_remote_metadata(
                repository,
                resolved_revision,
                temporary,
                hf_token,
                model_id,
            )
            if preflight_metadata is not None:
                preflight_capabilities = preflight_metadata.get("capabilities") or []
                requested = {
                    str(value).upper() for value in capabilities if isinstance(value, str)
                }
                if requested and preflight_capabilities and not requested.intersection(preflight_capabilities):
                    raise WorkerError(
                        "La pipeline disponible ne declare aucune des capabilities demandees.",
                        422,
                        code="PIPELINE_CAPABILITY_NOT_AVAILABLE",
                    )

            required_bytes = self._estimate_required_download_bytes(info)
            self._precheck_download_environment(required_bytes)
            self._log_model_state(model_id, "COMPATIBILITY_CHECK", "DOWNLOADING")
            with self._lock:
                self._model_states[model_id]["state"] = ModelState.DOWNLOADING

            xet_retries = max(1, int(os.getenv("VIDIOAI_HF_XET_RETRIES", "2")))
            allow_no_xet_fallback = os.getenv(
                "VIDIOAI_ENABLE_HF_XET_FALLBACK",
                "true",
            ).strip().lower() in {"1", "true", "yes", "on"}

            last_xet_error: Exception | None = None
            for attempt in range(1, xet_retries + 1):
                try:
                    weights_download_started = True
                    self._snapshot_download_with_env(
                        snapshot_download,
                        repo_id=repository,
                        revision=resolved_revision,
                        local_dir=temporary,
                        cache_dir=self.settings.hf_home,
                        token=hf_token,
                        disable_xet=False,
                        sequential_reconstruct=attempt > 1,
                    )
                    last_xet_error = None
                    break
                except Exception as error:
                    if not self._looks_like_xet_reconstruction_error(error):
                        raise
                    last_xet_error = error
                    self._clear_partial_directory(temporary)

            if last_xet_error is not None:
                if not allow_no_xet_fallback:
                    raise WorkerError(
                        "HF_XET_RECONSTRUCTION_ERROR: reconstruction HF-XET échouée après retries.",
                        502,
                        code="HF_XET_RECONSTRUCTION_ERROR",
                        retryable=True,
                    ) from last_xet_error
                try:
                    self._snapshot_download_with_env(
                        snapshot_download,
                        repo_id=repository,
                        revision=resolved_revision,
                        local_dir=temporary,
                        cache_dir=self.settings.hf_home,
                        token=hf_token,
                        disable_xet=True,
                        sequential_reconstruct=False,
                    )
                except Exception as fallback_error:
                    raise WorkerError(
                        "HF_XET_RECONSTRUCTION_ERROR: reconstruction HF-XET échouée, "
                        f"fallback sans XET impossible ({type(fallback_error).__name__}).",
                        502,
                        code="HF_XET_RECONSTRUCTION_ERROR",
                        retryable=True,
                    ) from fallback_error

            self._log_model_state(model_id, "DOWNLOADING", "DOWNLOADED")
            with self._lock:
                self._model_states[model_id].update(
                    state=ModelState.DOWNLOADED,
                    downloaded=True,
                )
            self._log_model_state(model_id, "DOWNLOADED", "VALIDATING")
            with self._lock:
                self._model_states[model_id]["state"] = ModelState.VALIDATING
            validation = self.validate_snapshot(temporary)

            self._log_model_state(
                model_id, "VALIDATING", "RESOLVING_DEPENDENCIES"
            )
            with self._lock:
                self._model_states[model_id][
                    "state"
                ] = ModelState.RESOLVING_DEPENDENCIES
            runtime_dependencies = self._merge_dependency_records(
                list(
                    (preflight_metadata or {}).get("runtime_dependencies") or []
                ),
                self._prepare_dependencies(temporary, model_id=model_id),
            )

            metadata = inspect_model_metadata(temporary)
            metadata = self._metadata_with_capabilities(metadata)
            resolved_capability = self._resolve_supported_capability(
                metadata,
                capabilities,
            )
            if resolved_capability is None:
                raise self._unsupported_pipeline_error(metadata)

            runtime_capabilities = list(metadata.get("capabilities") or [])
            destination = self._model_root(model_id) / resolved_revision
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)

            manifest = {
                "model_id": model_id,
                "repository": repository,
                "revision": resolved_revision,
                "capabilities": runtime_capabilities,
                "requested_capabilities": list(capabilities),
                "downloaded": True,
                "validated": True,
                "installed": True,
                "weights_valid": True,
                "runtime_available": self.runtime_status()["runtime_available"],
                "runtime_compatible": self._metadata_runtime_supported(metadata),
                "validation_test": False,
                "loaded": False,
                "ready": False,
                "state": ModelState.INSTALLED,
                "runtime_dependencies": runtime_dependencies,
                "installed_at": int(time.time()),
                **validation,
            }
            (temporary / "vidioai-model.json").write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, destination)

            pointer_payload = {
                "model_id": model_id,
                "repository": repository,
                "revision": resolved_revision,
            }
            pointer_path = self._active_pointer(model_id)
            pointer_temporary = pointer_path.with_suffix(".json.tmp")
            pointer_temporary.write_text(
                json.dumps(pointer_payload),
                encoding="utf-8",
            )
            os.replace(pointer_temporary, pointer_path)

            with self._lock:
                self._model_states[model_id] = manifest
            self._log_model_state(model_id, "RESOLVING_DEPENDENCIES", "INSTALLED")
            return manifest

        except DependencyResolutionError as error:
            status_code = (
                422
                if error.code
                in {
                    "DEPENDENCY_NOT_ALLOWED",
                    "DEPENDENCY_VERSION_CONFLICT",
                    "DEPENDENCY_PLATFORM_UNSUPPORTED",
                }
                else 502
            )
            worker_error = WorkerError(
                str(error),
                status_code,
                code=error.code,
                retryable=error.code == "DEPENDENCY_INSTALL_FAILED",
            )
            self._mark_failed(
                model_id,
                error=worker_error,
                downloaded=weights_download_started
                and self._directory_has_files(temporary),
                validated=True,
            )
            raise worker_error from error

        except WorkerError as error:
            self._log_model_state(model_id, "VALIDATING", "FAILED", reason=error.code)
            self._mark_failed(
                model_id,
                error=error,
                downloaded=weights_download_started and self._directory_has_files(temporary),
            )
            raise

        except Exception as error:
            hf_error_code = self._classify_hf_error(error)
            if hf_error_code == "HF_ACCESS_DENIED":
                worker_error = WorkerError(
                    "Accès Hugging Face requis: ce repository est protégé "
                    "(gated/private) et nécessite un HF_TOKEN valide.",
                    403,
                    code="HF_ACCESS_DENIED",
                    retryable=False,
                )
            elif hf_error_code == "HF_REVISION_NOT_FOUND":
                worker_error = WorkerError(
                    "Révision Hugging Face introuvable pour ce repository.",
                    404,
                    code="HF_REVISION_NOT_FOUND",
                    retryable=False,
                )
            elif hf_error_code == "HF_MODEL_NOT_FOUND":
                worker_error = WorkerError(
                    "Repository Hugging Face introuvable.",
                    404,
                    code="HF_MODEL_NOT_FOUND",
                    retryable=False,
                )
            elif hf_error_code == "HF_FILE_NOT_FOUND":
                worker_error = WorkerError(
                    "Fichier Hugging Face obligatoire introuvable.",
                    404,
                    code="HF_FILE_NOT_FOUND",
                    retryable=False,
                )
            elif self._looks_like_timeout(error):
                worker_error = WorkerError(
                    "Délai dépassé pendant le téléchargement Hugging Face.",
                    504,
                    code="HF_DOWNLOAD_TIMEOUT",
                    retryable=True,
                )
            elif self._looks_like_xet_reconstruction_error(error):
                worker_error = WorkerError(
                    "Erreur de reconstruction HF-XET pendant le téléchargement.",
                    502,
                    code="HF_XET_RECONSTRUCTION_ERROR",
                    retryable=True,
                )
            else:
                worker_error = WorkerError(
                    f"Installation impossible: {type(error).__name__}: {error}",
                    502,
                    code="HF_DOWNLOAD_ERROR",
                    retryable=False,
                )

            self._mark_failed(
                model_id,
                error=worker_error,
                downloaded=weights_download_started and self._directory_has_files(temporary),
            )
            raise worker_error from error

        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def model_status(self, model_id: str) -> dict[str, Any]:
        model_id = self._safe_segment(model_id)
        with self._lock:
            current = self._model_states.get(model_id)
            if current is not None and current.get("state") in {
                ModelState.DOWNLOADING,
                ModelState.RESOLVING_DEPENDENCIES,
                ModelState.DOWNLOADING_DEPENDENCY,
                ModelState.INSTALLING_DEPENDENCIES,
                ModelState.FAILED,
                ModelState.RUNTIME_UNAVAILABLE,
                ModelState.INCOMPATIBLE,
            }:
                return dict(current)
            if current is not None and self._active_pointer(model_id).is_file():
                return dict(current)

        try:
            snapshot, _ = self._active_snapshot(model_id)
            manifest = self._read_manifest(snapshot)
            if not manifest:
                raise WorkerError("Le manifest VidioAI du snapshot est absent.", 409)

            validation = self.validate_snapshot(snapshot)
            metadata, manifest = self._effective_snapshot_metadata(
                snapshot,
                manifest.get("requested_capabilities") or manifest.get("capabilities"),
            )
            resolved_capability = self._resolve_supported_capability(
                metadata,
                manifest.get("requested_capabilities") or manifest.get("capabilities"),
            )
            if resolved_capability is None:
                runtime_error = self._unsupported_pipeline_error(metadata)
                return {
                    "model_id": model_id,
                    "state": ModelState.FAILED,
                    "downloaded": True,
                    "validated": False,
                    "installed": False,
                    "weights_valid": validation.get("weights_valid", False),
                    "runtime_available": self.runtime_status()["runtime_available"],
                    "runtime_compatible": False,
                    "validation_test": False,
                    "loaded": False,
                    "ready": False,
                    "error_code": runtime_error.code,
                    "error": str(runtime_error),
                }

            with self._lock:
                loaded = self._loaded.get(model_id)

            status = {
                **manifest,
                "downloaded": True,
                "validated": True,
                "installed": True,
                "weights_valid": validation["weights_valid"],
                "runtime_compatible": self._metadata_runtime_supported(metadata),
                "loaded": loaded is not None,
                "ready": loaded is not None and loaded.pipeline is not None,
                "state": (
                    ModelState.READY
                    if loaded is not None and loaded.pipeline is not None
                    else ModelState.INSTALLED
                ),
                "validation_test": loaded is not None and loaded.validation_test,
            }
            return status

        except WorkerError:
            return {
                "model_id": model_id,
                "state": ModelState.NOT_INSTALLED,
                "downloaded": False,
                "validated": False,
                "installed": False,
                "weights_valid": False,
                "runtime_available": self.runtime_status()["runtime_available"],
                "runtime_compatible": False,
                "validation_test": False,
                "loaded": False,
                "ready": False,
            }

    def _precision_plan(
        self,
        torch: Any,
        metadata: dict[str, Any],
        cuda_available: bool,
    ) -> PrecisionPlan:
        bf16_supported = bool(
            cuda_available
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        return self._dtype_resolver.resolve(
            metadata,
            cuda_available=cuda_available,
            bf16_supported=bf16_supported,
        )

    def _load_pipeline(
        self,
        *,
        snapshot: Path,
        metadata: dict[str, Any],
        capability: str,
        adapter: Any,
        device: str,
        dtype: Any,
    ) -> Any:
        pipeline = adapter.load(
            str(snapshot),
            {"torch_dtype": dtype, "device": device},
            {
                "device": device,
                "class_name": metadata.get("class_name"),
                "metadata": metadata,
                "capability": capability,
            },
        )
        if device == "cuda" and hasattr(pipeline, "to"):
            pipeline = pipeline.to(device)
        capability_sets = CapabilityResolver().describe(metadata, pipeline)
        resolved_capabilities = capability_sets["runtime_capabilities"]
        if resolved_capabilities:
            metadata.update(capability_sets)
            metadata["capabilities"] = capability_sets["display_capabilities"]
            if capability not in resolved_capabilities:
                raise PipelineResolutionError(
                    f"La signature chargee ne supporte pas {capability}.",
                    code="PIPELINE_CAPABILITY_NOT_AVAILABLE",
                )

        optional_optimizations = []
        if os.getenv("VIDIOAI_ENABLE_VAE_TILING", "false").lower() in {"1", "true", "yes"}:
            optional_optimizations.append("enable_vae_tiling")
        if os.getenv("VIDIOAI_ENABLE_VAE_SLICING", "false").lower() in {"1", "true", "yes"}:
            optional_optimizations.append("enable_vae_slicing")
        for method_name in optional_optimizations:
            method = getattr(pipeline, method_name, None)
            if callable(method):
                try:
                    method()
                except (NotImplementedError, RuntimeError, ValueError):
                    pass
        return pipeline

    def _validate_loaded_pipeline(
        self,
        *,
        adapter: Any,
        pipeline: Any,
        metadata: dict[str, Any],
        capability: str,
        device: str,
        torch: Any,
    ) -> None:
        # Les modes nécessitant une vraie vidéo/masque/contrôle sont validés à la
        # première génération. Le chargement réel du pipeline reste obligatoire.
        if capability in {
            "VIDEO_TO_VIDEO",
            "VIDEO_INPAINTING",
            "VIDEO_UPSCALE",
            "INPAINTING",
            "CONTROLLED_IMAGE_GENERATION",
        }:
            if not callable(pipeline):
                raise RuntimeError("Le pipeline Diffusers chargé n'est pas appelable.")
            return

        generator = torch.Generator(device=device if device == "cuda" else "cpu")
        profile = ModelRuntimeProfile.from_metadata(metadata, pipeline)
        request: dict[str, Any] = {
            "prompt": "VidioAI runtime validation",
            "negative_prompt": None,
            "width": 64,
            "height": 64,
            "steps": 1,
            "guidance_scale": profile.guidance_scale,
            "frames": 5 if "VIDEO" in capability else None,
            "fps": profile.fps if "VIDEO" in capability else None,
            "duration_seconds": 1 if "VIDEO" in capability else None,
        }
        if "VIDEO" in capability:
            resolution = self._resolution_resolver.resolve(
                quality="480p",
                aspect_ratio="16:9",
                pipeline=pipeline,
                metadata=metadata,
                default_width=profile.width,
                default_height=profile.height,
            )
            request.update(
                {
                    "quality": resolution.requested_quality,
                    "aspect_ratio": resolution.requested_aspect_ratio,
                    "width": resolution.width,
                    "height": resolution.height,
                }
            )

        if capability in {
            "IMAGE_TO_IMAGE",
            "IMAGE_VARIATION",
            "IMAGE_UPSCALE",
            "OUTPAINTING",
            "IMAGE_TO_VIDEO",
            "MULTI_IMAGE_TO_VIDEO",
            "START_END_IMAGE_TO_VIDEO",
            "KEYFRAMES_TO_VIDEO",
        }:
            from PIL import Image

            image = Image.new("RGB", (128, 128), (127, 127, 127))
            request["input_image"] = image
            request["resolved_input_images"] = [image]
            request["input_images"] = [
                {"asset_id": "validation", "order": 0, "role": "start_frame"}
            ]

        output = adapter.generate(
            pipeline,
            {
                "device": device,
                "generator": generator,
                "metadata": metadata,
                "capability": capability,
            },
            request,
        )
        validation_images, validation_frames = self._output_normalizer.extract(
            output,
            video=capability in VIDEO_CAPABILITIES,
        )
        if not validation_images and not validation_frames:
            raise RuntimeError("Le pipeline n'a produit aucune sortie de validation.")

    def load_model(self, model_id: str) -> dict[str, Any]:
        model_id = self._safe_segment(model_id)
        self._log_model_state(model_id, "INSTALLED", "LOADING")
        snapshot, pointer = self._active_snapshot(model_id)
        validation = self.validate_snapshot(snapshot)
        with self._lock:
            self._model_states[model_id] = {
                "model_id": model_id,
                "state": ModelState.LOADING,
                "downloaded": True,
                "validated": True,
                "installed": True,
                "loaded": False,
                "ready": False,
            }
        torch = self._imports().torch
        cuda_available = bool(torch.cuda.is_available())

        if self.settings.gpu_required and not cuda_available:
            status = {
                "model_id": model_id,
                "state": ModelState.RUNTIME_UNAVAILABLE,
                "installed": True,
                "weights_valid": True,
                "runtime_available": True,
                "runtime_compatible": False,
                "validation_test": False,
                "loaded": False,
                "ready": False,
                "error": "CUDA est obligatoire mais indisponible.",
            }
            with self._lock:
                self._model_states[model_id] = status
            raise WorkerError(status["error"], 503, code="RUNTIME_UNAVAILABLE")

        metadata, manifest = self._effective_snapshot_metadata(snapshot)
        capability = self._resolve_supported_capability(
            metadata,
            manifest.get("requested_capabilities") or manifest.get("capabilities"),
        )
        if capability is None:
            runtime_error = self._unsupported_pipeline_error(metadata)
            self._log_model_state(
                model_id,
                "LOADING",
                "FAILED",
                reason=runtime_error.code,
            )
            self._mark_failed(
                model_id,
                error=runtime_error,
                downloaded=True,
                validated=True,
                installed=False,
            )
            raise runtime_error

        adapter = self._registry.select_for_capability(metadata, capability)
        if adapter is None:
            raise self._unsupported_pipeline_error(metadata, capability)

        device = "cuda" if cuda_available else "cpu"
        precision_plan = self._precision_plan(torch, metadata, cuda_available)
        dtype = self._dtype_resolver.materialize(torch, precision_plan)
        precision = precision_plan.precision
        gpu_before = self._nvidia_metrics()
        load_started = time.perf_counter()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()

        try:
            pipeline, repaired_dependencies = self._dependency_resolver.load_with_repair(
                lambda: self._load_pipeline(
                    snapshot=snapshot,
                    metadata=metadata,
                    capability=capability,
                    adapter=adapter,
                    device=device,
                    dtype=dtype,
                ),
                self._dependency_installer,
            )
            runtime_dependencies = self._merge_dependency_records(
                list(manifest.get("runtime_dependencies") or []),
                repaired_dependencies,
            )
            self._persist_runtime_dependencies(
                snapshot, manifest, runtime_dependencies
            )
        except Exception as error:
            code = (
                error.code
                if isinstance(
                    error, (PipelineResolutionError, DependencyResolutionError)
                )
                else "LOAD_FAILED"
            )
            self._log_model_state(model_id, "LOADING", "FAILED", reason=code)
            status = {
                "model_id": model_id,
                "state": ModelState.FAILED,
                "downloaded": True,
                "validated": True,
                "installed": True,
                "weights_valid": True,
                "runtime_available": True,
                "runtime_compatible": False,
                "validation_test": False,
                "loaded": False,
                "ready": False,
                "error": f"Chargement Diffusers impossible: {type(error).__name__}: {error}",
                "error_code": code,
            }
            if isinstance(
                error, (PipelineResolutionError, DependencyResolutionError)
            ) and error.dependency:
                status["dependency"] = error.dependency
            with self._lock:
                self._model_states[model_id] = status
            raise WorkerError(
                status["error"],
                422
                if code
                in {
                    "DIFFUSERS_VERSION_TOO_OLD",
                    "REMOTE_CODE_REQUIRED",
                    "MISSING_DEPENDENCY",
                    "DEPENDENCY_NOT_ALLOWED",
                    "DEPENDENCY_VERSION_CONFLICT",
                    "DEPENDENCY_PLATFORM_UNSUPPORTED",
                    "PIPELINE_CLASS_NOT_AVAILABLE",
                    "INVALID_MODEL_SNAPSHOT",
                }
                else 503,
                code=code,
                retryable=False,
            ) from error

        gpu_after = self._nvidia_metrics()
        process_peak = int(torch.cuda.max_memory_reserved()) if cuda_available else 0
        idle_vram = int((gpu_before or {}).get("vram_used_bytes", 0))
        total_vram = int((gpu_after or gpu_before or {}).get("vram_total_bytes", 0))
        observed_peak = max(
            int((gpu_after or {}).get("vram_used_bytes", 0)),
            idle_vram + process_peak,
        )
        if total_vram:
            observed_peak = min(observed_peak, total_vram)

        load_benchmark = {
            "gpu": str((gpu_after or gpu_before or {}).get("name", "CPU")),
            "vram_idle_bytes": idle_vram,
            "vram_after_load_bytes": int((gpu_after or {}).get("vram_used_bytes", 0)),
            "vram_peak_bytes": observed_peak,
            "ram_peak_bytes": self._ram_peak_bytes(),
            "runtime": "Diffusers",
            "precision": precision,
            "resolution_width": None,
            "resolution_height": None,
            "frames": None,
            "duration_seconds": None,
            "fps": None,
            "batch": 1,
            "attention_implementation": None,
            "vae_tiling": bool(getattr(pipeline, "vae_tiling", False)),
            "cpu_offload": False,
            "model_offload": False,
            "inference_seconds": time.perf_counter() - load_started,
        }

        loaded = LoadedModel(
            model_id=model_id,
            repository=pointer["repository"],
            revision=pointer["revision"],
            device=device,
            loaded_at=time.time(),
            validation_test=False,
            precision=precision,
            load_benchmark=load_benchmark,
            pipeline=pipeline,
            capability=capability,
            metadata=metadata,
            precision_plan=precision_plan,
        )
        precision_plan.components = self._dtype_resolver.inspect_components(pipeline)
        status = {
            "model_id": model_id,
            "state": ModelState.READY,
            "downloaded": True,
            "validated": True,
            "installed": True,
            "weights_valid": validation["weights_valid"],
            "runtime_available": True,
            "runtime_compatible": True,
            "validation_test": False,
            "loaded": True,
            "ready": True,
            "device": device,
            "capability": capability,
            "capabilities": list(metadata.get("capabilities") or []),
            "pipeline_class": metadata.get("class_name"),
            "repository": pointer["repository"],
            "revision": pointer["revision"],
            "benchmark": load_benchmark,
            "runtime_dependencies": runtime_dependencies,
            "precision_plan": precision_plan.as_dict(),
        }
        with self._lock:
            self._loaded[model_id] = loaded
            self._model_states[model_id] = status
        self._log_model_state(model_id, "LOADING", "READY")
        return status

    def unload_model(self, model_id: str) -> dict[str, Any]:
        model_id = self._safe_segment(model_id)
        with self._lock:
            loaded = self._loaded.pop(model_id, None)
        if loaded is not None:
            del loaded.pipeline
            gc.collect()
            try:
                torch = self._imports().torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except WorkerError:
                pass

        status = self.model_status(model_id)
        status = {
            **status,
            "state": (
                ModelState.INSTALLED
                if status.get("installed")
                else ModelState.NOT_INSTALLED
            ),
            "runtime_compatible": bool(
                status.get("installed") and status.get("weights_valid")
            ),
            "validation_test": False,
            "loaded": False,
            "ready": False,
        }
        with self._lock:
            self._model_states[model_id] = status
        return status

    def unload_all(self) -> dict[str, Any]:
        with self._lock:
            model_ids = list(self._loaded)
        for model_id in model_ids:
            self.unload_model(model_id)
        return {"unloaded": model_ids}

    def _ensure_pipeline_for_capability(
        self,
        loaded: LoadedModel,
        requested_capability: str,
    ) -> Any:
        metadata = loaded.metadata or {}
        adapter = self._registry.select_for_capability(metadata, requested_capability)
        if adapter is None:
            raise self._unsupported_pipeline_error(metadata, requested_capability)

        if loaded.capability == requested_capability and loaded.pipeline is not None:
            return adapter

        snapshot, _ = self._active_snapshot(loaded.model_id)
        torch = self._imports().torch

        old_pipeline = loaded.pipeline
        loaded.pipeline = None
        if old_pipeline is not None:
            del old_pipeline
        gc.collect()
        if loaded.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        precision_plan = self._precision_plan(
            torch,
            metadata,
            loaded.device == "cuda",
        )
        dtype = self._dtype_resolver.materialize(torch, precision_plan)
        try:
            pipeline = self._load_pipeline(
                snapshot=snapshot,
                metadata=metadata,
                capability=requested_capability,
                adapter=adapter,
                device=loaded.device,
                dtype=dtype,
            )
        except Exception as error:
            with self._lock:
                self._loaded.pop(loaded.model_id, None)
                self._model_states[loaded.model_id] = {
                    "model_id": loaded.model_id,
                    "state": ModelState.FAILED,
                    "downloaded": True,
                    "validated": True,
                    "installed": True,
                    "weights_valid": True,
                    "runtime_available": True,
                    "runtime_compatible": False,
                    "validation_test": False,
                    "loaded": False,
                    "ready": False,
                    "error": (
                        "Changement de pipeline impossible: "
                        f"{type(error).__name__}: {error}"
                    ),
                }
            raise WorkerError(
                f"Impossible de charger le pipeline {requested_capability}: {error}",
                503,
                code="LOAD_FAILED",
            ) from error

        loaded.pipeline = pipeline
        loaded.capability = requested_capability
        precision_plan.components = self._dtype_resolver.inspect_components(pipeline)
        loaded.precision_plan = precision_plan
        loaded.precision = precision_plan.precision
        return adapter

    def _output_path(self, relative_path: str) -> Path:
        candidate = (self.settings.outputs_dir / relative_path).resolve()
        root = self.settings.outputs_dir.resolve()
        if candidate == root or root not in candidate.parents:
            raise WorkerError("Le chemin de sortie quitte le volume autorisé.", 422)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _recover_dtype_pipeline(
        self,
        loaded: LoadedModel,
        requested_capability: str,
        adapter: Any,
        error: BaseException,
    ) -> None:
        plan = loaded.precision_plan
        if plan is None or not self._dtype_resolver.is_dtype_mismatch(error):
            raise error
        replacement = self._dtype_resolver.recovery_plan(
            plan,
            cuda_available=loaded.device == "cuda",
        )
        if replacement is None:
            raise WorkerError(
                f"DTYPE_MISMATCH: {type(error).__name__}: {error}",
                422,
                code="DTYPE_MISMATCH",
                retryable=False,
            ) from error

        snapshot, _ = self._active_snapshot(loaded.model_id)
        torch = self._imports().torch
        previous = loaded.pipeline
        loaded.pipeline = None
        if previous is not None:
            del previous
        gc.collect()
        if loaded.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            loaded.pipeline = self._load_pipeline(
                snapshot=snapshot,
                metadata=loaded.metadata or {},
                capability=requested_capability,
                adapter=adapter,
                device=loaded.device,
                dtype=self._dtype_resolver.materialize(torch, replacement),
            )
        except Exception as recovery_error:
            raise WorkerError(
                f"DTYPE_MISMATCH_RECOVERY_FAILED: {type(recovery_error).__name__}: {recovery_error}",
                422,
                code="DTYPE_MISMATCH",
                retryable=False,
            ) from recovery_error
        replacement.components = self._dtype_resolver.inspect_components(loaded.pipeline)
        loaded.precision_plan = replacement
        loaded.precision = replacement.precision

    def _generate_with_adapter(
        self,
        loaded: LoadedModel,
        request: dict[str, Any],
        *,
        job_id: str,
    ) -> dict[str, Any]:
        requested_capability = str(
            request.get("capability") or loaded.capability or "TEXT_TO_IMAGE"
        ).upper()
        adapter = self._ensure_pipeline_for_capability(
            loaded,
            requested_capability,
        )
        prepared_request = self._resolve_generation_inputs(request, loaded.pipeline)
        is_video_request = requested_capability in VIDEO_CAPABILITIES
        if is_video_request:
            profile = ModelRuntimeProfile.from_metadata(loaded.metadata or {}, loaded.pipeline)
            try:
                resolution = self._resolution_resolver.resolve(
                    quality=prepared_request.get("quality") or "480p",
                    aspect_ratio=prepared_request.get("aspect_ratio") or "16:9",
                    pipeline=loaded.pipeline,
                    metadata=loaded.metadata or {},
                    requested_width=prepared_request.get("width"),
                    requested_height=prepared_request.get("height"),
                    default_width=profile.width,
                    default_height=profile.height,
                )
            except ValueError as error:
                raise WorkerError(
                    str(error),
                    422,
                    code="RESOLUTION_UNSUPPORTED",
                    retryable=False,
                ) from error
            prepared_request.update(
                {
                    "quality": resolution.requested_quality,
                    "aspect_ratio": resolution.requested_aspect_ratio,
                    "width": resolution.width,
                    "height": resolution.height,
                    "dimension_multiple": resolution.dimension_multiple,
                }
            )
            normalized = profile.normalize(prepared_request, video=True)
            prepared_request.update(
                {
                    "fps": normalized["fps"],
                    "duration_seconds": normalized["duration_seconds"],
                    "frames": normalized["num_frames"],
                    "steps": normalized["num_inference_steps"],
                    "guidance_scale": normalized["guidance_scale"],
                }
            )

        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
            if cancel_event is None:
                raise WorkerError("Job actif introuvable.", 404)

        def emit_progress(progress: int) -> None:
            with self._lock:
                self._jobs[job_id]["progress"] = progress

        callback = GenerationProgressReporter(
            total_steps=int(prepared_request.get("steps") or 4),
            cancelled=cancel_event.is_set,
            emit=emit_progress,
        )

        torch = self._imports().torch
        generation_started = time.perf_counter()
        if loaded.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        generator_device = loaded.device if loaded.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        if request.get("seed") is not None:
            generator.manual_seed(request["seed"])

        runtime = {
            "device": loaded.device,
            "generator": generator,
            "callback": callback,
            "metadata": loaded.metadata or {},
            "capability": requested_capability,
        }
        try:
            try:
                output = adapter.generate(loaded.pipeline, runtime, prepared_request)
            except Exception as retry_error:
                if self._dtype_resolver.is_dtype_mismatch(retry_error):
                    raise WorkerError(
                        f"DTYPE_MISMATCH: {type(retry_error).__name__}: {retry_error}",
                        422,
                        code="DTYPE_MISMATCH",
                        retryable=False,
                    ) from retry_error
                raise
        except Exception as error:
            self._recover_dtype_pipeline(
                loaded,
                requested_capability,
                adapter,
                error,
            )
            output = adapter.generate(loaded.pipeline, runtime, prepared_request)
        if cancel_event.is_set():
            raise InterruptedError("Job annulé.")

        images, normalized_frames = self._output_normalizer.extract(output, video=is_video_request)
        if not images and not normalized_frames:
            raise RuntimeError("Le runtime n'a produit aucune sortie.")

        output_path = self._output_path(request["output_relative_path"])
        media_probe: dict[str, Any]
        if is_video_request:
            if not normalized_frames:
                raise RuntimeError(
                    "Le pipeline vidéo n'a renvoyé aucune séquence de frames exploitable."
                )
            fps = int(output.get("fps") or prepared_request.get("fps") or 24)
            media_probe = self._output_normalizer.write_video(normalized_frames, output_path, fps)
        else:
            media_probe = self._output_normalizer.write_image(images, output_path)

        gpu_after = self._nvidia_metrics()
        process_peak = (
            int(torch.cuda.max_memory_reserved()) if loaded.device == "cuda" else 0
        )
        idle_vram = int(loaded.load_benchmark.get("vram_idle_bytes", 0))
        total_vram = int((gpu_after or {}).get("vram_total_bytes", 0))
        observed_peak = max(
            int((gpu_after or {}).get("vram_used_bytes", 0)),
            idle_vram + process_peak,
        )
        if total_vram:
            observed_peak = min(observed_peak, total_vram)

        width = int(media_probe.get("width") or output.get("width") or prepared_request.get("width") or 512)
        height = int(media_probe.get("height") or output.get("height") or prepared_request.get("height") or 512)

        return {
            "job_id": job_id,
            "state": JobState.COMPLETED,
            "progress": 100,
            "output_relative_path": request["output_relative_path"],
            "width": width,
            "height": height,
            "requested_quality": prepared_request.get("quality") if is_video_request else None,
            "requested_aspect_ratio": prepared_request.get("aspect_ratio") if is_video_request else None,
            "actual_width": width,
            "actual_height": height,
            "actual_fps": media_probe.get("fps") if is_video_request else None,
            "actual_frames": media_probe.get("frames") if is_video_request else None,
            "sha256": self._sha256(output_path),
            "benchmark": {
                **loaded.load_benchmark,
                "vram_after_load_bytes": int(
                    loaded.load_benchmark.get("vram_after_load_bytes", 0)
                ),
                "vram_peak_bytes": observed_peak,
                "ram_peak_bytes": self._ram_peak_bytes(),
                "precision": loaded.precision,
                "precision_plan": (
                    loaded.precision_plan.as_dict()
                    if loaded.precision_plan is not None
                    else None
                ),
                "resolution_width": width,
                "resolution_height": height,
                "frames": media_probe.get("frames"),
                "duration_seconds": media_probe.get("duration"),
                "fps": media_probe.get("fps"),
                "batch": 1,
                "inference_seconds": time.perf_counter() - generation_started,
            },
        }

    def generate_image(self, request: dict[str, Any]) -> dict[str, Any]:
        job_id = request["job_id"]
        model_id = request["model_id"]
        with self._lock:
            loaded = self._loaded.get(model_id)
            if loaded is None or loaded.pipeline is None:
                raise WorkerError(
                    "Le modèle n'est pas READY.",
                    409,
                    code="MODEL_NOT_READY",
                    retryable=False,
                )
            cancel_event = threading.Event()
            self._cancel_events[job_id] = cancel_event
            self._jobs[job_id] = {
                "job_id": job_id,
                "state": JobState.RUNNING,
                "progress": 0,
                "error": None,
            }

        try:
            result = self._generate_with_adapter(loaded, request, job_id=job_id)
        except InterruptedError:
            result = {
                "job_id": job_id,
                "state": JobState.CANCELLED,
                "progress": self._jobs[job_id]["progress"],
                "error": None,
            }
        except Exception as error:
            result = {
                "job_id": job_id,
                "state": JobState.FAILED,
                "progress": self._jobs[job_id]["progress"],
                "error": f"{type(error).__name__}: {error}",
                "error_code": getattr(error, "code", "GENERATION_FAILED"),
            }
        finally:
            with self._lock:
                self._jobs[job_id] = result
                self._cancel_events.pop(job_id, None)
        return result

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            event = self._cancel_events.get(job_id)
            if event is None:
                raise WorkerError("Job actif introuvable.", 404)
            event.set()
            return {"job_id": job_id, "cancellation_requested": True}

    def job_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise WorkerError("Job worker introuvable.", 404)
            return dict(job)
