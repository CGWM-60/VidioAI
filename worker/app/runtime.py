"""Runtime Diffusers réel et gestion sûre des snapshots de modèles."""

from __future__ import annotations

import gc
import hashlib
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
from .config import Settings
from .schemas import JobState, ModelState


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
        self._runtime_modules: tuple[Any, Any] | None = None
        self._runtime_error: str | None = None
        self._registry = PipelineRegistry()

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
    def _log_model_state(model_id: str, from_state: str, to_state: str, reason: str | None = None) -> None:
        suffix = f" reason={reason}" if reason else ""
        print(f"MODEL_STATE {model_id} {from_state} -> {to_state}{suffix}")

    @staticmethod
    def _load_image(path: str | Path) -> Any:
        from PIL import Image

        image = Image.open(Path(path))
        try:
            return image.convert("RGB")
        finally:
            image.close()

    @staticmethod
    def _load_first_video_frame(path: str | Path, workspace: Path) -> Any:
        frame_path = workspace / f"frame-{uuid.uuid4()}.png"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(frame_path),
        ]
        try:
            subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
        except subprocess.SubprocessError as error:
            raise WorkerError(f"Décodage vidéo impossible: {error}", 422) from error
        try:
            return RuntimeManager._load_image(frame_path)
        finally:
            frame_path.unlink(missing_ok=True)

    def _resolve_generation_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(request)
        capability = str(prepared.get("capability") or "").upper()
        input_path = prepared.get("input_path")
        if isinstance(input_path, str) and input_path.strip():
            candidate = Path(input_path)
            if not candidate.is_file():
                raise WorkerError("Le fichier d'entrée est introuvable.", 422)
            if capability in {"VIDEO_TO_VIDEO", "VIDEO_INPAINTING", "VIDEO_UPSCALE"}:
                prepared["input_video"] = str(candidate)
                prepared["input_frames"] = [
                    self._load_first_video_frame(candidate, self.settings.work_dir)
                ]
            else:
                prepared["input_image"] = self._load_image(candidate)

        mask_path = prepared.get("mask_path")
        if isinstance(mask_path, str) and mask_path.strip():
            candidate = Path(mask_path)
            if not candidate.is_file():
                raise WorkerError("Le masque fourni est introuvable.", 422)
            prepared["mask_image"] = self._load_image(candidate)

        control_path = prepared.get("control_path")
        if isinstance(control_path, str) and control_path.strip():
            candidate = Path(control_path)
            if not candidate.is_file():
                raise WorkerError("L'image de contrôle est introuvable.", 422)
            prepared["control_image"] = self._load_image(candidate)

        resolved_images = []
        for item in prepared.get("input_images") or []:
            if not isinstance(item, dict):
                continue
            source = item.get("source") or item.get("path") or item.get("input_path")
            if not isinstance(source, str) or not source.strip():
                continue
            candidate = Path(source)
            if not candidate.is_file():
                raise WorkerError("Une image d'entrée référencée est introuvable.", 422)
            resolved_images.append(self._load_image(candidate))
        if resolved_images:
            prepared["resolved_input_images"] = resolved_images

        return prepared

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

    def _resolve_supported_capability(
        self,
        metadata: dict[str, Any],
        fallback_capabilities: list[str] | None = None,
    ) -> str | None:
        capabilities = metadata.get("capabilities") or fallback_capabilities or []
        normalized = {
            str(capability).upper()
            for capability in capabilities
            if isinstance(capability, str) and capability.strip()
        }
        for candidate in self._capability_order():
            if normalized and candidate not in normalized:
                continue
            adapter = self._registry.select_for_capability(metadata, candidate)
            if adapter is not None:
                return candidate
        return None

    def _imports(self) -> tuple[Any, Any]:
        """Import paresseux : /health reste disponible même si CUDA est cassé."""
        if self._runtime_modules is not None:
            return self._runtime_modules
        if self._runtime_error is not None:
            raise WorkerError(self._runtime_error, 503)
        try:
            import torch
            from huggingface_hub import HfApi, snapshot_download

            self._runtime_modules = (torch, (HfApi, snapshot_download))
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
    def _looks_like_hf_auth_error(error: Exception) -> bool:
        name = type(error).__name__
        if name in {"GatedRepoError", "HfHubHTTPError"}:
            return True
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {401, 403}:
            return True
        message = str(error).lower()
        return any(fragment in message for fragment in ["gated", "private", "authentication", "forbidden", "access"])

    @staticmethod
    def _looks_like_hf_not_found(error: Exception) -> bool:
        message = str(error).lower()
        return "404" in message and (
            "repository" in message or "repo" in message or "not found" in message
        )

    @staticmethod
    def _looks_like_hf_revision_not_found(error: Exception) -> bool:
        message = str(error).lower()
        return "revision" in message and "not found" in message

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
            or "xet" in message and "reconstruct" in message
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
        cache_dir = Path(os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE") or (self.settings.hf_home / "hub"))
        xet_dir = Path(os.getenv("HF_XET_CACHE") or (self.settings.hf_home / "xet"))
        tmp_dir = Path(os.getenv("TMPDIR") or self.settings.work_dir)

        for path, code, message in [
            (models_dir, "SCRATCH_NOT_WRITABLE", "Le dossier Scratch des modèles n'est pas inscriptible."),
            (cache_dir, "CACHE_NOT_WRITABLE", "Le cache Hugging Face n'est pas inscriptible."),
            (xet_dir, "CACHE_NOT_WRITABLE", "Le cache HF-XET n'est pas inscriptible."),
            (tmp_dir, "SCRATCH_NOT_WRITABLE", "Le dossier temporaire du worker n'est pas inscriptible."),
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
        # marge pour fichiers temporaires/cache/reconstruction
        return int(total * 2.2)

    @staticmethod
    def _clear_partial_directory(path: Path) -> None:
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

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

    def runtime_status(self) -> dict[str, Any]:
        configuration_errors = self.settings.configuration_errors()
        try:
            torch, _ = self._imports()
            runtime_available = True
            cuda_available = bool(torch.cuda.is_available())
            cuda_version = getattr(torch.version, "cuda", None)
            torch_version = getattr(torch, "__version__", None)
        except WorkerError as error:
            runtime_available = False
            cuda_available = False
            cuda_version = None
            torch_version = None
            if str(error) not in configuration_errors:
                configuration_errors.append(str(error))

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
            "errors": configuration_errors,
        }

    @staticmethod
    def _nvidia_metrics() -> dict[str, Any] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.total,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=5, check=True
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
        """Retourne le pic RSS du processus worker Linux en octets."""
        # Sur Linux, `ru_maxrss` est exprimé en KiB. Le worker de production est
        # conteneurisé sous Linux, ce qui rend cette conversion déterministe.
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
            "worker_status": "ready"
            if self.runtime_status()["ready"]
            else "not_ready",
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
        if "_class_name" not in parsed_index:
            raise WorkerError("Le manifest Diffusers ne déclare pas _class_name.", 422)

        weights = sorted(snapshot.rglob("*.safetensors"))
        total_weights = sum(path.stat().st_size for path in weights)
        if not weights or total_weights < self.settings.minimum_weights_bytes:
            raise WorkerError(
                "Le snapshot ne contient pas de poids safetensors cohérents.", 422
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

    def install_model(
        self,
        model_id: str,
        repository: str,
        revision: str,
        capabilities: list[str],
    ) -> dict[str, Any]:
        model_id = self._safe_segment(model_id)
        self._log_model_state(model_id, "DISCOVERED", "DOWNLOADING")
        # Un snapshot restauré depuis le cache L3 S3 est réutilisé uniquement
        # après la même validation complète que pour un téléchargement HF.
        try:
            snapshot, pointer = self._active_snapshot(model_id)
            # Ne jamais court-circuiter une mise à jour : `active.json` peut
            # pointer vers l'ancienne révision alors que le backend a demandé le
            # nouveau commit immuable publié par Hugging Face.
            if pointer.get("revision") != revision:
                raise WorkerError("Une révision plus récente doit être installée.", 409)
            validation = self.validate_snapshot(snapshot)
            cached = {
                "model_id": model_id,
                "repository": pointer["repository"],
                "revision": pointer["revision"],
                "installed": True,
                "weights_valid": validation["weights_valid"],
                "runtime_available": self.runtime_status()["runtime_available"],
                "runtime_compatible": False,
                "validation_test": False,
                "state": ModelState.INSTALLED,
                **validation,
            }
            with self._lock:
                self._model_states[model_id] = cached
            return cached
        except WorkerError:
            pass
        with self._lock:
            self._model_states[model_id] = {
                "model_id": model_id,
                "state": ModelState.DOWNLOADING,
                "downloaded": False,
                "validated": False,
                "loaded": False,
                "ready": False,
                "error": None,
            }

        # Le staging vit sur le même filesystem que la destination. `os.replace`
        # reste ainsi atomique même lorsque `/work` et `/models` sont deux mounts.
        temporary = (
            self.settings.models_dir
            / ".downloads"
            / f"download-{model_id}-{uuid.uuid4()}"
        )
        temporary.mkdir(parents=True)
        try:
            _, hub = self._imports()
            HfApi, snapshot_download = hub
            hf_token = self._hf_token()
            info = HfApi(token=hf_token).model_info(
                repository, revision=revision
            )
            resolved_revision = self._safe_segment(info.sha)

            required_bytes = self._estimate_required_download_bytes(info)
            self._precheck_download_environment(required_bytes)

            xet_retries = max(1, int(os.getenv("VIDIOAI_HF_XET_RETRIES", "2")))
            allow_no_xet_fallback = os.getenv("VIDIOAI_ENABLE_HF_XET_FALLBACK", "true").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

            last_xet_error: Exception | None = None
            for attempt in range(1, xet_retries + 1):
                try:
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
                if allow_no_xet_fallback:
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
                            f"HF_XET_RECONSTRUCTION_ERROR: reconstruction HF-XET échouée, fallback sans XET impossible ({type(fallback_error).__name__}).",
                            502,
                            code="HF_XET_RECONSTRUCTION_ERROR",
                            retryable=True,
                        ) from fallback_error
                else:
                    raise WorkerError(
                        "HF_XET_RECONSTRUCTION_ERROR: reconstruction HF-XET échouée après retries.",
                        502,
                        code="HF_XET_RECONSTRUCTION_ERROR",
                        retryable=True,
                    ) from last_xet_error

            validation = self.validate_snapshot(temporary)
            self._log_model_state(model_id, "DOWNLOADING", "VALIDATING")

            metadata = inspect_model_metadata(temporary)
            resolved_capability = self._resolve_supported_capability(metadata, capabilities)
            if resolved_capability is None:
                raise WorkerError(
                    "pipeline non supporté: aucun adapter Diffusers compatible n'a été trouvé pour ce modèle.",
                    422,
                    code="PIPELINE_UNSUPPORTED",
                    retryable=False,
                )

            destination = self._model_root(model_id) / resolved_revision
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)

            runtime_capabilities = metadata.get("capabilities") or capabilities or [resolved_capability]
            manifest = {
                "model_id": model_id,
                "repository": repository,
                "revision": resolved_revision,
                "capabilities": runtime_capabilities,
                "downloaded": True,
                "validated": True,
                "installed": True,
                "weights_valid": True,
                "runtime_available": self.runtime_status()["runtime_available"],
                "runtime_compatible": True,
                "validation_test": False,
                "loaded": False,
                "ready": False,
                "state": ModelState.INSTALLED,
                "installed_at": int(time.time()),
                **validation,
            }
            (temporary / "vidioai-model.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            os.replace(temporary, destination)
            pointer = {
                "model_id": model_id,
                "repository": repository,
                "revision": resolved_revision,
            }
            pointer_path = self._active_pointer(model_id)
            pointer_temporary = pointer_path.with_suffix(".json.tmp")
            pointer_temporary.write_text(json.dumps(pointer), encoding="utf-8")
            os.replace(pointer_temporary, pointer_path)
            with self._lock:
                self._model_states[model_id] = manifest
            self._log_model_state(model_id, "VALIDATING", "INSTALLED")
            return manifest
        except WorkerError as error:
            self._log_model_state(model_id, "VALIDATING", "FAILED", reason=error.code)
            with self._lock:
                self._model_states[model_id] = {
                    "model_id": model_id,
                    "state": ModelState.FAILED,
                    "downloaded": temporary.exists(),
                    "validated": False,
                    "installed": False,
                    "loaded": False,
                    "ready": False,
                    "error_code": error.code,
                    "retryable": error.retryable,
                    "error": str(error),
                }
            raise
        except Exception as error:
            if self._hf_token() is None and self._looks_like_hf_auth_error(error):
                message = (
                    "Accès Hugging Face requis: ce repository est protégé "
                    "(gated/private) et nécessite un HF_TOKEN valide."
                )
                with self._lock:
                    self._model_states[model_id] = {
                        "model_id": model_id,
                        "state": ModelState.FAILED,
                        "error_code": "HF_ACCESS_DENIED",
                        "retryable": False,
                        "error": message,
                    }
                raise WorkerError(message, 403, code="HF_ACCESS_DENIED") from error

            if self._looks_like_hf_not_found(error):
                message = "Repository Hugging Face introuvable."
                code = "HF_MODEL_NOT_FOUND"
                status = 404
                retryable = False
            elif self._looks_like_hf_revision_not_found(error):
                message = "Révision Hugging Face introuvable pour ce repository."
                code = "HF_REVISION_NOT_FOUND"
                status = 404
                retryable = False
            elif self._looks_like_timeout(error):
                message = "Délai dépassé pendant le téléchargement Hugging Face."
                code = "HF_DOWNLOAD_TIMEOUT"
                status = 504
                retryable = True
            elif self._looks_like_xet_reconstruction_error(error):
                message = "Erreur de reconstruction HF-XET pendant le téléchargement."
                code = "HF_XET_RECONSTRUCTION_ERROR"
                status = 502
                retryable = True
            else:
                message = f"Installation impossible: {type(error).__name__}: {error}"
                code = "HF_DOWNLOAD_ERROR"
                status = 502
                retryable = False

            with self._lock:
                self._model_states[model_id] = {
                    "model_id": model_id,
                    "state": ModelState.FAILED,
                    "downloaded": temporary.exists(),
                    "validated": False,
                    "installed": False,
                    "loaded": False,
                    "ready": False,
                    "error_code": code,
                    "retryable": retryable,
                    "error": message,
                }
            raise WorkerError(message, status, code=code, retryable=retryable) from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def model_status(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            current = self._model_states.get(model_id)
            # Les états transitoires/erreurs vivent en mémoire. Pour un état
            # installé, le pointeur disque reste toutefois la source d'autorité
            # afin qu'une suppression effectuée par le backend soit visible.
            if current is not None and current.get("state") in {
                ModelState.DOWNLOADING,
                ModelState.FAILED,
                ModelState.RUNTIME_UNAVAILABLE,
                ModelState.INCOMPATIBLE,
            }:
                return current
            if current is not None and self._active_pointer(model_id).is_file():
                return current
        try:
            snapshot, _ = self._active_snapshot(model_id)
            manifest_path = snapshot / "vidioai-model.json"
            if not manifest_path.is_file():
                raise WorkerError("Le manifest VidioAI du snapshot est absent.", 409)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = inspect_model_metadata(snapshot)
            resolved_capability = self._resolve_supported_capability(
                metadata, manifest.get("capabilities")
            )
            if resolved_capability is None:
                return {
                    "model_id": model_id,
                    "state": ModelState.FAILED,
                    "downloaded": True,
                    "validated": False,
                    "installed": False,
                    "weights_valid": manifest.get("weights_valid", False),
                    "runtime_available": self.runtime_status()["runtime_available"],
                    "runtime_compatible": False,
                    "validation_test": False,
                    "loaded": False,
                    "ready": False,
                    "error_code": "PIPELINE_UNSUPPORTED",
                    "error": "pipeline non supporté: aucun adapter Diffusers compatible n'a été trouvé pour ce modèle.",
                }
            manifest.setdefault("downloaded", True)
            manifest.setdefault("validated", True)
            manifest.setdefault("loaded", manifest.get("state") == ModelState.READY)
            manifest.setdefault("ready", manifest.get("state") == ModelState.READY)
            return manifest
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

    def load_model(self, model_id: str) -> dict[str, Any]:
        model_id = self._safe_segment(model_id)
        self._log_model_state(model_id, "INSTALLED", "LOADING")
        snapshot, pointer = self._active_snapshot(model_id)
        validation = self.validate_snapshot(snapshot)
        torch, _ = self._imports()
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
                "error": "CUDA est obligatoire mais indisponible.",
            }
            with self._lock:
                self._model_states[model_id] = status
            raise WorkerError(status["error"], 503)

        metadata = inspect_model_metadata(snapshot)
        capability = None
        for candidate in self._capability_order():
            adapter = self._registry.select_for_capability(metadata, candidate)
            if adapter is not None:
                capability = candidate
                break
        if capability is None:
            self._log_model_state(model_id, "LOADING", "FAILED", reason="PIPELINE_UNSUPPORTED")
            raise WorkerError(
                "pipeline non supporté: aucun adapter Diffusers compatible n'a été trouvé pour ce modèle.",
                422,
                code="PIPELINE_UNSUPPORTED",
            )
        adapter = self._registry.select_for_capability(metadata, capability)
        device = "cuda" if cuda_available else "cpu"
        dtype = torch.float16 if cuda_available else torch.float32
        precision = "FP16" if cuda_available else "FP32"
        gpu_before = self._nvidia_metrics()
        load_started = time.perf_counter()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
        try:
            pipeline = adapter.load(
                str(snapshot),
                {"torch_dtype": dtype, "device": device},
                {"device": device, "class_name": metadata.get("class_name")},
            )
            if device == "cuda" and hasattr(pipeline, "to"):
                pipeline = pipeline.to(device)
            validation_output = adapter.generate(
                pipeline,
                {"device": device, "generator": torch.Generator(device=device) if hasattr(torch, "Generator") else None},
                {
                    "prompt": "VidioAI runtime validation",
                    "negative_prompt": None,
                    "width": 64,
                    "height": 64,
                    "steps": 1,
                    "guidance_scale": 0.0,
                },
            )
            if not validation_output.get("images") and not validation_output.get("frames"):
                raise RuntimeError("Le pipeline n'a produit aucune sortie de validation.")
        except Exception as error:
            self._log_model_state(model_id, "LOADING", "FAILED", reason="LOAD_FAILED")
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
            }
            with self._lock:
                self._model_states[model_id] = status
            raise WorkerError(status["error"], 503) from error

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
            "resolution_width": 64,
            "resolution_height": 64,
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
            validation_test=True,
            precision=precision,
            load_benchmark=load_benchmark,
            pipeline=pipeline,
            capability=capability,
            metadata=metadata,
        )
        status = {
            "model_id": model_id,
            "state": ModelState.READY,
            "downloaded": True,
            "validated": True,
            "installed": True,
            "weights_valid": validation["weights_valid"],
            "runtime_available": True,
            "runtime_compatible": True,
            "validation_test": True,
            "loaded": True,
            "ready": True,
            "device": device,
            "capability": capability,
            "repository": pointer["repository"],
            "revision": pointer["revision"],
            "benchmark": load_benchmark,
        }
        with self._lock:
            self._loaded[model_id] = loaded
            self._model_states[model_id] = status
        self._log_model_state(model_id, "LOADING", "READY")
        return status

    def unload_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded.pop(model_id, None)
        if loaded is not None:
            del loaded.pipeline
            gc.collect()
            try:
                torch, _ = self._imports()
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
            "runtime_compatible": False,
            "validation_test": False,
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

    def _output_path(self, relative_path: str) -> Path:
        candidate = (self.settings.outputs_dir / relative_path).resolve()
        root = self.settings.outputs_dir.resolve()
        if candidate == root or root not in candidate.parents:
            raise WorkerError("Le chemin de sortie quitte le volume autorisé.", 422)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def _generate_with_adapter(self, loaded: LoadedModel, request: dict[str, Any], *, job_id: str) -> dict[str, Any]:
        requested_capability = request.get("capability") or loaded.capability or "TEXT_TO_IMAGE"
        adapter = self._registry.select_for_capability(loaded.metadata or {}, requested_capability)
        if adapter is None:
            raise WorkerError("Aucun adapter compatible ne peut générer cette capacité.", 422)
        prepared_request = self._resolve_generation_inputs(request)

        with self._lock:
            cancel_event = self._cancel_events.get(job_id)
            if cancel_event is None:
                raise WorkerError("Job actif introuvable.", 404)

        def callback(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: Any):
            if cancel_event.is_set():
                raise InterruptedError("Job annulé.")
            with self._lock:
                self._jobs[job_id]["progress"] = min(95, int(((step + 1) / max(1, request.get("steps", 4))) * 95))
            return callback_kwargs

        torch, _ = self._imports()
        generation_started = time.perf_counter()
        if loaded.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        generator_device = loaded.device if loaded.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        if request.get("seed") is not None:
            generator.manual_seed(request["seed"])
        runtime = {"device": loaded.device, "generator": generator, "callback": callback}
        output = adapter.generate(loaded.pipeline, runtime, prepared_request)
        if cancel_event.is_set():
            raise InterruptedError("Job annulé.")

        images = output.get("images") or []
        frames = output.get("frames") or []
        if not images and not frames:
            raise RuntimeError("Le runtime n'a produit aucune sortie.")

        output_path = self._output_path(request["output_relative_path"])
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        if images:
            images[0].save(temporary, format="PNG")
        else:
            from PIL import Image

            frame = frames[0]
            if isinstance(frame, list):
                frame = frame[0]
            if hasattr(frame, "save"):
                frame.save(temporary, format="PNG")
            else:
                Image.fromarray(frame).save(temporary, format="PNG")
        os.replace(temporary, output_path)
        gpu_after = self._nvidia_metrics()
        process_peak = int(torch.cuda.max_memory_reserved()) if loaded.device == "cuda" else 0
        idle_vram = int(loaded.load_benchmark.get("vram_idle_bytes", 0))
        total_vram = int((gpu_after or {}).get("vram_total_bytes", 0))
        observed_peak = max(int((gpu_after or {}).get("vram_used_bytes", 0)), idle_vram + process_peak)
        if total_vram:
            observed_peak = min(observed_peak, total_vram)
        return {
            "job_id": job_id,
            "state": JobState.COMPLETED,
            "progress": 100,
            "output_relative_path": request["output_relative_path"],
            "width": 512,
            "height": 512,
            "sha256": self._sha256(output_path),
            "benchmark": {
                **loaded.load_benchmark,
                "vram_after_load_bytes": int(loaded.load_benchmark.get("vram_after_load_bytes", 0)),
                "vram_peak_bytes": observed_peak,
                "ram_peak_bytes": self._ram_peak_bytes(),
                "precision": loaded.precision,
                "resolution_width": 512,
                "resolution_height": 512,
                "batch": 1,
                "inference_seconds": time.perf_counter() - generation_started,
            },
        }

    def generate_image(self, request: dict[str, Any]) -> dict[str, Any]:
        job_id = request["job_id"]
        model_id = request["model_id"]
        with self._lock:
            loaded = self._loaded.get(model_id)
            if loaded is None or not loaded.validation_test:
                raise WorkerError("Le modèle n'est pas READY.", 409)
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
