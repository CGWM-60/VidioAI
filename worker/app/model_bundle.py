"""Gestion générique d'un bundle VidioAI: modèle de base + LoRA(s) + recette.

Le module ne contient aucun branchement par repository/modèle. Les LoRA sont
résolus en révisions Hugging Face immuables, téléchargés localement pendant
l'installation, puis chargés exclusivement depuis le snapshot local.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any


BUNDLE_SCHEMA_VERSION = 1


class BundleError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "MODEL_BUNDLE_INVALID",
        status_code: int = 422,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


def _safe_adapter_name(value: str, index: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if not cleaned:
        cleaned = f"adapter-{index + 1}"
    return cleaned[:64]


def _validate_repository(repository: str) -> str:
    value = str(repository or "").strip()
    parts = value.split("/")
    if (
        len(parts) != 2
        or any(not part for part in parts)
        or any(part in {".", ".."} for part in parts)
    ):
        raise BundleError(
            f"Repository LoRA invalide: {value!r}",
            code="LORA_REPOSITORY_INVALID",
        )
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_."
    )
    if any(any(character not in allowed for character in part) for part in parts):
        raise BundleError(
            f"Repository LoRA invalide: {value!r}",
            code="LORA_REPOSITORY_INVALID",
        )
    return value


def _hub_bundle_error(
    error: Exception,
    repository: str,
    *,
    action: str,
) -> BundleError:
    names = {
        error_type.__name__
        for error_type in type(error).__mro__
    }
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)

    if "GatedRepoError" in names or status in {401, 403}:
        return BundleError(
            f"Accès Hugging Face requis pour le LoRA {repository}.",
            code="LORA_ACCESS_DENIED",
            status_code=403,
            retryable=False,
        )
    if "RevisionNotFoundError" in names:
        return BundleError(
            f"Révision LoRA introuvable pour {repository}.",
            code="LORA_REVISION_NOT_FOUND",
            status_code=404,
            retryable=False,
        )
    if "RepositoryNotFoundError" in names:
        return BundleError(
            f"Repository LoRA introuvable: {repository}.",
            code="LORA_REPOSITORY_NOT_FOUND",
            status_code=404,
            retryable=False,
        )
    if names.intersection({"RemoteEntryNotFoundError", "EntryNotFoundError"}):
        return BundleError(
            f"Fichier LoRA introuvable dans {repository}.",
            code="LORA_WEIGHT_NOT_FOUND",
            status_code=404,
            retryable=False,
        )
    return BundleError(
        f"{action} LoRA impossible pour {repository}: "
        f"{type(error).__name__}: {error}",
        code="LORA_DOWNLOAD_FAILED",
        status_code=502,
        retryable=True,
    )


def _normalized_scale(value: Any) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise BundleError(
            "Le poids d'un LoRA doit être numérique.",
            code="LORA_SCALE_INVALID",
        ) from error
    if not 0.0 <= scale <= 2.0:
        raise BundleError(
            "Le poids d'un LoRA doit être compris entre 0 et 2.",
            code="LORA_SCALE_INVALID",
        )
    return scale


class ModelBundleManager:
    """Installe et applique les composants d'un bundle de façon atomique."""

    @staticmethod
    def default_bundle(
        repository: str,
        revision: str,
        *,
        recipe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "base_model": {
                "repository": repository,
                "revision": revision,
            },
            "loras": [],
            "recipe": dict(recipe or {}),
        }

    @staticmethod
    def bundle_from_manifest(
        manifest: dict[str, Any] | None,
        *,
        repository: str,
        revision: str,
    ) -> dict[str, Any]:
        manifest = manifest or {}
        raw = manifest.get("bundle")
        if not isinstance(raw, dict):
            return ModelBundleManager.default_bundle(
                repository,
                revision,
            )

        base = raw.get("base_model")
        if not isinstance(base, dict):
            base = {}
        loras = raw.get("loras")
        if not isinstance(loras, list):
            loras = []
        recipe = raw.get("recipe")
        if not isinstance(recipe, dict):
            recipe = {}

        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "base_model": {
                "repository": str(
                    base.get("repository") or repository
                ),
                "revision": str(
                    base.get("revision") or revision
                ),
            },
            "loras": [
                dict(item)
                for item in loras
                if isinstance(item, dict)
            ],
            "recipe": dict(recipe),
        }

    @staticmethod
    def _candidate_weight_names(info: Any) -> list[str]:
        names: list[str] = []
        for sibling in getattr(info, "siblings", []) or []:
            filename = (
                getattr(sibling, "rfilename", None)
                or getattr(sibling, "path", None)
            )
            if (
                isinstance(filename, str)
                and filename.lower().endswith(".safetensors")
            ):
                names.append(filename)
        return sorted(dict.fromkeys(names))

    @staticmethod
    def select_weight_name(
        candidates: list[str],
        explicit: str | None = None,
    ) -> str:
        normalized = sorted(dict.fromkeys(candidates))
        if explicit:
            explicit = explicit.strip()
            if (
                not explicit.lower().endswith(".safetensors")
                or explicit not in normalized
            ):
                raise BundleError(
                    f"Poids LoRA introuvable: {explicit}",
                    code="LORA_WEIGHT_NOT_FOUND",
                )
            return explicit

        if not normalized:
            raise BundleError(
                "Aucun poids .safetensors n'a été trouvé dans le repository LoRA.",
                code="LORA_WEIGHT_NOT_FOUND",
            )
        if len(normalized) == 1:
            return normalized[0]

        def score(name: str) -> int:
            lower = name.lower()
            result = 0
            if "pytorch_lora_weights" in lower:
                result += 100
            if "adapter_model" in lower:
                result += 90
            if "lora" in lower:
                result += 50
            if "/" not in name:
                result += 5
            return result

        ranked = sorted(
            normalized,
            key=lambda item: (-score(item), item),
        )
        best_score = score(ranked[0])
        equally_ranked = [
            item for item in ranked if score(item) == best_score
        ]
        if len(equally_ranked) != 1:
            raise BundleError(
                "Plusieurs poids LoRA sont possibles. "
                "Renseignez explicitement weight_name.",
                code="LORA_WEIGHT_AMBIGUOUS",
            )
        return ranked[0]

    def materialize(
        self,
        *,
        snapshot: Path,
        repository: str,
        revision: str,
        loras: list[dict[str, Any]] | None,
        recipe: dict[str, Any] | None,
        token: str | None,
        cache_dir: Path,
        preserve_existing: bool,
    ) -> dict[str, Any]:
        """Télécharge uniquement les poids LoRA demandés puis publie atomiquement.

        `loras is None` signifie "conserver la composition existante".
        `loras == []` signifie "retirer tous les LoRA".
        `recipe is None` conserve la recette existante pendant une reconfiguration.
        """
        # Le chemin base-only ne doit pas dépendre du Hub LoRA.
        # L'import huggingface_hub est donc effectué uniquement lorsqu'au moins
        # un LoRA doit réellement être inspecté/téléchargé.
        snapshot = Path(snapshot)
        manifest_path = snapshot / "vidioai-model.json"
        try:
            existing_manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}

        current = self.bundle_from_manifest(
            existing_manifest,
            repository=repository,
            revision=revision,
        )
        desired_recipe = (
            dict(current.get("recipe") or {})
            if recipe is None and preserve_existing
            else dict(recipe or {})
        )

        if loras is None and preserve_existing:
            desired_loras = [
                dict(item)
                for item in current.get("loras") or []
                if isinstance(item, dict)
            ]
            return {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "base_model": {
                    "repository": repository,
                    "revision": revision,
                },
                "loras": desired_loras,
                "recipe": desired_recipe,
            }

        requested_loras = list(loras or [])

        HfApi = None
        hf_hub_download = None
        if requested_loras:
            try:
                from huggingface_hub import HfApi as _HfApi
                from huggingface_hub import hf_hub_download as _hf_hub_download
            except (ImportError, ModuleNotFoundError) as error:
                raise BundleError(
                    "Le support Hugging Face est requis uniquement pour télécharger un LoRA.",
                    code="LORA_RUNTIME_DEPENDENCY_MISSING",
                    status_code=503,
                    retryable=False,
                ) from error
            HfApi = _HfApi
            hf_hub_download = _hf_hub_download

        stage_root = (
            snapshot.parent
            / f".vidioai-bundle-{snapshot.name}-{uuid.uuid4()}"
        )
        stage_loras = stage_root / "loras"
        stage_loras.mkdir(parents=True, exist_ok=False)
        api = HfApi(token=token) if HfApi is not None else None
        materialized: list[dict[str, Any]] = []

        try:
            used_names: set[str] = set()
            for index, raw in enumerate(requested_loras):
                if not isinstance(raw, dict):
                    raise BundleError(
                        "Chaque LoRA doit être un objet.",
                        code="LORA_SPEC_INVALID",
                    )

                lora_repository = _validate_repository(
                    str(raw.get("repository") or "")
                )
                requested_revision = str(
                    raw.get("revision") or "main"
                ).strip()
                if not requested_revision:
                    requested_revision = "main"

                if api is None or hf_hub_download is None:
                    raise BundleError(
                        "Runtime Hugging Face LoRA indisponible.",
                        code="LORA_RUNTIME_DEPENDENCY_MISSING",
                        status_code=503,
                        retryable=False,
                    )

                try:
                    info = api.model_info(
                        lora_repository,
                        revision=requested_revision,
                        files_metadata=True,
                    )
                except Exception as error:
                    raise _hub_bundle_error(
                        error,
                        lora_repository,
                        action="Inspection",
                    ) from error
                resolved_revision = str(
                    getattr(info, "sha", None)
                    or requested_revision
                )
                candidates = self._candidate_weight_names(info)
                weight_name = self.select_weight_name(
                    candidates,
                    (
                        str(raw.get("weight_name")).strip()
                        if raw.get("weight_name")
                        else None
                    ),
                )

                requested_name = str(
                    raw.get("adapter_name")
                    or lora_repository.replace("/", "-")
                )
                adapter_name = _safe_adapter_name(
                    requested_name,
                    index,
                )
                if adapter_name in used_names:
                    adapter_name = _safe_adapter_name(
                        f"{adapter_name}-{index + 1}",
                        index,
                    )
                if adapter_name in used_names:
                    raise BundleError(
                        "Deux LoRA utilisent le même adapter_name.",
                        code="LORA_ADAPTER_NAME_DUPLICATE",
                    )
                used_names.add(adapter_name)

                adapter_dir = stage_loras / adapter_name
                adapter_dir.mkdir(parents=True, exist_ok=False)

                try:
                    local_file = Path(
                        hf_hub_download(
                            repo_id=lora_repository,
                            filename=weight_name,
                            revision=resolved_revision,
                            local_dir=adapter_dir,
                            cache_dir=cache_dir,
                            token=token,
                        )
                    )
                except Exception as error:
                    raise _hub_bundle_error(
                        error,
                        lora_repository,
                        action="Téléchargement",
                    ) from error

                if (
                    not local_file.is_file()
                    or local_file.suffix.lower() != ".safetensors"
                ):
                    raise BundleError(
                        f"Le poids LoRA téléchargé est invalide: {weight_name}",
                        code="LORA_WEIGHT_INVALID",
                    )

                relative_weight = str(
                    local_file.relative_to(adapter_dir)
                )
                file_sha = hashlib.sha256(
                    local_file.read_bytes()
                ).hexdigest()

                source = {
                    "repository": lora_repository,
                    "revision": resolved_revision,
                    "requested_revision": requested_revision,
                    "adapter_name": adapter_name,
                    "weight_name": relative_weight,
                    "scale": _normalized_scale(
                        raw.get("scale", 1.0)
                    ),
                    "enabled": bool(
                        raw.get("enabled", True)
                    ),
                    "local_path": (
                        f"vidioai/loras/{adapter_name}"
                    ),
                    "sha256": file_sha,
                }
                (adapter_dir / "vidioai-lora.json").write_text(
                    json.dumps(source, indent=2),
                    encoding="utf-8",
                )
                materialized.append(source)

            vidioai_root = snapshot / "vidioai"
            vidioai_root.mkdir(parents=True, exist_ok=True)
            target_loras = vidioai_root / "loras"
            previous = (
                vidioai_root
                / f"loras.previous-{uuid.uuid4()}"
            )

            had_previous = target_loras.exists()
            if had_previous:
                os.replace(target_loras, previous)
            try:
                os.replace(stage_loras, target_loras)
            except Exception:
                if had_previous and previous.exists():
                    os.replace(previous, target_loras)
                raise
            if previous.exists():
                shutil.rmtree(previous, ignore_errors=True)

            bundle = {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "base_model": {
                    "repository": repository,
                    "revision": revision,
                },
                "loras": materialized,
                "recipe": desired_recipe,
            }
            return bundle
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

    @staticmethod
    def apply_loras(
        pipeline: Any,
        snapshot: Path,
        bundle: dict[str, Any] | None,
    ) -> Any:
        raw_loras = (
            bundle.get("loras")
            if isinstance(bundle, dict)
            else None
        )
        enabled = [
            item
            for item in (raw_loras or [])
            if isinstance(item, dict)
            and item.get("enabled", True)
        ]
        if not enabled:
            return pipeline

        fingerprint = hashlib.sha256(
            json.dumps(
                enabled,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if getattr(
            pipeline,
            "_vidioai_lora_fingerprint",
            None,
        ) == fingerprint:
            return pipeline

        loader = getattr(pipeline, "load_lora_weights", None)
        if not callable(loader):
            raise BundleError(
                "Cette pipeline Diffusers ne supporte pas load_lora_weights.",
                code="LORA_UNSUPPORTED",
            )

        names: list[str] = []
        scales: list[float] = []
        for item in enabled:
            adapter_name = str(
                item.get("adapter_name") or ""
            )
            local_path = snapshot / str(
                item.get("local_path") or ""
            )
            weight_name = str(
                item.get("weight_name") or ""
            )
            if (
                not adapter_name
                or not local_path.is_dir()
                or not weight_name
                or not (local_path / weight_name).is_file()
            ):
                raise BundleError(
                    f"LoRA local incomplet: {adapter_name or 'sans nom'}",
                    code="LORA_LOCAL_SNAPSHOT_INVALID",
                )
            try:
                loader(
                    str(local_path),
                    weight_name=weight_name,
                    adapter_name=adapter_name,
                    local_files_only=True,
                )
            except TypeError:
                # Certaines versions Diffusers ne déclarent pas
                # local_files_only sur ce mixin. Le chemin reste local.
                try:
                    loader(
                        str(local_path),
                        weight_name=weight_name,
                        adapter_name=adapter_name,
                    )
                except Exception as error:
                    raise BundleError(
                        f"Chargement LoRA impossible: {adapter_name}: "
                        f"{type(error).__name__}: {error}",
                        code="LORA_LOAD_FAILED",
                    ) from error
            except Exception as error:
                raise BundleError(
                    f"Chargement LoRA impossible: {adapter_name}: "
                    f"{type(error).__name__}: {error}",
                    code="LORA_LOAD_FAILED",
                ) from error
            names.append(adapter_name)
            scales.append(_normalized_scale(item.get("scale", 1.0)))

        setter = getattr(pipeline, "set_adapters", None)
        if callable(setter):
            try:
                setter(
                    names,
                    adapter_weights=scales,
                )
            except TypeError:
                setter(names, scales)
            except Exception as error:
                raise BundleError(
                    f"Activation des LoRA impossible: "
                    f"{type(error).__name__}: {error}",
                    code="LORA_LOAD_FAILED",
                ) from error
        elif len(names) != 1 or abs(scales[0] - 1.0) > 1e-9:
            raise BundleError(
                "La pipeline charge un LoRA mais ne permet pas de régler "
                "plusieurs adapters/poids.",
                code="LORA_SCALE_UNSUPPORTED",
            )

        setattr(
            pipeline,
            "_vidioai_lora_fingerprint",
            fingerprint,
        )
        return pipeline
