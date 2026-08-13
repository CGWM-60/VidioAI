"""Versioned ModelPack registry with atomic, last-known-good reloads."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..workflows.validator import WorkflowValidationError, WorkflowValidator
from .schema import ModelPack, ModelPackError


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    packs: dict[str, ModelPack]
    packs_dir: Path
    workflows_dir: Path
    workflows: dict[str, dict[str, Any]]
    source: str


class ModelPackRegistry:
    """Load a bundled registry plus an optional backend-owned active overlay.

    The active layout is shared with the Rust backend::

        registry.json
        model-packs/<id>/<version>/manifest.json
        active/<id>.json
        workflows/<workflow_version>/<template>.json

    A candidate replaces the in-memory snapshot only after the index, every
    active pointer, every versioned manifest and every referenced workflow have
    been parsed and checksum-verified. A partial or invalid update therefore
    leaves the last valid snapshot executable.
    """

    def __init__(
        self,
        directory: Path | str,
        *,
        workflows_directory: Path | str | None = None,
        active_directory: Path | str | None = None,
        current_vidioai_version: str | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.workflows_directory = Path(
            workflows_directory or self.directory.parent / "workflows"
        )
        self.active_directory = Path(active_directory) if active_directory else None
        self.current_vidioai_version = (
            current_vidioai_version
            or os.getenv("VIDIOAI_VERSION", "0.1.0")
        ).strip()
        self._lock = threading.RLock()
        self._last_probe: tuple[tuple[str, int, int], ...] | None = None
        self._generation = 0
        self.last_reload_error: str | None = None
        self._snapshot = self._load_bundled_snapshot()
        self.reload_if_changed(force=True)

    @staticmethod
    def _read_object(path: Path, *, label: str) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelPackError(f"{label} invalide: {error}") from error
        if not isinstance(payload, dict):
            raise ModelPackError(f"{label} doit être un objet JSON.")
        return payload

    @staticmethod
    def _sha256_bytes(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _sha256_file(cls, path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise ModelPackError(f"Fichier de registre illisible: {path}") from error
        return digest.hexdigest()

    @staticmethod
    def _canonical_pack_bytes(raw: dict[str, Any]) -> bytes:
        # Matches serde_json::to_vec(ModelPack): compact UTF-8 JSON while
        # preserving struct-field/BTreeMap order emitted by the Rust writer.
        return json.dumps(
            raw,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _fingerprint(root: Path | None) -> tuple[tuple[str, int, int], ...]:
        if root is None or not root.is_dir():
            return ()
        values: list[tuple[str, int, int]] = []
        try:
            paths = sorted(root.rglob("*.json"))
        except OSError:
            return ()
        for path in paths:
            try:
                stat = path.stat()
                relative = path.relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
            values.append((relative, stat.st_mtime_ns, stat.st_size))
        return tuple(values)

    @staticmethod
    def _segment(value: Any, *, label: str) -> str:
        result = str(value or "")
        if not _SAFE_SEGMENT_RE.fullmatch(result) or result.startswith("."):
            raise ModelPackError(f"{label} invalide: {result!r}")
        return result

    @staticmethod
    def _digest(value: Any, *, label: str) -> str:
        result = str(value or "").lower()
        if not _SHA256_RE.fullmatch(result):
            raise ModelPackError(f"SHA-256 invalide pour {label}.")
        return result

    @staticmethod
    def _version_parts(value: str) -> tuple[int, int, int] | None:
        match = re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
        if match is None:
            return None
        return tuple(int(part or 0) for part in match.groups())  # type: ignore[return-value]

    def _check_minimum_version(self, required: Any, *, pack_id: str) -> None:
        required_value = str(required or "").strip()
        required_parts = self._version_parts(required_value)
        current_parts = self._version_parts(self.current_vidioai_version)
        if not required_value or required_parts is None:
            raise ModelPackError(
                f"min_vidioai_version invalide pour {pack_id}."
            )
        # Development identifiers are intentionally accepted: release images
        # expose a numeric VIDIOAI_VERSION and receive the strict comparison.
        if current_parts is not None and required_parts > current_parts:
            raise ModelPackError(
                f"ModelPack {pack_id} requiert VidioAI {required_value}."
            )

    @staticmethod
    def _path_below(root: Path, relative: str, *, label: str) -> Path:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise ModelPackError(f"Chemin {label} interdit: {relative}")
        root_resolved = root.resolve()
        path = (root / relative_path).resolve()
        if root_resolved not in path.parents:
            raise ModelPackError(f"Chemin {label} hors registre: {relative}")
        return path

    def _load_bundled_snapshot(self) -> RegistrySnapshot:
        if not self.directory.is_dir() or not self.workflows_directory.is_dir():
            raise ModelPackError("Arborescence bundled packs/workflows absente.")
        workflows: dict[str, dict[str, Any]] = {}
        for path in sorted(self.workflows_directory.glob("*.json")):
            raw = self._read_object(path, label=f"Workflow {path.name}")
            try:
                WorkflowValidator.validate_template(raw)
            except WorkflowValidationError as error:
                raise ModelPackError(
                    f"Workflow bundled invalide {path.name}: {error}"
                ) from error
            workflows[path.name] = raw
        packs: dict[str, ModelPack] = {}
        for path in sorted(self.directory.glob("*.json")):
            raw = self._read_object(path, label=f"ModelPack {path.name}")
            try:
                pack = ModelPack.from_dict(raw)
            except ModelPackError as error:
                raise ModelPackError(
                    f"ModelPack bundled invalide {path.name}: {error}"
                ) from error
            if pack.id in packs:
                raise ModelPackError(f"Identifiant ModelPack dupliqué: {pack.id}")
            self._validate_pack_workflow_names(pack, workflows)
            packs[pack.id] = pack
        if not packs:
            raise ModelPackError("Registre ModelPack bundled vide.")
        return RegistrySnapshot(
            packs=packs,
            packs_dir=self.directory,
            workflows_dir=self.workflows_directory,
            workflows=workflows,
            source="bundled",
        )

    @staticmethod
    def _validate_pack_workflow_names(
        pack: ModelPack,
        workflows: dict[str, dict[str, Any]],
    ) -> None:
        for template in pack.workflow_by_capability.values():
            if Path(template).name != template or template not in workflows:
                raise ModelPackError(
                    f"Workflow {template} absent/invalide pour {pack.id}."
                )

    @staticmethod
    def _index_entries(index: dict[str, Any]) -> list[dict[str, Any]]:
        if int(index.get("schema_version") or 0) != 1:
            raise ModelPackError("Version de registry.json non prise en charge.")
        raw_entries = index.get("packs")
        if not isinstance(raw_entries, list):
            raise ModelPackError("registry.json.packs doit être une liste.")
        entries: list[dict[str, Any]] = []
        for raw in raw_entries:
            if not isinstance(raw, dict):
                raise ModelPackError("Entrée registry.json invalide.")
            entries.append(raw)
        return entries

    def _validate_manifest(
        self,
        raw: dict[str, Any],
        *,
        expected_id: str,
        expected_version: str,
        label: str,
    ) -> tuple[ModelPack, str, str, list[dict[str, Any]]]:
        if int(raw.get("schema_version") or 0) != 1:
            raise ModelPackError(f"{label}: schema_version invalide.")
        pack_id = self._segment(raw.get("id"), label=f"{label}.id")
        version = self._segment(raw.get("version"), label=f"{label}.version")
        if pack_id != expected_id or version != expected_version:
            raise ModelPackError(f"{label}: identité/version incohérente.")
        pack_raw = raw.get("pack")
        if not isinstance(pack_raw, dict):
            raise ModelPackError(f"{label}.pack doit être un objet.")
        try:
            pack = ModelPack.from_dict(pack_raw)
        except ModelPackError as error:
            raise ModelPackError(f"{label}.pack invalide: {error}") from error
        if pack.id != pack_id:
            raise ModelPackError(f"{label}: identifiant pack incohérent.")
        expected_digest = self._digest(raw.get("sha256"), label=label)
        actual_digest = self._sha256_bytes(self._canonical_pack_bytes(pack_raw))
        if actual_digest != expected_digest:
            raise ModelPackError(
                f"MODEL_PACK_CHECKSUM_MISMATCH pour {pack_id}/{version}."
            )
        workflow_version = self._segment(
            raw.get("workflow_version"),
            label=f"{label}.workflow_version",
        )
        self._check_minimum_version(raw.get("min_vidioai_version"), pack_id=pack_id)
        workflows = raw.get("workflows")
        if not isinstance(workflows, list):
            raise ModelPackError(f"{label}.workflows doit être une liste.")
        normalized_workflows: list[dict[str, Any]] = []
        for item in workflows:
            if not isinstance(item, dict):
                raise ModelPackError(f"{label}.workflows contient une entrée invalide.")
            normalized_workflows.append(item)
        return pack, expected_digest, workflow_version, normalized_workflows

    def _validate_workflows(
        self,
        root: Path,
        *,
        pack: ModelPack,
        workflow_version: str,
        records: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        by_capability: dict[str, dict[str, Any]] = {}
        for record in records:
            capability = str(record.get("capability") or "").strip().upper()
            if capability in by_capability or capability not in pack.capabilities:
                raise ModelPackError(
                    f"Workflow capability invalide/dupliquée pour {pack.id}: {capability}"
                )
            template = str(record.get("template") or "")
            expected_prefix = f"workflows/{workflow_version}/"
            if not template.startswith(expected_prefix):
                raise ModelPackError(
                    f"Chemin workflow incohérent pour {pack.id}: {template}"
                )
            filename = Path(template).name
            if not _SAFE_SEGMENT_RE.fullmatch(filename) or not filename.endswith(".json"):
                raise ModelPackError(f"Nom de workflow invalide: {template}")
            declared = pack.workflow_for(capability)
            if declared != filename:
                raise ModelPackError(
                    f"Workflow {template} ne correspond pas au pack {pack.id}/{capability}."
                )
            path = self._path_below(root, template, label="workflow")
            if not path.is_file():
                raise ModelPackError(f"Workflow actif absent: {template}")
            expected_digest = self._digest(
                record.get("sha256"), label=f"workflow {template}"
            )
            if self._sha256_file(path) != expected_digest:
                raise ModelPackError(
                    f"WORKFLOW_CHECKSUM_MISMATCH pour {template}."
                )
            raw = self._read_object(path, label=f"Workflow {template}")
            try:
                WorkflowValidator.validate_template(raw)
            except WorkflowValidationError as error:
                raise ModelPackError(
                    f"Workflow actif invalide {template}: {error}"
                ) from error
            by_capability[capability] = raw
        expected_capabilities = set(pack.workflow_by_capability)
        if set(by_capability) != expected_capabilities:
            missing = sorted(expected_capabilities.symmetric_difference(by_capability))
            raise ModelPackError(
                f"Inventaire workflows incomplet pour {pack.id}: {', '.join(missing)}"
            )
        return {
            pack.workflow_for(capability) or "": workflow
            for capability, workflow in by_capability.items()
        }

    def _load_active_snapshot(self, root: Path) -> RegistrySnapshot | None:
        index_path = root / "registry.json"
        if not root.is_dir() or not index_path.is_file():
            return None
        index = self._read_object(index_path, label="registry.json")
        entries = self._index_entries(index)
        active_entries: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if entry.get("active") is not True:
                continue
            pack_id = self._segment(entry.get("id"), label="registry.pack.id")
            if pack_id in active_entries:
                raise ModelPackError(f"Plusieurs versions actives pour {pack_id}.")
            active_entries[pack_id] = entry
        if not active_entries:
            raise ModelPackError("Le registre actif ne contient aucun ModelPack actif.")

        pointer_dir = root / "active"
        pointer_paths = sorted(pointer_dir.glob("*.json")) if pointer_dir.is_dir() else []
        pointer_ids = {path.stem for path in pointer_paths}
        if pointer_ids != set(active_entries):
            changed = sorted(pointer_ids.symmetric_difference(active_entries))
            raise ModelPackError(
                "Pointeurs actifs incohérents: " + ", ".join(changed)
            )

        packs: dict[str, ModelPack] = {}
        workflows: dict[str, dict[str, Any]] = {}
        workflow_sources: dict[str, str] = {}
        workflow_versions: set[str] = set()
        for pack_id in sorted(active_entries):
            entry = active_entries[pack_id]
            version = self._segment(
                entry.get("version"), label=f"registry.{pack_id}.version"
            )
            pointer_path = pointer_dir / f"{pack_id}.json"
            artifact_path = root / "model-packs" / pack_id / version / "manifest.json"
            pointer = self._read_object(pointer_path, label=f"active/{pack_id}.json")
            artifact = self._read_object(
                artifact_path,
                label=f"model-packs/{pack_id}/{version}/manifest.json",
            )
            if pointer != artifact:
                raise ModelPackError(
                    f"Pointeur actif et artefact divergent pour {pack_id}/{version}."
                )
            pack, digest, workflow_version, manifest_workflows = self._validate_manifest(
                pointer,
                expected_id=pack_id,
                expected_version=version,
                label=f"active/{pack_id}.json",
            )
            entry_digest = self._digest(entry.get("sha256"), label=f"registry.{pack_id}")
            if entry_digest != digest:
                raise ModelPackError(f"SHA registry/manifest divergent pour {pack_id}.")
            if str(entry.get("workflow_version") or "") != workflow_version:
                raise ModelPackError(
                    f"workflow_version registry/manifest divergent pour {pack_id}."
                )
            if str(entry.get("min_vidioai_version") or "") != str(
                pointer.get("min_vidioai_version") or ""
            ):
                raise ModelPackError(
                    f"min_vidioai_version registry/manifest divergent pour {pack_id}."
                )
            entry_workflows = entry.get("workflows")
            if entry_workflows is not None and entry_workflows != manifest_workflows:
                raise ModelPackError(
                    f"Workflows registry/manifest divergents pour {pack_id}."
                )
            loaded_workflows = self._validate_workflows(
                root,
                pack=pack,
                workflow_version=workflow_version,
                records=manifest_workflows,
            )
            for filename, workflow in loaded_workflows.items():
                source = f"{workflow_version}/{filename}"
                if filename in workflows and workflow_sources[filename] != source:
                    if workflows[filename] != workflow:
                        raise ModelPackError(
                            f"Nom de workflow actif ambigu: {filename}."
                        )
                workflows[filename] = workflow
                workflow_sources[filename] = source
            packs[pack.id] = pack
            workflow_versions.add(workflow_version)

        # Workflows are served from the immutable in-memory snapshot. This path
        # remains diagnostic only when multiple workflow versions are active.
        workflows_dir = (
            root / "workflows" / next(iter(workflow_versions))
            if len(workflow_versions) == 1
            else root / "workflows"
        )
        return RegistrySnapshot(
            packs=packs,
            packs_dir=root / "model-packs",
            workflows_dir=workflows_dir,
            workflows=workflows,
            source="active",
        )

    def reload_if_changed(self, *, force: bool = False) -> bool:
        probe = self._fingerprint(self.active_directory)
        with self._lock:
            if not force and probe == self._last_probe:
                return False
            self._last_probe = probe
            try:
                candidate = (
                    self._load_active_snapshot(self.active_directory)
                    if self.active_directory is not None
                    else None
                )
            except ModelPackError as error:
                self.last_reload_error = str(error)
                return False
            if candidate is None:
                # An absent optional mount is valid at startup. Once an active
                # snapshot was loaded, an empty/temporarily unavailable mount
                # cannot silently revoke that last-known-good authority.
                if self._snapshot.source != "active":
                    self.last_reload_error = None
                return False
            self._snapshot = candidate
            self._generation += 1
            self.last_reload_error = None
            return True

    def reload(self) -> bool:
        return self.reload_if_changed(force=True)

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def workflows_dir(self) -> Path:
        with self._lock:
            return self._snapshot.workflows_dir

    def workflow_templates(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._snapshot.workflows)

    @property
    def source(self) -> str:
        with self._lock:
            return self._snapshot.source

    def all(self) -> tuple[ModelPack, ...]:
        self.reload_if_changed()
        with self._lock:
            return tuple(self._snapshot.packs.values())

    def get(self, pack_id: str) -> ModelPack | None:
        self.reload_if_changed()
        with self._lock:
            return self._snapshot.packs.get(pack_id)

    def __iter__(self) -> Iterable[ModelPack]:
        return iter(self.all())
