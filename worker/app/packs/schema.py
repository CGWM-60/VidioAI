"""Validated, versioned ModelPack contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ModelPackError(ValueError):
    def __init__(self, message: str, *, code: str = "MODEL_PACK_INVALID") -> None:
        super().__init__(message)
        self.code = code


class ModelPackStatus(StrEnum):
    READY = "READY"
    EXPERIMENTAL = "EXPERIMENTAL"
    DOWNLOADABLE = "DOWNLOADABLE"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class ModelPack:
    schema_version: int
    id: str
    family: str
    status: ModelPackStatus
    engine: str
    capabilities: tuple[str, ...]
    architectures: tuple[str, ...]
    pipeline_classes: tuple[str, ...]
    workflow_by_capability: dict[str, str]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    components: dict[str, Any]
    defaults: dict[str, Any]
    memory_policy: dict[str, Any]
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelPack":
        if not isinstance(raw, dict):
            raise ModelPackError("Un ModelPack doit être un objet JSON.")
        required = {
            "schema_version",
            "id",
            "family",
            "status",
            "engine",
            "capabilities",
            "architectures",
            "pipeline_classes",
            "workflow_by_capability",
            "inputs",
            "outputs",
            "components",
            "defaults",
            "memory_policy",
            "presets",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ModelPackError(f"Champs ModelPack absents: {', '.join(missing)}")
        try:
            status = ModelPackStatus(str(raw["status"]).upper())
        except ValueError as error:
            raise ModelPackError("Statut ModelPack invalide.") from error
        pack_id = str(raw["id"]).strip()
        family = str(raw["family"]).strip()
        engine = str(raw["engine"]).strip().lower()
        if not pack_id or not family or engine not in {"diffusers", "comfyui"}:
            raise ModelPackError("id, family ou engine ModelPack invalide.")

        def strings(name: str) -> tuple[str, ...]:
            value = raw[name]
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ModelPackError(f"{name} doit être une liste de chaînes.")
            return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))

        capabilities = tuple(value.upper() for value in strings("capabilities"))
        workflows = raw["workflow_by_capability"]
        if not isinstance(workflows, dict):
            raise ModelPackError("workflow_by_capability doit être un objet.")
        workflow_by_capability = {
            str(key).upper(): str(value)
            for key, value in workflows.items()
            if str(key).strip() and str(value).strip()
        }
        if any(capability not in workflow_by_capability for capability in capabilities):
            raise ModelPackError("Chaque capability doit déclarer un workflow.")
        mappings: dict[str, dict[str, Any]] = {}
        for name in ("inputs", "outputs", "components", "defaults", "memory_policy", "presets"):
            value = raw[name]
            if not isinstance(value, dict):
                raise ModelPackError(f"{name} doit être un objet.")
            mappings[name] = dict(value)
        return cls(
            schema_version=int(raw["schema_version"]),
            id=pack_id,
            family=family,
            status=status,
            engine=engine,
            capabilities=capabilities,
            architectures=strings("architectures"),
            pipeline_classes=strings("pipeline_classes"),
            workflow_by_capability=workflow_by_capability,
            inputs=mappings["inputs"],
            outputs=mappings["outputs"],
            components=mappings["components"],
            defaults=mappings["defaults"],
            memory_policy=mappings["memory_policy"],
            presets={str(key).upper(): dict(value) for key, value in mappings["presets"].items() if isinstance(value, dict)},
        )

    @property
    def advanced_parameters(self) -> list[str]:
        declared = self.inputs.get("advanced_parameters") or []
        return [str(value) for value in declared if isinstance(value, str)]

    def workflow_for(self, capability: str) -> str | None:
        return self.workflow_by_capability.get(capability.upper())

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["capabilities"] = list(self.capabilities)
        value["architectures"] = list(self.architectures)
        value["pipeline_classes"] = list(self.pipeline_classes)
        return value
