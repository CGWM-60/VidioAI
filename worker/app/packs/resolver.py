"""Resolve families by architecture and pipeline contracts, never repo id."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .registry import ModelPackRegistry
from .schema import ModelPack, ModelPackStatus


@dataclass(frozen=True, slots=True)
class ModelPackResolution:
    pack: ModelPack | None
    status: ModelPackStatus
    score: int
    matched_by: tuple[str, ...]
    reason: str

    def as_fields(self, capability: str | None = None) -> dict[str, Any]:
        pack = self.pack
        workflow = pack.workflow_for(capability or "") if pack and capability else None
        return {
            "model_pack_id": pack.id if pack else None,
            "model_pack_status": self.status.value,
            "engine": pack.engine if pack else None,
            "workflow": workflow,
            "advanced_parameters": pack.advanced_parameters if pack else [],
            "presets": pack.presets if pack else {},
            "model_pack_reason": self.reason,
        }


class ModelPackResolver:
    def __init__(self, registry: ModelPackRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _tokens(metadata: dict[str, Any], key: str) -> set[str]:
        values: list[Any] = []
        value = metadata.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, (list, tuple, set)):
            values.extend(value)
        config = metadata.get("config") or {}
        nested = config.get(key) if isinstance(config, dict) else None
        if isinstance(nested, str):
            values.append(nested)
        elif isinstance(nested, (list, tuple, set)):
            values.extend(nested)
        return {str(item).strip().lower() for item in values if str(item).strip()}

    def resolve(self, metadata: dict[str, Any], capability: str | None = None) -> ModelPackResolution:
        architectures = self._tokens(metadata, "architectures")
        for container_name in ("model_index", "modular_model_index"):
            container = metadata.get(container_name) or {}
            if isinstance(container, dict):
                for value in (container.get("_class_name"), container.get("_blocks_class_name")):
                    if value:
                        architectures.add(str(value).strip().lower())
                for component in container.values():
                    if isinstance(component, (list, tuple)):
                        architectures.update(
                            str(item).strip().lower()
                            for item in component
                            if isinstance(item, str) and item.strip()
                        )
        pipeline_classes = self._tokens(metadata, "pipeline_classes")
        for value in (metadata.get("pipeline_class"), metadata.get("class_name")):
            if value:
                pipeline_classes.add(str(value).strip().lower())
        family_tokens = self._tokens(metadata, "family") | self._tokens(metadata, "model_type")
        requested = capability.upper() if capability else None
        best: tuple[int, ModelPack, list[str]] | None = None
        for pack in self.registry:
            if requested and requested not in pack.capabilities:
                continue
            matches: list[str] = []
            score = 0
            architecture_matches = architectures & {value.lower() for value in pack.architectures}
            class_matches = pipeline_classes & {value.lower() for value in pack.pipeline_classes}
            family_match = pack.family.lower() in family_tokens
            if architecture_matches:
                score += 100 + 5 * len(architecture_matches)
                matches.append("architecture")
            if class_matches:
                score += 80 + 5 * len(class_matches)
                matches.append("pipeline_class")
            if family_match:
                score += 40
                matches.append("family")
            if (
                "*" in pack.pipeline_classes
                and pipeline_classes
                and str(metadata.get("library_name") or "diffusers").lower() in {"", "diffusers"}
            ):
                score += 5
                matches.append("generic_diffusers")
            if score and (best is None or score > best[0]):
                best = (score, pack, matches)
        if best is None:
            return ModelPackResolution(
                pack=None,
                status=ModelPackStatus.UNSUPPORTED,
                score=0,
                matched_by=(),
                reason="Aucun ModelPack ne correspond aux architectures ou classes déclarées.",
            )
        score, pack, matches = best
        return ModelPackResolution(
            pack=pack,
            status=pack.status,
            score=score,
            matched_by=tuple(matches),
            reason=f"ModelPack {pack.id} résolu via {', '.join(matches)}.",
        )
