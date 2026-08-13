"""Atomic checks performed immediately before model execution."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..hardware.execution_plan import ExecutionPlan
from ..packs.schema import ModelPack, ModelPackStatus
from ..workflows.builder import BuiltWorkflow, WorkflowBuilder
from ..workflows.validator import WorkflowValidationError


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    ok: bool
    code: str | None = None
    message: str = ""


@dataclass(frozen=True, slots=True)
class PreflightError:
    code: str
    message: str
    retryable: bool = False


@dataclass(slots=True)
class PreflightResult:
    status: str
    ready: bool
    model_id: str
    model_pack_id: str | None
    engine: str | None
    workflow: str | None
    execution_plan: ExecutionPlan | None
    checks: list[PreflightCheck] = field(default_factory=list)
    errors: list[PreflightError] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    built_workflow: BuiltWorkflow | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "model_id": self.model_id,
            "model_pack_id": self.model_pack_id,
            "engine": self.engine,
            "workflow": self.workflow,
            "execution_plan": self.execution_plan.as_dict() if self.execution_plan else None,
            "checks": [asdict(value) for value in self.checks],
            "errors": [asdict(value) for value in self.errors],
            "diagnostics": self.diagnostics,
        }


class PreflightService:
    def __init__(self, workflow_builder: WorkflowBuilder) -> None:
        self.workflow_builder = workflow_builder

    def run(
        self,
        *,
        model_id: str,
        pack: ModelPack | None,
        capability: str,
        request: dict[str, Any],
        snapshot: Path,
        execution_plan: ExecutionPlan | None,
        engine_health: Callable[[], dict[str, Any]],
        dependency_errors: list[str] | None = None,
        diagnostics: dict[str, Any] | None = None,
        model_loaded: bool = False,
        component_values: dict[str, Any] | None = None,
        component_error: str | None = None,
    ) -> PreflightResult:
        checks: list[PreflightCheck] = []
        errors: list[PreflightError] = []

        def check(name: str, ok: bool, code: str, message: str, *, retryable: bool = False) -> None:
            checks.append(PreflightCheck(name=name, ok=ok, code=None if ok else code, message=message))
            if not ok:
                errors.append(PreflightError(code=code, message=message, retryable=retryable))

        check("model_pack", pack is not None, "MODEL_PACK_MISSING", "Aucun ModelPack exécutable n'a été résolu.")
        if pack is not None:
            check(
                "model_pack_status",
                pack.status in {ModelPackStatus.READY, ModelPackStatus.EXPERIMENTAL},
                "MODEL_INCOMPATIBLE",
                f"Le ModelPack est {pack.status.value}.",
            )
            check("capability", capability in pack.capabilities, "MODEL_INCOMPATIBLE", f"Capability {capability} absente du ModelPack.")
        required_components: list[str] = []
        if pack is not None:
            for key in ("checkpoint", "vae"):
                value = pack.components.get(key)
                if isinstance(value, str) and value:
                    required_components.append(value)
            required_components.extend(str(value) for value in pack.components.get("text_encoders") or [] if value)
        missing = [value for value in required_components if not (snapshot / value).exists()]
        files_ok = not component_error and (
            model_loaded or (snapshot.is_dir() and not missing)
        )
        check(
            "model_files",
            files_ok,
            "MODEL_FILE_MISSING",
            (
                component_error
                or (
                    "Modèle déjà résident en mémoire."
                    if model_loaded
                    else f"Composants absents: {', '.join(missing)}"
                    if missing
                    else "Snapshot et composants présents."
                )
            ),
        )
        check("dependencies", not dependency_errors, "DEPENDENCY_MISSING", "; ".join(dependency_errors or []) or "Dépendances disponibles.")
        output_relative = request.get("output_relative_path")
        output_ok = output_relative is None or isinstance(output_relative, str)
        check("storage", output_ok, "OUTPUT_FAILED", "Chemin de sortie valide." if output_ok else "Chemin de sortie invalide.")
        needs_ffmpeg = "VIDEO" in capability
        media_ok = not needs_ffmpeg or (shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None)
        check("ffmpeg", media_ok, "DEPENDENCY_MISSING", "FFmpeg/ffprobe disponibles." if media_ok else "FFmpeg/ffprobe absents.")
        health = engine_health()
        engine_ok = bool(health.get("ready"))
        check(
            "engine",
            engine_ok,
            str(health.get("error_code") or "ENGINE_UNAVAILABLE"),
            str(health.get("error") or "Moteur disponible."),
            retryable=True,
        )
        plan_ok = execution_plan is not None and execution_plan.feasible
        check("execution_plan", plan_ok, "INSUFFICIENT_VRAM", execution_plan.reason if execution_plan else "ExecutionPlan absent.", retryable=True)

        built: BuiltWorkflow | None = None
        if pack is not None and not component_error:
            try:
                built = self.workflow_builder.build(
                    pack,
                    capability,
                    str(request.get("quality") or "BALANCED"),
                    request,
                    execution_plan=execution_plan,
                    component_values=component_values,
                )
                checks.append(PreflightCheck(name="workflow", ok=True, message="Workflow validé."))
                available_node_types = health.get("available_node_types")
                if pack.engine == "comfyui":
                    if not isinstance(available_node_types, list):
                        check(
                            "comfyui_nodes",
                            False,
                            "NODE_MISSING",
                            "Inventaire /object_info ComfyUI indisponible.",
                            retryable=True,
                        )
                    else:
                        required_node_types = {
                            str(node.get("class_type"))
                            for node in built.workflow.values()
                            if isinstance(node, dict) and node.get("class_type")
                        }
                        missing_node_types = sorted(
                            required_node_types - {str(value) for value in available_node_types}
                        )
                        check(
                            "comfyui_nodes",
                            not missing_node_types,
                            "NODE_MISSING",
                            (
                                "Tous les nodes ComfyUI requis sont installés."
                                if not missing_node_types
                                else "Nodes ComfyUI absents: " + ", ".join(missing_node_types)
                            ),
                        )
            except WorkflowValidationError as error:
                checks.append(PreflightCheck(name="workflow", ok=False, code=error.code, message=str(error)))
                errors.append(PreflightError(code=error.code, message=str(error)))
        return PreflightResult(
            status="READY_TO_RUN" if not errors else "BLOCKED",
            ready=not errors,
            model_id=model_id,
            model_pack_id=pack.id if pack else None,
            engine=pack.engine if pack else None,
            workflow=pack.workflow_for(capability) if pack else None,
            execution_plan=execution_plan,
            checks=checks,
            errors=errors,
            diagnostics={**(diagnostics or {}), "engine_health": health},
            built_workflow=built,
        )
