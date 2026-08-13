"""Build concrete ComfyUI API workflows only from explicit ModelPack bindings."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hardware.execution_plan import ExecutionPlan
from ..packs.schema import ModelPack
from .validator import WorkflowValidationError, WorkflowValidator


@dataclass(frozen=True, slots=True)
class BuiltWorkflow:
    template: str
    workflow: dict[str, Any]
    output_nodes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"template": self.template, "workflow": self.workflow, "output_nodes": list(self.output_nodes)}


class WorkflowBuilder:
    EXECUTION_BINDINGS = (
        "dtype",
        "quantization",
        "attention",
        "vae_tiling",
        "vae_slicing",
        "model_cpu_offload",
        "sequential_cpu_offload",
        "component_placement",
    )
    def __init__(
        self,
        directory: Path | str,
        *,
        templates: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.directory = Path(directory)
        # Active-registry templates are copied into the validated registry
        # snapshot. Keeping another private copy here prevents a concurrent
        # backend rename/write from changing a workflow between preflight and
        # queue submission.
        self._templates = copy.deepcopy(templates) if templates is not None else None

    def load(self, template_name: str) -> dict[str, Any]:
        safe = Path(template_name).name
        if safe != template_name or not safe.endswith(".json"):
            raise WorkflowValidationError("Nom de workflow invalide.")
        if self._templates is not None:
            template = self._templates.get(safe)
            if template is None:
                raise WorkflowValidationError(
                    f"Workflow absent: {safe}", code="WORKFLOW_MISSING"
                )
            result = copy.deepcopy(template)
            WorkflowValidator.validate_template(result)
            return result
        path = self.directory / safe
        if not path.is_file():
            raise WorkflowValidationError(f"Workflow absent: {safe}", code="WORKFLOW_MISSING")
        try:
            template = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkflowValidationError(f"Workflow JSON invalide: {safe}") from error
        WorkflowValidator.validate_template(template)
        return template

    @staticmethod
    def _request_values(pack: ModelPack, preset: str, request: dict[str, Any]) -> dict[str, Any]:
        values = dict(pack.defaults)
        values.update(pack.presets.get(preset.upper(), {}))
        aliases = {
            "steps": "steps",
            "guidance_scale": "cfg",
            "cfg": "cfg",
            "width": "width",
            "height": "height",
            "frames": "frames",
            "fps": "fps",
            "seed": "seed",
            "prompt": "prompt",
            "negative_prompt": "negative_prompt",
            "input_path": "input_image",
            "batch": "batch",
            "dtype": "dtype",
            "quantization": "quantization",
            "attention": "attention",
            "vae_tiling": "vae_tiling",
            "vae_slicing": "vae_slicing",
            "model_cpu_offload": "model_cpu_offload",
            "sequential_cpu_offload": "sequential_cpu_offload",
            "component_placement": "component_placement",
        }
        resolution = values.get("resolution")
        if isinstance(resolution, dict):
            values.setdefault("width", resolution.get("width"))
            values.setdefault("height", resolution.get("height"))
        values.setdefault("batch", 1)
        values.setdefault(
            "component_placement",
            dict(pack.memory_policy.get("component_placement") or {}),
        )
        for source, target in aliases.items():
            if request.get(source) is not None:
                values[target] = request[source]
        if values.get("input_image") is None:
            raw_images = request.get("input_images")
            if isinstance(raw_images, list):
                ordered = sorted(
                    (item for item in raw_images if isinstance(item, dict)),
                    key=lambda item: int(item.get("order") or 0),
                )
                for item in ordered:
                    source = item.get("source") or item.get("path") or item.get("input_path")
                    if source:
                        values["input_image"] = source
                        break
        components = request.get("_component_values") or pack.components
        values["checkpoint"] = components.get("checkpoint")
        values["vae"] = components.get("vae")
        text_encoders = components.get("text_encoders") or []
        for index, value in enumerate(text_encoders, start=1):
            values[f"text_encoder_{index}"] = value
        return values

    @classmethod
    def materialize_execution_plan(
        cls,
        pack: ModelPack,
        template: dict[str, Any],
        request: dict[str, Any],
        execution_plan: ExecutionPlan | None,
    ) -> dict[str, Any]:
        values = dict(request)
        if execution_plan is None:
            return values
        values.update(
            {
                "width": execution_plan.resolution["width"],
                "height": execution_plan.resolution["height"],
                "frames": execution_plan.frames,
                "batch": execution_plan.batch,
                "dtype": execution_plan.dtype,
                "quantization": execution_plan.quantization,
                "attention": execution_plan.attention or "default",
                "vae_tiling": execution_plan.vae_tiling,
                "vae_slicing": execution_plan.vae_slicing,
                "model_cpu_offload": execution_plan.model_cpu_offload,
                "sequential_cpu_offload": execution_plan.sequential_cpu_offload,
                "component_placement": dict(execution_plan.component_placement),
            }
        )
        if execution_plan.fps is not None:
            values["fps"] = execution_plan.fps
        if pack.engine != "comfyui":
            return values
        bindings = template.get("bindings") or {}
        required = set()
        if execution_plan.dtype.upper() not in {"AUTO", "DEFAULT"}:
            required.add("dtype")
        if execution_plan.quantization:
            required.add("quantization")
        if execution_plan.attention not in {None, "", "default"}:
            required.add("attention")
        if execution_plan.vae_tiling:
            required.add("vae_tiling")
        if execution_plan.vae_slicing:
            required.add("vae_slicing")
        if execution_plan.model_cpu_offload:
            required.add("model_cpu_offload")
        if execution_plan.sequential_cpu_offload:
            required.add("sequential_cpu_offload")
        if execution_plan.component_placement:
            required.add("component_placement")
        missing = sorted(required - bindings.keys())
        if missing:
            raise WorkflowValidationError(
                "ExecutionPlan non représentable par le workflow: "
                + ", ".join(missing),
                code="EXECUTION_PLAN_NOT_APPLIED",
            )
        return values

    def build(
        self,
        pack: ModelPack,
        capability: str,
        preset: str,
        request: dict[str, Any],
        *,
        execution_plan: ExecutionPlan | None = None,
        component_values: dict[str, Any] | None = None,
    ) -> BuiltWorkflow:
        template_name = pack.workflow_for(capability)
        if template_name is None:
            raise WorkflowValidationError(f"Capability sans workflow: {capability}")
        template = self.load(template_name)
        workflow = copy.deepcopy(template["workflow"])
        materialized = self.materialize_execution_plan(
            pack,
            template,
            request,
            execution_plan,
        )
        if component_values is not None:
            materialized["_component_values"] = component_values
        values = self._request_values(pack, preset, materialized)
        for name, binding in template["bindings"].items():
            required = bool(binding.get("required", False))
            value = values.get(name)
            if value is None:
                if required:
                    raise WorkflowValidationError(f"Valeur obligatoire absente: {name}", code="WORKFLOW_INPUT_MISSING")
                continue
            value = self._transform_binding_value(
                name,
                value,
                str(binding.get("transform") or ""),
            )
            section = str(binding.get("section") or "inputs")
            if section not in {"inputs", "_meta"}:
                raise WorkflowValidationError(
                    f"Section de binding interdite: {section}",
                )
            workflow[str(binding["node"])].setdefault(section, {})[
                str(binding["field"])
            ] = value
        if execution_plan is not None and execution_plan.vae_tiling:
            tiled = False
            for node in workflow.values():
                if node.get("class_type") == "VAEDecode":
                    node["class_type"] = "VAEDecodeTiled"
                    node["inputs"].setdefault("tile_size", 512)
                    node["inputs"].setdefault("overlap", 64)
                    tiled = True
            if pack.engine == "comfyui" and not tiled:
                raise WorkflowValidationError(
                    "ExecutionPlan requiert VAE tiling, mais aucun VAEDecode compatible n'existe.",
                    code="EXECUTION_PLAN_NOT_APPLIED",
                )
        WorkflowValidator.validate_built(workflow)
        return BuiltWorkflow(
            template=template_name,
            workflow=workflow,
            output_nodes=tuple(str(value) for value in template.get("output_nodes") or []),
        )

    @staticmethod
    def _transform_binding_value(name: str, value: Any, transform: str) -> Any:
        if transform == "comfy_dtype":
            mapping = {
                "AUTO": "default",
                "DEFAULT": "default",
                "FP16": "fp16",
                "BF16": "bf16",
                "FP32": "fp32",
            }
            normalized = str(value).upper()
            if normalized not in mapping:
                raise WorkflowValidationError(
                    f"dtype ComfyUI non supporté: {value}",
                    code="EXECUTION_PLAN_NOT_APPLIED",
                )
            return mapping[normalized]
        if transform == "comfy_quantization":
            normalized = str(value or "none").upper()
            mapping = {
                "NONE": "default",
                "FP8_E4M3FN": "fp8_e4m3fn",
                "FP8_E5M2": "fp8_e5m2",
            }
            if normalized not in mapping:
                raise WorkflowValidationError(
                    f"quantization ComfyUI non supportée: {value}",
                    code="EXECUTION_PLAN_NOT_APPLIED",
                )
            return mapping[normalized]
        if transform:
            raise WorkflowValidationError(
                f"Transformation de binding inconnue pour {name}: {transform}",
            )
        return value
