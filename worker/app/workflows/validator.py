"""Static validation for versioned ComfyUI API workflow templates."""

from __future__ import annotations

from typing import Any


class WorkflowValidationError(ValueError):
    def __init__(self, message: str, *, code: str = "WORKFLOW_INVALID") -> None:
        super().__init__(message)
        self.code = code


class WorkflowValidator:
    @staticmethod
    def validate_template(template: dict[str, Any]) -> None:
        if not isinstance(template, dict):
            raise WorkflowValidationError("Le workflow doit être un objet JSON.")
        if int(template.get("schema_version") or 0) != 1:
            raise WorkflowValidationError("Version de workflow non prise en charge.")
        workflow = template.get("workflow")
        bindings = template.get("bindings")
        if not isinstance(workflow, dict) or not workflow:
            raise WorkflowValidationError("Le workflow API ne contient aucun node.")
        if not isinstance(bindings, dict):
            raise WorkflowValidationError("Les bindings du workflow sont absents.")
        for node_id, node in workflow.items():
            if not str(node_id).strip() or not isinstance(node, dict):
                raise WorkflowValidationError("Node workflow invalide.", code="NODE_MISSING")
            if not isinstance(node.get("class_type"), str) or not node["class_type"].strip():
                raise WorkflowValidationError(f"class_type absent pour le node {node_id}.", code="NODE_MISSING")
            if not isinstance(node.get("inputs"), dict):
                raise WorkflowValidationError(f"inputs absent pour le node {node_id}.")
        for name, binding in bindings.items():
            if not isinstance(binding, dict):
                raise WorkflowValidationError(f"Binding invalide: {name}")
            node_id = str(binding.get("node") or "")
            field = str(binding.get("field") or "")
            section = str(binding.get("section") or "inputs")
            if node_id not in workflow:
                raise WorkflowValidationError(f"Node de binding absent: {node_id}", code="NODE_MISSING")
            if not field:
                raise WorkflowValidationError(f"Champ de binding absent: {name}")
            if section not in {"inputs", "_meta"}:
                raise WorkflowValidationError(
                    f"Section de binding interdite: {section}"
                )

    @staticmethod
    def validate_built(workflow: dict[str, Any]) -> None:
        if not isinstance(workflow, dict) or not workflow:
            raise WorkflowValidationError("Workflow construit vide.")
        for node_id, node in workflow.items():
            if not isinstance(node, dict) or not node.get("class_type") or not isinstance(node.get("inputs"), dict):
                raise WorkflowValidationError(f"Node construit invalide: {node_id}", code="NODE_MISSING")
