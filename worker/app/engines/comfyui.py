"""Headless ComfyUI HTTP client using only the Python standard library."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

from .base import EngineError, InferenceEngine


class ComfyUIEngine(InferenceEngine):
    name = "comfyui"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 15.0,
        poll_interval_seconds: float = 0.5,
        execution_timeout_seconds: float = 3600.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            raise EngineError(f"ComfyUI indisponible: {error}", code="COMFYUI_UNAVAILABLE", retryable=True) from error
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EngineError("Réponse ComfyUI invalide.", code="COMFYUI_RESPONSE_INVALID") from error

    def _binary_request(self, path: str) -> bytes:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method="GET",
            headers={"Accept": "application/octet-stream,image/*,video/*"},
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            raise EngineError(
                f"Sortie ComfyUI indisponible: {error}",
                code="COMFYUI_OUTPUT_UNAVAILABLE",
                retryable=True,
            ) from error
        if not raw:
            raise EngineError(
                "ComfyUI a retourné une sortie vide.",
                code="COMFYUI_OUTPUT_INVALID",
            )
        return raw

    def health(self) -> dict[str, Any]:
        try:
            payload = self._request("GET", "/system_stats")
            return {"ready": isinstance(payload, dict), "engine": self.name, "details": payload}
        except EngineError as error:
            return {"ready": False, "engine": self.name, "error": str(error), "error_code": error.code}

    def queue(self) -> dict[str, Any]:
        value = self._request("GET", "/queue")
        return value if isinstance(value, dict) else {}

    def history(self, prompt_id: str) -> dict[str, Any]:
        encoded = urllib.parse.quote(prompt_id, safe="")
        history = self._request("GET", f"/history/{encoded}")
        record = history.get(prompt_id, {}) if isinstance(history, dict) else {}
        return record if isinstance(record, dict) else {}

    def outputs(self, prompt_id: str) -> dict[str, Any]:
        record = self.history(prompt_id)
        outputs = record.get("outputs") or {}
        return outputs if isinstance(outputs, dict) else {}

    @staticmethod
    def output_descriptors(outputs: dict[str, Any]) -> list[dict[str, str]]:
        """Flatten the descriptors produced by SaveImage/VideoCombine nodes."""
        descriptors: list[dict[str, str]] = []
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for values in node_output.values():
                if not isinstance(values, list):
                    continue
                for value in values:
                    if not isinstance(value, dict) or not value.get("filename"):
                        continue
                    descriptors.append(
                        {
                            "filename": str(value["filename"]),
                            "subfolder": str(value.get("subfolder") or ""),
                            "type": str(value.get("type") or "output"),
                        }
                    )
        return descriptors

    def view(self, descriptor: dict[str, Any]) -> bytes:
        filename = str(descriptor.get("filename") or "")
        subfolder = str(descriptor.get("subfolder") or "")
        output_type = str(descriptor.get("type") or "output")
        if (
            not filename
            or "\x00" in filename
            or "\x00" in subfolder
            or output_type not in {"output", "temp"}
            or any(part == ".." for part in filename.replace("\\", "/").split("/"))
            or any(part == ".." for part in subfolder.replace("\\", "/").split("/"))
        ):
            raise EngineError(
                "Descripteur de sortie ComfyUI invalide.",
                code="COMFYUI_OUTPUT_INVALID",
            )
        query = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": output_type}
        )
        return self._binary_request(f"/view?{query}")

    @staticmethod
    def _history_error(record: dict[str, Any]) -> str | None:
        status = record.get("status") or {}
        if not isinstance(status, dict):
            return None
        status_string = str(status.get("status_str") or "").lower()
        completed = status.get("completed")
        if status_string not in {"error", "failed"} and completed is not False:
            return None
        messages = status.get("messages") or record.get("messages") or []
        return f"ComfyUI a échoué: {messages or status_string or 'erreur inconnue'}"

    @staticmethod
    def _progress_from_queue(queue: dict[str, Any], prompt_id: str) -> int:
        for value in queue.get("queue_running") or []:
            if isinstance(value, (list, tuple)) and prompt_id in {
                str(item) for item in value
            }:
                return 50
        for index, value in enumerate(queue.get("queue_pending") or []):
            if isinstance(value, (list, tuple)) and prompt_id in {
                str(item) for item in value
            }:
                return min(40, 5 + index)
        return 90

    def execute(
        self,
        payload: dict[str, Any],
        *,
        progress: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        workflow = payload.get("workflow")
        if not isinstance(workflow, dict):
            raise EngineError("Workflow ComfyUI absent.", code="WORKFLOW_INVALID")
        client_id = str(payload.get("client_id") or uuid.uuid4())
        queued = self._request("POST", "/prompt", {"prompt": workflow, "client_id": client_id})
        prompt_id = str(queued.get("prompt_id") or "") if isinstance(queued, dict) else ""
        if not prompt_id:
            raise EngineError("ComfyUI n'a pas retourné prompt_id.", code="COMFYUI_QUEUE_FAILED")
        if progress:
            progress(1)
        deadline = time.monotonic() + self.execution_timeout_seconds
        while True:
            if time.monotonic() >= deadline:
                try:
                    self.cancel(prompt_id)
                except EngineError:
                    pass
                raise EngineError(
                    "Délai d'exécution ComfyUI dépassé.",
                    code="COMFYUI_EXECUTION_TIMEOUT",
                    retryable=True,
                )
            if cancelled and cancelled():
                self.cancel(prompt_id)
                raise InterruptedError("Job ComfyUI annulé.")
            history = self.history(prompt_id)
            error = self._history_error(history)
            if error:
                raise EngineError(error, code="COMFYUI_EXECUTION_FAILED")
            outputs = history.get("outputs") or {}
            if outputs:
                if progress:
                    progress(100)
                return {"prompt_id": prompt_id, "outputs": outputs}
            queue = self.queue()
            if progress:
                progress(self._progress_from_queue(queue, prompt_id))
            time.sleep(self.poll_interval_seconds)

    def cancel(self, execution_id: str) -> dict[str, Any]:
        try:
            self._request("POST", "/queue", {"delete": [execution_id]})
            self._request("POST", "/interrupt", {})
        except EngineError as error:
            raise EngineError(str(error), code="COMFYUI_CANCEL_FAILED", retryable=True) from error
        return {"execution_id": execution_id, "cancellation_requested": True}

    def free(self) -> dict[str, Any]:
        self._request("POST", "/free", {"unload_models": True, "free_memory": True})
        return {"success": True, "engine": self.name}
