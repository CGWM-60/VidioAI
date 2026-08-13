"""Adapter around the existing in-process Diffusers runtime."""

from __future__ import annotations

from typing import Any, Callable

from .base import InferenceEngine


class DiffusersEngine(InferenceEngine):
    name = "diffusers"

    def __init__(self, execute_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self._execute_callback = execute_callback

    def health(self) -> dict[str, Any]:
        return {"ready": self._execute_callback is not None, "engine": self.name}

    def execute(
        self,
        payload: dict[str, Any],
        *,
        progress: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        del progress, cancelled
        if self._execute_callback is None:
            raise RuntimeError("Callback Diffusers non configuré.")
        return self._execute_callback(payload)

    def cancel(self, execution_id: str) -> dict[str, Any]:
        return {"execution_id": execution_id, "cancellation_requested": False}

    def free(self) -> dict[str, Any]:
        return {"success": True, "engine": self.name}
