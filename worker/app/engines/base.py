"""Common engine lifecycle and execution contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class EngineError(RuntimeError):
    def __init__(self, message: str, *, code: str = "ENGINE_ERROR", retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class InferenceEngine(ABC):
    name: str

    @abstractmethod
    def health(self) -> dict[str, Any]: ...

    @abstractmethod
    def execute(
        self,
        payload: dict[str, Any],
        *,
        progress: Callable[[int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def cancel(self, execution_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def free(self) -> dict[str, Any]: ...
