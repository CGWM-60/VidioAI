"""Adaptateur de progression commun aux pipelines Diffusers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class GenerationProgressReporter:
    def __init__(
        self,
        *,
        total_steps: int,
        cancelled: Callable[[], bool],
        emit: Callable[[int], None],
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.cancelled = cancelled
        self.emit = emit

    def __call__(
        self,
        _pipeline: Any,
        step: int,
        _timestep: Any,
        callback_kwargs: Any,
    ) -> Any:
        if self.cancelled():
            raise InterruptedError("Job annule.")
        self.emit(min(95, int(((step + 1) / self.total_steps) * 95)))
        return callback_kwargs
