"""Concrete resource plan applied by an inference engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ExecutionPlan:
    strategy: str
    feasible: bool
    dtype: str
    quantization: str | None
    attention: str | None
    vae_tiling: bool
    vae_slicing: bool
    model_cpu_offload: bool
    sequential_cpu_offload: bool
    component_placement: dict[str, str]
    resolution: dict[str, int]
    frames: int
    fps: int | None
    batch: int
    weights_memory_bytes: int
    runtime_memory_bytes: int
    latent_memory_bytes: int
    reserved_memory_bytes: int
    safety_reserve_bytes: int
    estimated_peak_vram_bytes: int
    vram_total_bytes: int
    vram_free_bytes: int
    ram_required_bytes: int
    scratch_required_bytes: int
    fallbacks: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
