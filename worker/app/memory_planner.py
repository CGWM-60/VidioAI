"""Planification generique de la residence GPU des pipelines Diffusers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GIB = 1024**3


@dataclass(slots=True)
class MemoryPlan:
    strategy: str
    feasible: bool
    vram_total_bytes: int
    vram_free_bytes: int
    vram_pipeline_bytes: int
    ram_total_bytes: int
    ram_available_bytes: int
    vidioai_ram_bytes: int
    scratch_available_bytes: int
    model_bytes: int
    estimated_peak_bytes: int
    inference_headroom_bytes: int
    safety_margin_bytes: int
    capability: str
    width: int
    height: int
    frames: int
    optimizations: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryPlanner:
    """Choisit une strategie avant toute copie globale de pipeline vers CUDA."""

    @staticmethod
    def _activation_bytes(
        *, capability: str, width: int, height: int, frames: int, precision: str
    ) -> int:
        element_bytes = 4 if precision == "FP32" else 2
        pixels = max(64, width) * max(64, height)
        if "VIDEO" in capability:
            # Les activations spatio-temporelles dominent rapidement les poids.
            return max(3 * GIB, pixels * max(1, frames) * element_bytes * 32)
        return max(1 * GIB, pixels * element_bytes * 256)

    def plan(
        self,
        *,
        vram_total_bytes: int,
        vram_free_bytes: int,
        model_bytes: int,
        vram_pipeline_bytes: int = 0,
        ram_total_bytes: int = 0,
        ram_available_bytes: int = 0,
        vidioai_ram_bytes: int = 0,
        scratch_available_bytes: int = 0,
        disk_offload_supported: bool = False,
        precision: str,
        capability: str,
        width: int = 1024,
        height: int = 1024,
        frames: int = 1,
        current_strategy: str | None = None,
    ) -> MemoryPlan:
        if vram_total_bytes <= 0:
            return MemoryPlan(
                strategy="CPU",
                feasible=True,
                vram_total_bytes=0,
                vram_free_bytes=0,
                vram_pipeline_bytes=0,
                ram_total_bytes=max(0, ram_total_bytes),
                ram_available_bytes=max(0, ram_available_bytes),
                vidioai_ram_bytes=max(0, vidioai_ram_bytes),
                scratch_available_bytes=max(0, scratch_available_bytes),
                model_bytes=max(0, model_bytes),
                estimated_peak_bytes=0,
                inference_headroom_bytes=0,
                safety_margin_bytes=0,
                capability=capability,
                width=width,
                height=height,
                frames=frames,
                reason="CUDA indisponible",
            )

        safety = max(2 * GIB, int(vram_total_bytes * 0.10))
        current_free = max(0, min(vram_free_bytes, vram_total_bytes))
        pipeline_resident = max(
            0,
            min(vram_pipeline_bytes, max(0, vram_total_bytes - current_free)),
        )
        usable_now = max(0, current_free - safety)
        # Lors d'un replan, la VRAM de la pipeline courante peut être récupérée
        # avant d'activer un offload. Elle ne doit toutefois jamais être comptée
        # comme libre pour conserver FULL_GPU.
        usable_after_transition = max(0, current_free + pipeline_resident - safety)
        ram_safety = max(4 * GIB, int(ram_total_bytes * 0.10))
        usable_ram = max(0, ram_available_bytes - ram_safety)
        precision_factor = 1.7 if precision == "FP32" else 1.0
        resident = int(max(0, model_bytes) * precision_factor)
        activations = self._activation_bytes(
            capability=capability,
            width=width,
            height=height,
            frames=frames,
            precision=precision,
        )
        transient_buffers = max(1 * GIB, int(vram_total_bytes * 0.05))
        inference_headroom = activations + transient_buffers
        model_already_resident = pipeline_resident > 0 and current_strategy == "FULL_GPU"
        full_required = inference_headroom if model_already_resident else resident + inference_headroom
        model_offload_required = int(resident * 0.35) + int(inference_headroom * 0.85)
        sequential_required = int(resident * 0.12) + int(inference_headroom * 0.55)

        optimizations = []
        if "VIDEO" in capability:
            optimizations.extend(["VAE_SLICING", "VAE_TILING"])

        model_offload_ram = int(resident * 0.75) + int(activations * 0.25)
        sequential_ram = int(resident * 0.92) + int(activations * 0.35)
        disk_required = int(resident * 0.05) + int(inference_headroom * 0.50)
        disk_ram = int(resident * 0.15) + int(inference_headroom * 0.20)
        disk_scratch = int(resident * 1.10)

        ranks = {
            "FULL_GPU": 0,
            "MODEL_CPU_OFFLOAD": 1,
            "SEQUENTIAL_CPU_OFFLOAD": 2,
            "DISK_OFFLOAD": 3,
        }
        minimum_rank = ranks.get(current_strategy or "FULL_GPU", 0)

        if minimum_rank <= 0 and full_required <= usable_now:
            strategy = "FULL_GPU"
            required = full_required
            reason = "Poids, activations et buffers tiennent en VRAM avec headroom explicite"
        elif (
            minimum_rank <= 1
            and model_offload_required <= usable_after_transition
            and model_offload_ram <= usable_ram
        ):
            strategy = "MODEL_CPU_OFFLOAD"
            required = model_offload_required
            optimizations.extend(["VAE_SLICING", "VAE_TILING"])
            reason = "FULL_GPU depasse la VRAM sure ; offload modele selectionne"
        elif (
            minimum_rank <= 2
            and sequential_required <= usable_after_transition
            and sequential_ram <= usable_ram
        ):
            strategy = "SEQUENTIAL_CPU_OFFLOAD"
            required = sequential_required
            optimizations.extend(["VAE_SLICING", "VAE_TILING"])
            reason = "Offload sequentiel requis pour conserver la marge CUDA"
        elif (
            disk_offload_supported
            and minimum_rank <= 3
            and disk_required <= usable_after_transition
            and disk_ram <= usable_ram
            and disk_scratch <= scratch_available_bytes
        ):
            strategy = "DISK_OFFLOAD"
            required = disk_required
            optimizations.extend(["VAE_SLICING", "VAE_TILING"])
            reason = "RAM et VRAM insuffisantes ; offload Scratch sélectionné en dernier recours"
        else:
            strategy = "INSUFFICIENT_VRAM"
            required = sequential_required
            reason = "VRAM insuffisante meme avec offload sequentiel"

        return MemoryPlan(
            strategy=strategy,
            feasible=strategy != "INSUFFICIENT_VRAM",
            vram_total_bytes=vram_total_bytes,
            vram_free_bytes=current_free,
            vram_pipeline_bytes=pipeline_resident,
            ram_total_bytes=ram_total_bytes,
            ram_available_bytes=ram_available_bytes,
            vidioai_ram_bytes=vidioai_ram_bytes,
            scratch_available_bytes=scratch_available_bytes,
            model_bytes=model_bytes,
            estimated_peak_bytes=required,
            inference_headroom_bytes=inference_headroom,
            safety_margin_bytes=safety,
            capability=capability,
            width=width,
            height=height,
            frames=frames,
            optimizations=list(dict.fromkeys(optimizations)),
            reason=reason,
        )
