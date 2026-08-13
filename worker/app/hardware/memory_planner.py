"""Pack-aware memory planner producing an enforceable ExecutionPlan."""

from __future__ import annotations

from typing import Any

from .detector import HardwareProfile
from .execution_plan import ExecutionPlan


GIB = 1024**3


class MemoryPlanner:
    @staticmethod
    def _memory_breakdown(
        *, weights: int, width: int, height: int, frames: int, batch: int, dtype: str, video: bool
    ) -> tuple[int, int, int]:
        element_bytes = 4 if dtype.upper() == "FP32" else 2
        pixels = max(64, width) * max(64, height) * max(1, batch)
        latent = max(256 * 1024**2, pixels * max(1, frames if video else 1) * element_bytes * 8)
        runtime = max(1 * GIB, (3 * GIB if video else 1 * GIB), latent * (4 if video else 2))
        reserved = max(1 * GIB, int(weights * 0.03))
        return runtime, latent, reserved

    def execution_plan(
        self,
        *,
        hardware: HardwareProfile,
        model_pack: Any,
        weights_memory_bytes: int,
        dtype: str,
        quantization: str | None,
        width: int,
        height: int,
        frames: int,
        fps: int | None,
        batch: int = 1,
        ram_available_bytes: int | None = None,
        scratch_available_bytes: int = 0,
    ) -> ExecutionPlan:
        gpu = hardware.primary_gpu
        policy = dict(getattr(model_pack, "memory_policy", {}) or {})
        component_placement = dict(policy.get("component_placement") or {})
        weights = max(0, int(weights_memory_bytes))
        effective_weights = weights
        effective_dtype = dtype.upper()
        effective_quantization = quantization
        video = any("VIDEO" in value for value in getattr(model_pack, "capabilities", ()))
        chosen_width = max(64, int(width))
        chosen_height = max(64, int(height))
        chosen_frames = max(1, int(frames))
        runtime, latent, reserved = self._memory_breakdown(
            weights=effective_weights,
            width=chosen_width,
            height=chosen_height,
            frames=chosen_frames,
            batch=batch,
            dtype=effective_dtype,
            video=video,
        )
        total = gpu.vram_total_bytes if gpu else 0
        free = gpu.vram_free_bytes if gpu else 0
        safety = int(policy.get("safety_reserve_bytes") or max(2 * GIB, int(total * 0.10))) if total else 0
        available = max(0, free - safety)
        ram_available = hardware.ram_available_bytes if ram_available_bytes is None else ram_available_bytes
        fallbacks: list[str] = []

        if gpu is None:
            strategy = "CPU"
            feasible = not bool(policy.get("cuda_required", False))
            reason = "CUDA indisponible; exécution CPU autorisée par le ModelPack." if feasible else "Le ModelPack requiert CUDA."
            gpu_weights = 0
            ram_required = effective_weights + runtime
        else:
            def select(*, allow_offload: bool) -> tuple[str, bool, int, int, str] | None:
                full_peak = effective_weights + runtime + latent + reserved
                force_strategy = str(policy.get("force_strategy") or "").upper()
                # Limit permanent weights independently from the transient peak.
                if (
                    force_strategy not in {"MODEL_CPU_OFFLOAD", "SEQUENTIAL_CPU_OFFLOAD"}
                    and full_peak <= available
                    and effective_weights <= int(total * 0.72)
                ):
                    return (
                        "FULL_GPU",
                        True,
                        effective_weights,
                        int(effective_weights * 0.10),
                        "Poids et mémoire d'inférence tiennent avec réserve de sécurité.",
                    )
                if not allow_offload:
                    return None
                model_gpu = min(int(effective_weights * 0.58), max(0, available - runtime - latent - reserved))
                model_ram = effective_weights - model_gpu + int(runtime * 0.25)
                if (
                    force_strategy != "SEQUENTIAL_CPU_OFFLOAD"
                    and policy.get("supports_cpu_offload", True)
                    and model_gpu + runtime + latent + reserved <= available
                    and model_ram <= max(0, ram_available - 4 * GIB)
                ):
                    return (
                        "MODEL_CPU_OFFLOAD",
                        True,
                        model_gpu,
                        model_ram,
                        "Résidence permanente limitée; offload CPU avec marge GPU.",
                    )
                sequential_gpu = min(int(effective_weights * 0.22), max(0, available - runtime - latent - reserved))
                sequential_ram = effective_weights - sequential_gpu + int(runtime * 0.40)
                if (
                    policy.get("supports_sequential_offload", True)
                    and sequential_gpu + runtime + latent + reserved <= available
                    and sequential_ram <= max(0, ram_available - 4 * GIB)
                ):
                    return (
                        "SEQUENTIAL_CPU_OFFLOAD",
                        True,
                        sequential_gpu,
                        sequential_ram,
                        "Offload séquentiel choisi pour préserver la réserve GPU.",
                    )
                return None

            selected = select(allow_offload=False)
            if selected is None and effective_dtype == "FP32":
                effective_dtype = str(policy.get("preferred_dtype") or "FP16").upper()
                effective_weights = max(1, effective_weights // 2)
                runtime, latent, reserved = self._memory_breakdown(
                    weights=effective_weights, width=chosen_width, height=chosen_height,
                    frames=chosen_frames, batch=batch, dtype=effective_dtype, video=video,
                )
                fallbacks.append(f"DTYPE_{effective_dtype}")
                selected = select(allow_offload=False)
            if selected is None and policy.get("efficient_attention", True):
                runtime = int(runtime * 0.82)
                fallbacks.append("EFFICIENT_ATTENTION")
                selected = select(allow_offload=False)
            if selected is None:
                runtime = int(runtime * 0.92)
                fallbacks.append("VAE_SLICING")
                selected = select(allow_offload=False)
            if selected is None:
                latent = int(latent * 0.80)
                fallbacks.append("VAE_TILING")
                selected = select(allow_offload=False)
            if selected is None:
                selected = select(allow_offload=True)
                if selected is not None:
                    fallbacks.append(selected[0])
            if selected is None and policy.get("supports_quantization", False) and not effective_quantization:
                effective_quantization = "INT8_WEIGHT_ONLY"
                effective_weights = max(1, effective_weights // 2)
                reserved = max(1 * GIB, int(effective_weights * 0.03))
                fallbacks.append("QUANTIZATION_INT8")
                selected = select(allow_offload=True)
                if selected is not None and selected[0] != "FULL_GPU":
                    fallbacks.append(selected[0])
            if selected is None:
                chosen_width = max(64, (int(chosen_width * 0.75) // 32) * 32)
                chosen_height = max(64, (int(chosen_height * 0.75) // 32) * 32)
                runtime, latent, reserved = self._memory_breakdown(
                    weights=effective_weights, width=chosen_width, height=chosen_height,
                    frames=chosen_frames, batch=batch, dtype=effective_dtype, video=video,
                )
                runtime = int(runtime * 0.82 * 0.92)
                latent = int(latent * 0.80)
                fallbacks.append("RESOLUTION_REDUCED")
                selected = select(allow_offload=True)
                if selected is not None and selected[0] != "FULL_GPU":
                    fallbacks.append(selected[0])
            if selected is None and video and chosen_frames > 1:
                chosen_frames = max(1, chosen_frames // 2)
                runtime, latent, reserved = self._memory_breakdown(
                    weights=effective_weights, width=chosen_width, height=chosen_height,
                    frames=chosen_frames, batch=batch, dtype=effective_dtype, video=video,
                )
                runtime = int(runtime * 0.82 * 0.92)
                latent = int(latent * 0.80)
                fallbacks.append("FRAMES_REDUCED")
                selected = select(allow_offload=True)
                if selected is not None and selected[0] != "FULL_GPU":
                    fallbacks.append(selected[0])
            if selected is None:
                strategy, feasible, gpu_weights, ram_required, reason = (
                    "INSUFFICIENT_VRAM",
                    False,
                    0,
                    effective_weights + runtime,
                    "Aucune stratégie déclarée par le ModelPack ne respecte les marges GPU/RAM.",
                )
            else:
                strategy, feasible, gpu_weights, ram_required, reason = selected

        estimated = gpu_weights + runtime + latent + reserved
        placement = component_placement or {
            "transformer": "gpu" if strategy == "FULL_GPU" else "gpu_temporary",
            "text_encoder": "gpu" if strategy == "FULL_GPU" else "cpu_offload",
            "vae": "gpu" if strategy == "FULL_GPU" else "cpu_offload",
        }
        return ExecutionPlan(
            strategy=strategy,
            feasible=feasible,
            dtype=effective_dtype,
            quantization=effective_quantization,
            attention="efficient" if "EFFICIENT_ATTENTION" in fallbacks else policy.get("attention"),
            vae_tiling="VAE_TILING" in fallbacks,
            vae_slicing="VAE_SLICING" in fallbacks,
            model_cpu_offload=strategy == "MODEL_CPU_OFFLOAD",
            sequential_cpu_offload=strategy == "SEQUENTIAL_CPU_OFFLOAD",
            component_placement=placement,
            resolution={"width": chosen_width, "height": chosen_height},
            frames=chosen_frames,
            fps=int(fps) if fps else None,
            batch=max(1, int(batch)),
            weights_memory_bytes=effective_weights,
            runtime_memory_bytes=runtime,
            latent_memory_bytes=latent,
            reserved_memory_bytes=reserved,
            safety_reserve_bytes=safety,
            estimated_peak_vram_bytes=estimated,
            vram_total_bytes=total,
            vram_free_bytes=free,
            ram_required_bytes=ram_required,
            scratch_required_bytes=0,
            fallbacks=list(dict.fromkeys(fallbacks)),
            reason=reason,
        )


MemoryPlan = ExecutionPlan
