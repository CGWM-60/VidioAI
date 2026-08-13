"""Portable hardware detector with optional, encapsulated NVIDIA providers."""

from __future__ import annotations

import platform
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class GpuTelemetry:
    index: int
    name: str
    backend: str
    vram_total_bytes: int
    vram_free_bytes: int
    vram_used_bytes: int
    driver_version: str | None = None
    compute_capability: str | None = None
    utilization_percent: float | None = None
    temperature_celsius: float | None = None
    processes: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["processes"] = list(self.processes)
        return value


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    platform: str
    architecture: str
    cpu_count: int
    ram_total_bytes: int
    ram_available_bytes: int
    cuda_available: bool
    cuda_version: str | None
    apple_silicon: bool
    gpus: tuple[GpuTelemetry, ...] = ()
    torch_memory_allocated_bytes: int = 0
    torch_memory_reserved_bytes: int = 0
    vidioai_model_resident_bytes: int = 0

    @property
    def primary_gpu(self) -> GpuTelemetry | None:
        return self.gpus[0] if self.gpus else None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gpus"] = [item.as_dict() for item in self.gpus]
        value["gpu_count"] = len(self.gpus)
        return value


class HardwareDetector:
    def __init__(
        self,
        *,
        torch_provider: Callable[[], Any] | None = None,
        memory_provider: Callable[[], dict[str, int]] | None = None,
        nvidia_provider: Callable[[], list[GpuTelemetry]] | None = None,
    ) -> None:
        self._torch_provider = torch_provider
        self._memory_provider = memory_provider or (lambda: {})
        self._nvidia_provider = nvidia_provider

    @staticmethod
    def _nvidia_smi() -> list[GpuTelemetry]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,compute_cap,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        gpus: list[GpuTelemetry] = []
        for line in result.stdout.splitlines():
            try:
                index, name, driver, capability, total, free, used, utilization, temperature = [
                    value.strip() for value in line.split(",", maxsplit=8)
                ]
                gpus.append(
                    GpuTelemetry(
                        index=int(index),
                        name=name,
                        backend="CUDA",
                        vram_total_bytes=int(total) * 1024 * 1024,
                        vram_free_bytes=int(free) * 1024 * 1024,
                        vram_used_bytes=int(used) * 1024 * 1024,
                        driver_version=driver,
                        compute_capability=capability,
                        utilization_percent=float(utilization),
                        temperature_celsius=float(temperature),
                    )
                )
            except (TypeError, ValueError):
                continue
        return gpus

    def detect(self, *, model_resident_bytes: int = 0) -> HardwareProfile:
        memory = self._memory_provider()
        torch = self._torch_provider() if self._torch_provider is not None else None
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
        gpus: list[GpuTelemetry] = []
        allocated = 0
        reserved = 0
        if cuda_available:
            disable_probe = os.getenv("VIDIOAI_DISABLE_NVIDIA_PROBE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if self._nvidia_provider is not None:
                gpus = list(self._nvidia_provider())
            elif not disable_probe:
                gpus = list(self._nvidia_smi())
            try:
                allocated = int(torch.cuda.memory_allocated())
                reserved = int(torch.cuda.memory_reserved())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            if not gpus and hasattr(torch.cuda, "mem_get_info"):
                try:
                    free, total = torch.cuda.mem_get_info()
                    name = torch.cuda.get_device_name(0) if hasattr(torch.cuda, "get_device_name") else "NVIDIA CUDA"
                    capability = torch.cuda.get_device_capability(0) if hasattr(torch.cuda, "get_device_capability") else None
                    gpus = [
                        GpuTelemetry(
                            index=0,
                            name=str(name),
                            backend="CUDA",
                            vram_total_bytes=int(total),
                            vram_free_bytes=int(free),
                            vram_used_bytes=max(0, int(total) - int(free)),
                            compute_capability=(".".join(map(str, capability)) if capability else None),
                        )
                    ]
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
        elif self._nvidia_provider is not None:
            # Explicit providers are used by deterministic hardware simulation
            # tests and host-agent integrations without probing local CUDA.
            gpus = list(self._nvidia_provider())
            cuda_available = bool(gpus)
        machine = platform.machine().lower()
        system = platform.system()
        return HardwareProfile(
            platform=system,
            architecture=machine,
            cpu_count=int(memory.get("cpu_count") or os.cpu_count() or 0),
            ram_total_bytes=int(memory.get("ram_total_bytes") or 0),
            ram_available_bytes=int(memory.get("ram_available_bytes") or 0),
            cuda_available=cuda_available,
            cuda_version=str(cuda_version) if cuda_version else None,
            apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
            gpus=tuple(gpus),
            torch_memory_allocated_bytes=allocated,
            torch_memory_reserved_bytes=reserved,
            vidioai_model_resident_bytes=max(0, int(model_resident_bytes)),
        )

    @staticmethod
    def occupied_diagnostic(profile: HardwareProfile, *, loaded_models: int) -> dict[str, Any]:
        gpu = profile.primary_gpu
        occupied = bool(gpu and loaded_models == 0 and gpu.vram_used_bytes > max(2 * 1024**3, int(gpu.vram_total_bytes * 0.20)))
        return {
            "gpu_memory_occupied": occupied,
            "error_code": "GPU_MEMORY_OCCUPIED" if occupied else None,
            "nvml_gpu_used_bytes": gpu.vram_used_bytes if gpu else 0,
            "torch_allocated_bytes": profile.torch_memory_allocated_bytes,
            "torch_reserved_bytes": profile.torch_memory_reserved_bytes,
            "vidioai_model_resident_bytes": profile.vidioai_model_resident_bytes,
            "gpu_processes": list(gpu.processes) if gpu else [],
        }
