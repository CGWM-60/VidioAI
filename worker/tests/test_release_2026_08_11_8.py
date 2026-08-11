from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from app.adapters.generic_diffusers import GenericDiffusersAdapter
from app.memory_planner import GIB, MemoryPlanner


def _request() -> dict[str, object]:
    return {"capability": "IMAGE_TO_IMAGE", "prompt": "test", "steps": 1}


def test_generic_adapter_returns_non_empty_images_when_frames_are_absent() -> None:
    image = Image.new("RGB", (16, 16))

    class Pipeline:
        def __call__(self, prompt):
            del prompt
            return SimpleNamespace(images=[image])

    output = GenericDiffusersAdapter().generate(Pipeline(), {"metadata": {}}, _request())
    assert output["images"] == [image]
    assert "frames" not in output


def test_empty_frames_never_mask_valid_images() -> None:
    image = Image.new("RGB", (16, 16))

    class Pipeline:
        def __call__(self, prompt):
            del prompt
            return SimpleNamespace(images=[image], frames=[])

    output = GenericDiffusersAdapter().generate(Pipeline(), {"metadata": {}}, _request())
    assert output["images"] == [image]
    assert "frames" not in output


def test_memory_planner_uses_offload_when_full_gpu_is_not_safe() -> None:
    plan = MemoryPlanner().plan(
        vram_total_bytes=44 * GIB,
        vram_free_bytes=18 * GIB,
        ram_total_bytes=120 * GIB,
        ram_available_bytes=90 * GIB,
        vidioai_ram_bytes=4 * GIB,
        scratch_available_bytes=500 * GIB,
        model_bytes=22 * GIB,
        precision="FP16",
        capability="IMAGE_TO_VIDEO",
        width=832,
        height=480,
        frames=49,
    )
    assert plan.feasible is True
    assert plan.strategy in {"MODEL_CPU_OFFLOAD", "SEQUENTIAL_CPU_OFFLOAD"}


def test_memory_planner_refuses_cpu_offload_when_ram_is_not_safe() -> None:
    plan = MemoryPlanner().plan(
        vram_total_bytes=44 * GIB,
        vram_free_bytes=18 * GIB,
        ram_total_bytes=120 * GIB,
        ram_available_bytes=5 * GIB,
        vidioai_ram_bytes=4 * GIB,
        scratch_available_bytes=500 * GIB,
        disk_offload_supported=False,
        model_bytes=22 * GIB,
        precision="FP16",
        capability="IMAGE_TO_VIDEO",
        width=832,
        height=480,
        frames=49,
    )
    assert plan.feasible is False
    assert plan.strategy == "INSUFFICIENT_VRAM"
