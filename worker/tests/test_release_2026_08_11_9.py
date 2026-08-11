from __future__ import annotations

import threading
from types import SimpleNamespace

from app.memory_planner import GIB, MemoryPlan, MemoryPlanner
from app.runtime import RuntimeManager


def _plan(**overrides):
    values = {
        "vram_total_bytes": 44 * GIB,
        "vram_free_bytes": 40 * GIB,
        "ram_total_bytes": 120 * GIB,
        "ram_available_bytes": 96 * GIB,
        "vidioai_ram_bytes": 8 * GIB,
        "scratch_available_bytes": 500 * GIB,
        "disk_offload_supported": True,
        "model_bytes": 4 * GIB,
        "precision": "FP16",
        "capability": "TEXT_TO_VIDEO",
        "width": 832,
        "height": 480,
        "frames": 49,
    }
    values.update(overrides)
    return MemoryPlanner().plan(**values)


def test_full_gpu_is_refused_when_loaded_pipeline_leaves_no_inference_headroom() -> None:
    plan = _plan(
        vram_free_bytes=int(4.4 * GIB),
        vram_pipeline_bytes=int(39.6 * GIB),
        model_bytes=40 * GIB,
        current_strategy="FULL_GPU",
    )
    assert plan.strategy == "MODEL_CPU_OFFLOAD"
    assert plan.inference_headroom_bytes > plan.vram_free_bytes


def test_memory_planner_fallback_order_uses_ram_then_scratch() -> None:
    model_offload = _plan(vram_free_bytes=18 * GIB, model_bytes=22 * GIB)
    assert model_offload.strategy == "MODEL_CPU_OFFLOAD"

    sequential = _plan(vram_free_bytes=13 * GIB, model_bytes=22 * GIB)
    assert sequential.strategy == "SEQUENTIAL_CPU_OFFLOAD"

    disk = _plan(
        vram_free_bytes=9 * GIB,
        ram_total_bytes=32 * GIB,
        ram_available_bytes=10 * GIB,
        model_bytes=22 * GIB,
    )
    assert disk.strategy == "DISK_OFFLOAD"

    insufficient = _plan(
        vram_free_bytes=9 * GIB,
        ram_total_bytes=32 * GIB,
        ram_available_bytes=10 * GIB,
        scratch_available_bytes=0,
        disk_offload_supported=False,
        model_bytes=22 * GIB,
    )
    assert insufficient.strategy == "INSUFFICIENT_VRAM"
    assert insufficient.feasible is False


def test_request_replan_receives_real_model_and_pipeline_sizes() -> None:
    manager = object.__new__(RuntimeManager)
    manager._lock = threading.RLock()
    manager._model_states = {"model": {}}
    captured = {}

    request_plan = MemoryPlan(
        strategy="MODEL_CPU_OFFLOAD",
        feasible=True,
        vram_total_bytes=44 * GIB,
        vram_free_bytes=4 * GIB,
        vram_pipeline_bytes=39 * GIB,
        ram_total_bytes=120 * GIB,
        ram_available_bytes=96 * GIB,
        vidioai_ram_bytes=8 * GIB,
        scratch_available_bytes=500 * GIB,
        model_bytes=40 * GIB,
        estimated_peak_bytes=18 * GIB,
        inference_headroom_bytes=5 * GIB,
        safety_margin_bytes=4 * GIB,
        capability="TEXT_TO_VIDEO",
        width=832,
        height=480,
        frames=49,
    )

    def memory_plan(_torch, _metadata, _capability, _precision, model_bytes, **kwargs):
        captured.update(model_bytes=model_bytes, **kwargs)
        return request_plan

    manager._memory_plan = memory_plan
    manager._cleanup_disk_offload = lambda _pipeline: None
    manager._remove_offload_hooks = lambda _pipeline: None
    manager._apply_memory_plan = lambda pipeline, _plan, _device: pipeline
    pipeline = SimpleNamespace(to=lambda _device: None)
    loaded = SimpleNamespace(
        model_id="model",
        device="cuda",
        metadata={},
        precision_plan=SimpleNamespace(precision="FP16"),
        memory_plan=SimpleNamespace(model_bytes=40 * GIB, strategy="FULL_GPU"),
        load_benchmark={
            "vram_idle_bytes": 1 * GIB,
            "vram_after_load_bytes": 40 * GIB,
        },
        pipeline=pipeline,
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

    manager._adapt_memory_plan_for_request(
        loaded,
        {"width": 832, "height": 480, "frames": 49},
        "TEXT_TO_VIDEO",
        torch,
    )

    assert captured["model_bytes"] == 40 * GIB
    assert captured["vram_pipeline_bytes"] == 39 * GIB
    assert captured["width"] == 832
    assert captured["height"] == 480
    assert captured["frames"] == 49
    assert loaded.memory_plan is request_plan


def test_cuda_oom_cleanup_unloads_pipeline_and_exposes_installed_state(tmp_path) -> None:
    manager = object.__new__(RuntimeManager)
    manager._lock = threading.RLock()
    manager._loaded = {}
    manager._model_states = {}
    emptied = []
    manager._imports = lambda: SimpleNamespace(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                empty_cache=lambda: emptied.append(True),
            )
        )
    )
    offload = tmp_path / "offload"
    offload.mkdir()
    pipeline = SimpleNamespace(_vidioai_disk_offload_dir=str(offload))
    loaded = SimpleNamespace(
        model_id="model",
        repository="owner/model",
        revision="revision",
        pipeline=pipeline,
    )
    manager._loaded["model"] = loaded

    manager._release_pipeline_after_oom(loaded)

    assert loaded.pipeline is None
    assert "model" not in manager._loaded
    assert manager._model_states["model"]["state"] == "INSTALLED"
    assert manager._model_states["model"]["error_code"] == "INSUFFICIENT_VRAM"
    assert not offload.exists()
    assert emptied == [True]
