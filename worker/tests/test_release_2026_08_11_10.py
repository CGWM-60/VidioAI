from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.adapters.generic_diffusers import GenericDiffusersAdapter
from app.main import create_app
from app.memory_planner import GIB, MemoryPlan, MemoryPlanner
from app.runtime import RuntimeManager
from app.video_frame_planner import VideoFramePlanner

from test_worker import CAPABILITY_ENDPOINTS, _base_image_payload, _base_video_payload, settings


def test_frame_planner_prefers_pipeline_default_over_duration_product() -> None:
    class Pipeline:
        def __call__(self, prompt, num_frames=25):
            del prompt, num_frames

    plan = VideoFramePlanner().plan(
        pipeline=Pipeline(), metadata={}, capability="IMAGE_TO_VIDEO",
        requested_duration=6, requested_fps=24, requested_frames=None,
    )
    assert plan.requested_frames == 144
    assert plan.inference_frames == 25
    assert plan.default_frames == 25
    assert plan.source == "pipeline_signature.num_frames"


def test_frame_planner_clamps_and_aligns_explicit_frames() -> None:
    class Pipeline:
        config = SimpleNamespace(max_frames=49, temporal_multiple=4)

        def __call__(self, prompt, num_frames):
            del prompt, num_frames

    plan = VideoFramePlanner().plan(
        pipeline=Pipeline(), metadata={}, capability="TEXT_TO_VIDEO",
        requested_duration=6, requested_fps=24, requested_frames=60,
    )
    assert plan.inference_frames == 49
    assert plan.inference_frames <= 49
    assert (plan.inference_frames - 1) % 4 == 0


def test_frame_planner_keeps_free_explicit_request() -> None:
    class Pipeline:
        def __call__(self, prompt, num_frames):
            del prompt, num_frames

    plan = VideoFramePlanner().plan(
        pipeline=Pipeline(), metadata={}, capability="TEXT_TO_VIDEO",
        requested_duration=6, requested_fps=24, requested_frames=33,
    )
    assert plan.inference_frames == 33


def test_frame_planner_never_injects_unknown_parameter() -> None:
    class Pipeline:
        def __call__(self, prompt):
            del prompt

    plan = VideoFramePlanner().plan(
        pipeline=Pipeline(), metadata={}, capability="TEXT_TO_VIDEO",
        requested_duration=6, requested_fps=24, requested_frames=33,
    )
    assert plan.parameter is None
    assert plan.inference_frames is None


def test_generic_adapter_injects_decode_chunk_only_when_accepted() -> None:
    observed = {}

    class Pipeline:
        def __call__(self, prompt, num_frames=3, decode_chunk_size=None):
            observed.update(num_frames=num_frames, decode_chunk_size=decode_chunk_size)
            return SimpleNamespace(frames=[["frame"]])

    GenericDiffusersAdapter().generate(
        Pipeline(),
        {"metadata": {}, "capability": "TEXT_TO_VIDEO"},
        {"prompt": "test", "capability": "TEXT_TO_VIDEO", "frames": 3, "decode_chunk_size": 1},
    )
    assert observed == {"num_frames": 3, "decode_chunk_size": 1}


def test_memory_planner_uses_final_inference_frames_and_observed_peak() -> None:
    planner = MemoryPlanner()
    values = dict(
        vram_total_bytes=80 * GIB,
        vram_free_bytes=70 * GIB,
        ram_total_bytes=256 * GIB,
        ram_available_bytes=220 * GIB,
        vidioai_ram_bytes=8 * GIB,
        scratch_available_bytes=500 * GIB,
        disk_offload_supported=True,
        model_bytes=20 * GIB,
        precision="FP16",
        capability="IMAGE_TO_VIDEO",
        width=832,
        height=480,
    )
    final = planner.plan(**values, frames=25, observed_previous_peak_bytes=0)
    product = planner.plan(**values, frames=144, observed_previous_peak_bytes=0)
    assert final.frames == 25
    assert final.frames != 144
    assert final.inference_headroom_bytes <= product.inference_headroom_bytes
    observed = planner.plan(**values, frames=25, observed_previous_peak_bytes=30 * GIB)
    assert observed.inference_headroom_bytes >= final.inference_headroom_bytes


def test_memory_planner_48_gib_escalates_from_dangerous_full_gpu() -> None:
    plan = MemoryPlanner().plan(
        vram_total_bytes=48 * GIB,
        vram_free_bytes=44 * GIB,
        ram_total_bytes=96 * GIB,
        ram_available_bytes=88 * GIB,
        vidioai_ram_bytes=4 * GIB,
        scratch_available_bytes=300 * GIB,
        disk_offload_supported=True,
        model_bytes=40 * GIB,
        precision="FP16",
        capability="IMAGE_TO_VIDEO",
        width=832,
        height=480,
        frames=25,
    )
    assert plan.strategy in {"MODEL_CPU_OFFLOAD", "SEQUENTIAL_CPU_OFFLOAD"}


def test_oom_rebuild_cleans_pipeline_and_escalates_once_with_final_frames(tmp_path) -> None:
    manager = object.__new__(RuntimeManager)
    cleaned = []
    emptied = []
    captured = {}
    replacement = object()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    manager._cleanup_disk_offload = lambda pipeline: cleaned.append(("disk", pipeline))
    manager._remove_offload_hooks = lambda pipeline: cleaned.append(("hooks", pipeline))
    manager._active_snapshot = lambda _model_id: (snapshot, {})
    manager._dtype_resolver = SimpleNamespace(materialize=lambda _torch, _plan: "float16")
    manager._release_pipeline_after_oom = lambda _loaded: pytest.fail("plan should be feasible")

    next_plan = MemoryPlan(
        strategy="MODEL_CPU_OFFLOAD", feasible=True,
        vram_total_bytes=80 * GIB, vram_free_bytes=70 * GIB,
        vram_pipeline_bytes=0, ram_total_bytes=256 * GIB,
        ram_available_bytes=220 * GIB, vidioai_ram_bytes=8 * GIB,
        scratch_available_bytes=500 * GIB, model_bytes=20 * GIB,
        estimated_peak_bytes=12 * GIB, inference_headroom_bytes=8 * GIB,
        safety_margin_bytes=8 * GIB, capability="IMAGE_TO_VIDEO",
        width=832, height=480, frames=25,
    )

    def memory_plan(_torch, _metadata, _capability, _precision, _model_bytes, **kwargs):
        captured.update(kwargs)
        return next_plan

    manager._memory_plan = memory_plan
    manager._load_pipeline = lambda **_kwargs: replacement

    class OldPipeline:
        moved_to = None

        def to(self, device):
            self.moved_to = device

    old = OldPipeline()
    loaded = SimpleNamespace(
        model_id="model", pipeline=old, device="cuda", metadata={},
        precision_plan=object(),
        memory_plan=SimpleNamespace(strategy="FULL_GPU", model_bytes=20 * GIB),
        load_benchmark={"vram_peak_bytes": 30 * GIB},
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: emptied.append(True),
    ))

    manager._rebuild_pipeline_after_oom(
        loaded, object(), "IMAGE_TO_VIDEO",
        {"width": 832, "height": 480, "frames": 25}, torch,
    )

    assert old.moved_to == "cpu"
    assert cleaned == [("disk", old), ("hooks", old)]
    assert emptied == [True]
    assert captured["current_strategy"] == "MODEL_CPU_OFFLOAD"
    assert captured["frames"] == 25
    assert loaded.pipeline is replacement
    assert loaded.memory_plan is next_plan


@pytest.mark.parametrize("capability,endpoint", sorted(CAPABILITY_ENDPOINTS.items()))
@pytest.mark.parametrize(
    "status_code,error_code,retryable",
    [(409, "INSUFFICIENT_VRAM", True), (422, "DTYPE_MISMATCH", False), (500, "GENERATION_FAILED", False)],
)
def test_all_generation_endpoints_preserve_runtime_errors(
    tmp_path, monkeypatch, capability, endpoint, status_code, error_code, retryable
) -> None:
    app = create_app(settings(tmp_path))

    def failed(_payload):
        return {
            "state": "FAILED",
            "error": "structured failure",
            "status_code": status_code,
            "error_code": error_code,
            "retryable": retryable,
        }

    monkeypatch.setattr(app.state.manager, "generate_image", failed)
    payload = _base_video_payload() if "VIDEO" in capability else _base_image_payload()
    response = TestClient(app).post(
        endpoint,
        json=payload,
        headers={"X-VidioAI-Worker-Token": "test-token"},
    )
    assert response.status_code == status_code
    assert response.json() == {
        "error": "structured failure",
        "code": error_code,
        "retryable": retryable,
    }
