from __future__ import annotations

import json
import io
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

from app.engines.base import EngineError
from app.engines.comfyui import ComfyUIEngine
from app.generation.preflight import PreflightService
from app.hardware.detector import GpuTelemetry, HardwareDetector, HardwareProfile
from app.hardware.memory_planner import GIB, MemoryPlanner
from app.hardware.execution_plan import ExecutionPlan
from app.packs.registry import ModelPackRegistry
from app.packs.resolver import ModelPackResolver
from app.packs.schema import ModelPackStatus
from app.runtime import RuntimeManager
from app.workflows.builder import WorkflowBuilder
from app.workflows.comfy_models import ComfyModelError, ComfyModelMaterializer
from app.workflows.validator import WorkflowValidationError, WorkflowValidator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKS_DIRECTORY = REPOSITORY_ROOT / "model-packs"
WORKFLOWS_DIRECTORY = REPOSITORY_ROOT / "workflows"


def _registry() -> ModelPackRegistry:
    return ModelPackRegistry(PACKS_DIRECTORY)


def _gpu_profile(
    vram_gib: int,
    *,
    free_gib: int | None = None,
    ram_gib: int = 128,
    name: str = "Simulated NVIDIA GPU",
) -> HardwareProfile:
    free = vram_gib if free_gib is None else free_gib
    gpu = GpuTelemetry(
        index=0,
        name=name,
        backend="CUDA",
        vram_total_bytes=vram_gib * GIB,
        vram_free_bytes=free * GIB,
        vram_used_bytes=(vram_gib - free) * GIB,
        driver_version="simulated",
        compute_capability="9.0",
    )
    return HardwareProfile(
        platform="Linux",
        architecture="x86_64",
        cpu_count=32,
        ram_total_bytes=ram_gib * GIB,
        ram_available_bytes=ram_gib * GIB,
        cuda_available=True,
        cuda_version="simulated",
        apple_silicon=False,
        gpus=(gpu,),
    )


class _JsonResponse:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _BytesResponse(_JsonResponse):
    def read(self) -> bytes:
        return bytes(self.payload)


def _request_details(request: Any) -> tuple[str, str, dict[str, Any]]:
    body = json.loads(request.data.decode("utf-8")) if request.data else {}
    return request.get_method(), urlsplit(request.full_url).path, body


def test_model_pack_manifests_are_versioned_and_complete() -> None:
    registry = _registry()
    expected = {
        "flux-t2i-v1",
        "wan22-t2v-v1",
        "wan22-i2v-v1",
        "ltx-t2v-v1",
        "ltx-i2v-v1",
        "minimax-h3-diffusers-v1",
        "legacy-diffusers-v1",
    }

    assert expected <= {pack.id for pack in registry.all()}
    for pack in registry:
        assert pack.schema_version == 1
        assert pack.status in {
            ModelPackStatus.READY,
            ModelPackStatus.EXPERIMENTAL,
            ModelPackStatus.DOWNLOADABLE,
        }
        assert pack.engine in {"diffusers", "comfyui"}
        assert pack.capabilities
        assert all(pack.workflow_for(capability) for capability in pack.capabilities)


@pytest.mark.parametrize(
    ("capability", "metadata", "expected_pack"),
    [
        (
            "TEXT_TO_IMAGE",
            {"architectures": ["FluxTransformer2DModel"]},
            "flux-t2i-v1",
        ),
        (
            "TEXT_TO_VIDEO",
            {"architectures": ["WanTransformer3DModel"]},
            "wan22-t2v-v1",
        ),
        (
            "IMAGE_TO_VIDEO",
            {"pipeline_class": "WanImageToVideoPipeline"},
            "wan22-i2v-v1",
        ),
        (
            "TEXT_TO_VIDEO",
            {"pipeline_class": "LTXPipeline"},
            "ltx-t2v-v1",
        ),
        (
            "IMAGE_TO_VIDEO",
            {"pipeline_class": "LTXImageToVideoPipeline"},
            "ltx-i2v-v1",
        ),
        (
            "TEXT_TO_VIDEO",
            {"architectures": ["MiniMaxH3Transformer3DModel"]},
            "minimax-h3-diffusers-v1",
        ),
    ],
)
def test_model_pack_resolution_uses_architecture_or_pipeline_class(
    capability: str,
    metadata: dict[str, Any],
    expected_pack: str,
) -> None:
    resolver = ModelPackResolver(_registry())
    metadata_with_irrelevant_repository = {
        **metadata,
        "repository": "unrelated-owner/unrelated-model",
        "repo_id": "unrelated-owner/another-model",
    }

    resolution = resolver.resolve(metadata_with_irrelevant_repository, capability)

    assert resolution.pack is not None
    assert resolution.pack.id == expected_pack
    assert set(resolution.matched_by) & {"architecture", "pipeline_class"}


def test_model_pack_resolution_does_not_infer_family_from_repo_id() -> None:
    resolution = ModelPackResolver(_registry()).resolve(
        {"repo_id": "black-forest-labs/FLUX.1-dev"},
        "TEXT_TO_IMAGE",
    )

    assert resolution.pack is None
    assert resolution.status is ModelPackStatus.UNSUPPORTED


def test_all_declared_workflows_validate_and_build() -> None:
    registry = _registry()
    builder = WorkflowBuilder(WORKFLOWS_DIRECTORY)
    request = {
        "prompt": "A deterministic worker foundation test",
        "negative_prompt": "",
        "input_path": "fixtures/input.png",
        "width": 640,
        "height": 384,
        "frames": 17,
        "fps": 8,
        "steps": 7,
        "guidance_scale": 2.0,
        "seed": 42,
    }

    for path in sorted(WORKFLOWS_DIRECTORY.glob("*.json")):
        WorkflowValidator.validate_template(builder.load(path.name))

    for pack in registry:
        for capability in pack.capabilities:
            built = builder.build(pack, capability, "FAST", request)
            assert built.workflow
            assert built.output_nodes
            WorkflowValidator.validate_built(built.workflow)


def test_workflow_builder_applies_preset_then_explicit_request_values() -> None:
    pack = _registry().get("flux-t2i-v1")
    assert pack is not None
    builder = WorkflowBuilder(WORKFLOWS_DIRECTORY)
    built = builder.build(
        pack,
        "TEXT_TO_IMAGE",
        "FAST",
        {
            "prompt": "Explicit values win",
            "steps": 17,
            "width": 768,
            "height": 512,
            "guidance_scale": 4.25,
        },
    )
    template = builder.load(pack.workflow_for("TEXT_TO_IMAGE") or "")

    def bound_value(name: str) -> Any:
        binding = template["bindings"][name]
        return built.workflow[str(binding["node"])]["inputs"][str(binding["field"])]

    assert bound_value("steps") == 17
    assert bound_value("width") == 768
    assert bound_value("height") == 512
    assert bound_value("cfg") == pytest.approx(4.25)


def _fallback_execution_plan() -> ExecutionPlan:
    return ExecutionPlan(
        strategy="MODEL_CPU_OFFLOAD",
        feasible=True,
        dtype="BF16",
        quantization=None,
        attention="efficient",
        vae_tiling=True,
        vae_slicing=True,
        model_cpu_offload=True,
        sequential_cpu_offload=False,
        component_placement={
            "transformer": "gpu_temporary",
            "text_encoder": "cpu_offload",
            "vae": "cpu_offload",
        },
        resolution={"width": 640, "height": 384},
        frames=33,
        fps=12,
        batch=1,
        weights_memory_bytes=8 * GIB,
        runtime_memory_bytes=2 * GIB,
        latent_memory_bytes=1 * GIB,
        reserved_memory_bytes=1 * GIB,
        safety_reserve_bytes=2 * GIB,
        estimated_peak_vram_bytes=8 * GIB,
        vram_total_bytes=16 * GIB,
        vram_free_bytes=14 * GIB,
        ram_required_bytes=8 * GIB,
        scratch_required_bytes=0,
        fallbacks=[
            "EFFICIENT_ATTENTION",
            "VAE_SLICING",
            "VAE_TILING",
            "MODEL_CPU_OFFLOAD",
            "RESOLUTION_REDUCED",
            "FRAMES_REDUCED",
        ],
        reason="Simulated fallback plan",
    )


def test_comfy_model_materializer_maps_snapshot_files_and_plan_into_sent_workflow(
    tmp_path: Path,
) -> None:
    pack = _registry().get("wan22-t2v-v1")
    assert pack is not None
    snapshot = tmp_path / "models" / "wan" / "revision"
    expected_sources: list[Path] = []
    for component, filename in (
        ("transformer", "diffusion_pytorch_model.safetensors"),
        ("text_encoder", "model.safetensors"),
        ("vae", "diffusion_pytorch_model.safetensors"),
    ):
        path = snapshot / component / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(component.encode("utf-8"))
        expected_sources.append(path.resolve())
    materializer = ComfyModelMaterializer(tmp_path / "models")
    materialized = materializer.materialize(
        snapshot=snapshot,
        model_id="wan-test",
        pack=pack,
    )
    assert materialized.components["checkpoint"].endswith(".safetensors")
    assert materialized.components["vae"].endswith(".safetensors")
    assert len(materialized.components["text_encoders"]) == 1
    assert {Path(record["source"]) for record in materialized.links} == set(
        expected_sources
    )
    for record in materialized.links:
        link = tmp_path / "models" / record["category"] / record["name"]
        assert link.is_symlink()
        assert link.resolve() == Path(record["source"])

    plan = _fallback_execution_plan()
    built = WorkflowBuilder(WORKFLOWS_DIRECTORY).build(
        pack,
        "TEXT_TO_VIDEO",
        "QUALITY",
        {"prompt": "fallback workflow", "width": 1024, "height": 576, "frames": 97},
        execution_plan=plan,
        component_values=materialized.components,
    )
    captured: dict[str, Any] = {}

    def opener(request: Any, *, timeout: float) -> _JsonResponse:
        del timeout
        method, path, body = _request_details(request)
        if (method, path) == ("POST", "/prompt"):
            captured.update(body["prompt"])
            return _JsonResponse({"prompt_id": "plan-prompt"})
        if (method, path) == ("GET", "/history/plan-prompt"):
            return _JsonResponse(
                {"plan-prompt": {"outputs": {"8": {"files": [{"filename": "x.mp4"}]}}}}
            )
        raise AssertionError(f"Unexpected request: {method} {path}")

    ComfyUIEngine(
        "http://comfy.invalid",
        poll_interval_seconds=0,
        opener=opener,
    ).execute({"workflow": built.workflow})

    assert captured["1"]["inputs"]["unet_name"] == materialized.components["checkpoint"]
    assert captured["2"]["inputs"]["clip_name"] == materialized.components["text_encoders"][0]
    assert captured["3"]["inputs"]["vae_name"] == materialized.components["vae"]
    assert captured["5"]["inputs"]["width"] == 640
    assert captured["5"]["inputs"]["height"] == 384
    assert captured["5"]["inputs"]["length"] == 33
    assert captured["1"]["inputs"]["weight_dtype"] == "bf16"
    assert captured["7"]["class_type"] == "VAEDecodeTiled"
    assert captured["8"]["_meta"] == {
        "vidioai_attention": "efficient",
        "vidioai_vae_tiling": True,
        "vidioai_vae_slicing": True,
        "vidioai_model_cpu_offload": True,
        "vidioai_sequential_cpu_offload": False,
        "vidioai_component_placement": plan.component_placement,
    }

    removed = materializer.remove("wan-test")
    assert len(removed["removed"]) == 3
    assert not any(
        (tmp_path / "models" / record["category"] / record["name"]).exists()
        for record in materialized.links
    )


def test_comfy_model_materializer_rejects_traversal_and_rolls_back(tmp_path: Path) -> None:
    pack = _registry().get("wan22-t2v-v1")
    assert pack is not None
    snapshot = tmp_path / "models" / "snapshot"
    snapshot.mkdir(parents=True)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    unsafe = replace(
        pack,
        components={**pack.components, "checkpoint": "../../outside.safetensors"},
    )
    materializer = ComfyModelMaterializer(tmp_path / "models")

    with pytest.raises(ComfyModelError, match="hors snapshot"):
        materializer.materialize(snapshot=snapshot, model_id="unsafe", pack=unsafe)

    assert not list((tmp_path / "models").glob("*/vidioai-unsafe-*"))


def test_workflow_validator_rejects_a_binding_to_an_unknown_node() -> None:
    invalid = {
        "schema_version": 1,
        "workflow": {"1": {"class_type": "SaveImage", "inputs": {}}},
        "bindings": {"prompt": {"node": "404", "field": "text"}},
    }

    with pytest.raises(WorkflowValidationError, match="Node de binding absent") as error:
        WorkflowValidator.validate_template(invalid)

    assert error.value.code == "NODE_MISSING"


@pytest.mark.parametrize("vram_gib", [12, 16, 24, 48, 80])
def test_hardware_detector_accepts_explicit_simulated_gpu_profiles(vram_gib: int) -> None:
    calls = 0

    def simulated_nvidia() -> list[GpuTelemetry]:
        nonlocal calls
        calls += 1
        return list(_gpu_profile(vram_gib).gpus)

    no_cuda_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )
    detector = HardwareDetector(
        torch_provider=lambda: no_cuda_torch,
        memory_provider=lambda: {
            "cpu_count": 16,
            "ram_total_bytes": 64 * GIB,
            "ram_available_bytes": 60 * GIB,
        },
        nvidia_provider=simulated_nvidia,
    )

    profile = detector.detect()

    assert calls == 1
    assert profile.cuda_available is True
    assert profile.primary_gpu is not None
    assert profile.primary_gpu.vram_total_bytes == vram_gib * GIB
    assert profile.ram_available_bytes == 60 * GIB


@pytest.mark.parametrize(
    ("system", "machine", "apple_silicon"),
    [("Darwin", "arm64", True), ("Linux", "x86_64", False)],
)
def test_hardware_detector_handles_apple_silicon_and_cpu_only_without_probe(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    apple_silicon: bool,
) -> None:
    import app.hardware.detector as detector_module

    def forbidden_nvidia_probe() -> list[GpuTelemetry]:
        raise AssertionError("nvidia-smi must not be probed in a CPU-only test")

    monkeypatch.setattr(detector_module.platform, "system", lambda: system)
    monkeypatch.setattr(detector_module.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        HardwareDetector,
        "_nvidia_smi",
        staticmethod(forbidden_nvidia_probe),
    )
    no_cuda_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
    )

    profile = HardwareDetector(
        torch_provider=lambda: no_cuda_torch,
        memory_provider=lambda: {
            "ram_total_bytes": 32 * GIB,
            "ram_available_bytes": 24 * GIB,
        },
    ).detect()

    assert profile.cuda_available is False
    assert profile.gpus == ()
    assert profile.apple_silicon is apple_silicon
    assert profile.platform == system
    assert profile.architecture == machine


def test_memory_planner_h100_80g_with_64g_ram_keeps_margin_and_offloads() -> None:
    pack = _registry().get("minimax-h3-diffusers-v1")
    assert pack is not None
    hardware = _gpu_profile(80, ram_gib=64, name="Simulated NVIDIA H100 80GB HBM3")

    plan = MemoryPlanner().execution_plan(
        hardware=hardware,
        model_pack=pack,
        weights_memory_bytes=64 * GIB,
        dtype="BF16",
        quantization=None,
        width=768,
        height=512,
        frames=124,
        fps=24,
        ram_available_bytes=64 * GIB,
    )

    assert plan.feasible is True
    assert plan.strategy in {"MODEL_CPU_OFFLOAD", "SEQUENTIAL_CPU_OFFLOAD"}
    assert plan.strategy != "FULL_GPU"
    assert plan.model_cpu_offload or plan.sequential_cpu_offload
    assert plan.estimated_peak_vram_bytes + plan.safety_reserve_bytes <= 80 * GIB
    assert plan.ram_required_bytes <= 60 * GIB

    expected_order = [
        "EFFICIENT_ATTENTION",
        "VAE_SLICING",
        "VAE_TILING",
        plan.strategy,
    ]
    positions = [plan.fallbacks.index(value) for value in expected_order]
    assert positions == sorted(positions)


def test_comfyui_client_health_queue_history_outputs_and_free_are_mocked() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    history_calls = 0

    def opener(request: Any, *, timeout: float) -> _JsonResponse:
        nonlocal history_calls
        assert timeout > 0
        method, path, body = _request_details(request)
        calls.append((method, path, body))
        if (method, path) == ("GET", "/system_stats"):
            return _JsonResponse({"system": {"os": "mock"}})
        if (method, path) == ("POST", "/prompt"):
            assert body["prompt"] == {"1": {"class_type": "Mock", "inputs": {}}}
            return _JsonResponse({"prompt_id": "prompt-1"})
        if (method, path) == ("GET", "/history/prompt-1"):
            history_calls += 1
            if history_calls == 1:
                return _JsonResponse({"prompt-1": {}})
            return _JsonResponse(
                {
                    "prompt-1": {
                        "outputs": {
                            "9": {
                                "images": [
                                    {
                                        "filename": "result.png",
                                        "subfolder": "",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        if (method, path) == ("GET", "/queue"):
            return _JsonResponse(
                {"queue_running": [[0, "prompt-1", {}]], "queue_pending": []}
            )
        if (method, path) == ("POST", "/free"):
            assert body == {"unload_models": True, "free_memory": True}
            return _JsonResponse({})
        raise AssertionError(f"Unexpected ComfyUI request: {method} {path}")

    progress: list[int] = []
    engine = ComfyUIEngine(
        "http://comfy.invalid",
        poll_interval_seconds=0,
        execution_timeout_seconds=1,
        opener=opener,
    )

    assert engine.health()["ready"] is True
    result = engine.execute(
        {"workflow": {"1": {"class_type": "Mock", "inputs": {}}}},
        progress=progress.append,
    )
    outputs = engine.outputs("prompt-1")
    freed = engine.free()

    assert result["prompt_id"] == "prompt-1"
    assert result["outputs"]["9"]["images"][0]["filename"] == "result.png"
    assert outputs == result["outputs"]
    assert progress[0] == 1
    assert 50 in progress
    assert progress[-1] == 100
    assert freed == {"success": True, "engine": "comfyui"}
    assert ("GET", "/queue") in {(method, path) for method, path, _ in calls}


def test_comfyui_client_surfaces_execution_error() -> None:
    def opener(request: Any, *, timeout: float) -> _JsonResponse:
        del timeout
        method, path, _body = _request_details(request)
        if (method, path) == ("POST", "/prompt"):
            return _JsonResponse({"prompt_id": "failed-prompt"})
        if (method, path) == ("GET", "/history/failed-prompt"):
            return _JsonResponse(
                {
                    "failed-prompt": {
                        "status": {
                            "status_str": "error",
                            "completed": False,
                            "messages": ["mock node failed"],
                        }
                    }
                }
            )
        raise AssertionError(f"Unexpected ComfyUI request: {method} {path}")

    engine = ComfyUIEngine(
        "http://comfy.invalid",
        poll_interval_seconds=0,
        opener=opener,
    )

    with pytest.raises(EngineError, match="mock node failed") as error:
        engine.execute({"workflow": {"1": {"class_type": "Mock", "inputs": {}}}})

    assert error.value.code == "COMFYUI_EXECUTION_FAILED"
    assert error.value.retryable is False


def test_comfyui_queue_history_view_materializes_atomic_image(
    tmp_path: Path,
) -> None:
    from PIL import Image

    encoded = io.BytesIO()
    Image.new("RGB", (96, 64), (12, 34, 56)).save(encoded, format="PNG")
    history_calls = 0

    def opener(request: Any, *, timeout: float) -> _JsonResponse:
        nonlocal history_calls
        assert timeout > 0
        method, path, body = _request_details(request)
        if (method, path) == ("POST", "/prompt"):
            assert body["prompt"]
            return _JsonResponse({"prompt_id": "materialize-prompt"})
        if (method, path) == ("GET", "/history/materialize-prompt"):
            history_calls += 1
            if history_calls == 1:
                return _JsonResponse({"materialize-prompt": {}})
            return _JsonResponse(
                {
                    "materialize-prompt": {
                        "outputs": {
                            "6": {
                                "images": [
                                    {
                                        "filename": "result.png",
                                        "subfolder": "vidioai",
                                        "type": "output",
                                    }
                                ]
                            }
                        }
                    }
                }
            )
        if (method, path) == ("GET", "/queue"):
            return _JsonResponse(
                {"queue_running": [[0, "materialize-prompt", {}]]}
            )
        if (method, path) == ("GET", "/view"):
            query = urlsplit(request.full_url).query
            assert "filename=result.png" in query
            assert "subfolder=vidioai" in query
            return _BytesResponse(encoded.getvalue())
        raise AssertionError(f"Unexpected ComfyUI request: {method} {path}")

    engine = ComfyUIEngine(
        "http://comfy.invalid",
        poll_interval_seconds=0,
        execution_timeout_seconds=1,
        opener=opener,
    )
    result = engine.execute(
        {"workflow": {"1": {"class_type": "Mock", "inputs": {}}}}
    )
    manager = object.__new__(RuntimeManager)
    manager.settings = SimpleNamespace(outputs_dir=tmp_path)
    manager._comfyui = engine

    path, probe, descriptor = manager._materialize_comfy_output(
        outputs=result["outputs"],
        output_relative_path="generations/comfy-result.png",
        video=False,
    )

    assert path == tmp_path / "generations/comfy-result.png"
    assert path.is_file()
    assert probe == {"width": 96, "height": 64}
    assert descriptor["filename"] == "result.png"
    assert RuntimeManager._sha256(path)
    assert list(path.parent.glob("*.tmp.png")) == []


def test_comfyui_client_timeout_cancels_queued_execution() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def opener(request: Any, *, timeout: float) -> _JsonResponse:
        del timeout
        method, path, body = _request_details(request)
        calls.append((method, path, body))
        if (method, path) == ("POST", "/prompt"):
            return _JsonResponse({"prompt_id": "slow-prompt"})
        if (method, path) in {("POST", "/queue"), ("POST", "/interrupt")}:
            return _JsonResponse({})
        raise AssertionError(f"Unexpected ComfyUI request: {method} {path}")

    engine = ComfyUIEngine(
        "http://comfy.invalid",
        execution_timeout_seconds=0,
        opener=opener,
    )

    with pytest.raises(EngineError, match="Délai") as error:
        engine.execute({"workflow": {"1": {"class_type": "Mock", "inputs": {}}}})

    assert error.value.code == "COMFYUI_EXECUTION_TIMEOUT"
    assert error.value.retryable is True
    assert ("POST", "/queue", {"delete": ["slow-prompt"]}) in calls
    assert ("POST", "/interrupt", {}) in calls


def test_preflight_ready_path_is_atomic_and_uses_built_workflow(tmp_path: Path) -> None:
    pack = _registry().get("flux-t2i-v1")
    assert pack is not None
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for component in (
        pack.components.get("checkpoint"),
        pack.components.get("vae"),
        *(pack.components.get("text_encoders") or []),
    ):
        assert isinstance(component, str)
        (snapshot / component).mkdir(parents=True)

    request = {
        "prompt": "A preflight test",
        "quality": "FAST",
        "width": 512,
        "height": 512,
        "frames": 1,
        "seed": 7,
        "output_relative_path": "generations/preflight.png",
    }
    plan = MemoryPlanner().execution_plan(
        hardware=_gpu_profile(24, ram_gib=64),
        model_pack=pack,
        weights_memory_bytes=4 * GIB,
        dtype="BF16",
        quantization=None,
        width=512,
        height=512,
        frames=1,
        fps=None,
    )

    result = PreflightService(WorkflowBuilder(WORKFLOWS_DIRECTORY)).run(
        model_id="flux-test",
        pack=pack,
        capability="TEXT_TO_IMAGE",
        request=request,
        snapshot=snapshot,
        execution_plan=plan,
        engine_health=lambda: {"ready": True, "engine": "comfyui-mock"},
        dependency_errors=[],
        diagnostics={"source": "unit-test"},
    )

    assert result.ready is True
    assert result.status == "READY_TO_RUN"
    assert result.errors == []
    assert result.model_pack_id == "flux-t2i-v1"
    assert result.built_workflow is not None
    assert result.built_workflow.workflow
    assert all(check.ok for check in result.checks)
    assert result.as_dict()["diagnostics"]["engine_health"]["ready"] is True


def test_unload_all_with_no_loaded_model_still_runs_cleanup() -> None:
    manager = object.__new__(RuntimeManager)
    manager._lock = threading.RLock()
    manager._loaded = {}
    manager._comfyui = None
    resource_calls = 0

    def resources() -> dict[str, Any]:
        nonlocal resource_calls
        resource_calls += 1
        return {
            "gpu": {"available": False},
            "memory": {"ram_available_bytes": 32 * GIB},
            "diagnostics": {"gpu_memory_occupied": False},
        }

    manager.resources = resources
    manager._imports = lambda: SimpleNamespace(
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    result = manager.unload_all()

    assert result["success"] is True
    assert result["models_unloaded"] == []
    assert result["unloaded"] == []
    assert resource_calls == 2
    assert result["before_memory"]["gpu"] == {"available": False}
    assert result["after_memory"]["memory"]["ram_available_bytes"] == 32 * GIB
    assert "nettoyage runtime" in result["message"]


def test_unload_all_reports_comfy_free_failure_with_memory_diagnostics() -> None:
    manager = object.__new__(RuntimeManager)
    manager._lock = threading.RLock()
    manager._loaded = {}

    class BrokenComfy:
        def free(self) -> None:
            raise EngineError("mock free failure", code="COMFYUI_UNAVAILABLE")

    manager._comfyui = BrokenComfy()
    calls = 0

    def resources() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"gpu": None, "memory": {"call": calls}, "diagnostics": {}}

    manager.resources = resources
    manager._imports = lambda: SimpleNamespace(
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    result = manager.unload_all()

    assert result["success"] is False
    assert result["code"] == "COMFYUI_FREE_FAILED"
    assert "mock free failure" in result["message"]
    assert result["before_memory"]["memory"] == {"call": 1}
    assert result["after_memory"]["memory"] == {"call": 2}
