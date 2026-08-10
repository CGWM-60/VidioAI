from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.adapters.image_to_image import ImageToImageAdapter
from app.adapters.image_to_video import ImageToVideoAdapter
from app.adapters.text_to_image import TextToImageAdapter
from app.adapters.text_to_video import TextToVideoAdapter
from app.adapters.video_to_video import VideoToVideoAdapter
from app.adapters.registry import PipelineRegistry
from app.adapters.inspectors import inspect_model_metadata
from app.config import Settings
from app.main import create_app
from app.runtime import RuntimeManager, WorkerError


CAPABILITY_ENDPOINTS: dict[str, str] = {
    "TEXT_TO_IMAGE": "/v1/generate/text-to-image",
    "IMAGE_TO_IMAGE": "/v1/generate/image-to-image",
    "INPAINTING": "/v1/generate/inpainting",
    "OUTPAINTING": "/v1/generate/outpainting",
    "IMAGE_VARIATION": "/v1/generate/image-variation",
    "IMAGE_UPSCALE": "/v1/generate/image-upscale",
    "CONTROLLED_IMAGE_GENERATION": "/v1/generate/controlled-image-generation",
    "TEXT_TO_VIDEO": "/v1/generate/text-to-video",
    "IMAGE_TO_VIDEO": "/v1/generate/image-to-video",
    "MULTI_IMAGE_TO_VIDEO": "/v1/generate/multi-image-to-video",
    "START_END_IMAGE_TO_VIDEO": "/v1/generate/start-end-image-to-video",
    "KEYFRAMES_TO_VIDEO": "/v1/generate/keyframes-to-video",
    "VIDEO_TO_VIDEO": "/v1/generate/video-to-video",
    "VIDEO_INPAINTING": "/v1/generate/video-inpainting",
    "VIDEO_UPSCALE": "/v1/generate/video-upscale",
}


def _base_image_payload() -> dict[str, Any]:
    return {
        "job_id": "job-12345678",
        "model_id": "stable-image-core",
        "prompt": "image prompt valid",
        "output_relative_path": "generations/out.png",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 0.0,
    }


def _base_video_payload() -> dict[str, Any]:
    return {
        "job_id": "job-87654321",
        "model_id": "stable-video-core",
        "prompt": "video prompt valid",
        "output_relative_path": "generations/out.mp4",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance_scale": 0.0,
        "duration_seconds": 2,
        "fps": 8,
        "frames": 8,
    }


def settings(tmp_path: Path, *, profile: str = "LOCAL") -> Settings:
    return Settings(
        app_env=profile,
        gpu_required=profile == "GPU_PRODUCTION",
        models_dir=tmp_path / "models",
        work_dir=tmp_path / "work",
        outputs_dir=tmp_path / "outputs",
        hf_home=tmp_path / "hf",
        worker_token="test-token",
        minimum_weights_bytes=1024,
        default_model_id="stable-image-core",
        default_repository="stabilityai/sd-turbo",
    )


def test_health_is_liveness_only(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_gpu_profile_never_claims_ready_without_real_cuda(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path, profile="GPU_PRODUCTION")))
    response = client.get(
        "/ready", headers={"X-VidioAI-Worker-Token": "test-token"}
    )
    payload = response.json()
    assert response.status_code == 503
    assert payload["ready"] is False
    assert payload["gpu_required"] is True
    assert payload["cuda_available"] is False


def test_internal_routes_require_the_worker_token(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    assert client.get("/v1/resources").status_code == 401
    assert (
        client.get(
            "/v1/resources",
            headers={"X-VidioAI-Worker-Token": "test-token"},
        ).status_code
        == 200
    )


def test_capabilities_endpoint_lists_all_modalities(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    response = client.get(
        "/v1/capabilities",
        headers={"X-VidioAI-Worker-Token": "test-token"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["supported"]) == {
        "TEXT_TO_IMAGE",
        "IMAGE_TO_IMAGE",
        "INPAINTING",
        "OUTPAINTING",
        "IMAGE_VARIATION",
        "IMAGE_UPSCALE",
        "CONTROLLED_IMAGE_GENERATION",
        "TEXT_TO_VIDEO",
        "IMAGE_TO_VIDEO",
        "MULTI_IMAGE_TO_VIDEO",
        "START_END_IMAGE_TO_VIDEO",
        "KEYFRAMES_TO_VIDEO",
        "VIDEO_TO_VIDEO",
        "VIDEO_INPAINTING",
        "VIDEO_UPSCALE",
    }


@pytest.mark.parametrize("capability,endpoint", sorted(CAPABILITY_ENDPOINTS.items()))
def test_generation_endpoint_routes_capability_to_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
    endpoint: str,
) -> None:
    app = create_app(settings(tmp_path))
    observed: list[dict[str, Any]] = []

    def fake_generate_image(payload: dict[str, Any]) -> dict[str, Any]:
        observed.append(payload)
        return {
            "state": "COMPLETED",
            "job_id": payload["job_id"],
            "output_relative_path": payload["output_relative_path"],
        }

    monkeypatch.setattr(app.state.manager, "generate_image", fake_generate_image)
    client = TestClient(app)
    payload = _base_video_payload() if "VIDEO" in capability else _base_image_payload()
    response = client.post(
        endpoint,
        json=payload,
        headers={"X-VidioAI-Worker-Token": "test-token"},
    )
    assert response.status_code == 200
    assert observed
    assert observed[-1]["capability"] == capability


@pytest.mark.parametrize("capability,endpoint", sorted(CAPABILITY_ENDPOINTS.items()))
def test_generation_endpoint_schema_validation_for_all_capabilities(
    tmp_path: Path,
    capability: str,
    endpoint: str,
) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    payload = _base_video_payload() if "VIDEO" in capability else _base_image_payload()
    payload["prompt"] = "no"
    response = client.post(
        endpoint,
        json=payload,
        headers={"X-VidioAI-Worker-Token": "test-token"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("capability,endpoint", sorted(CAPABILITY_ENDPOINTS.items()))
def test_generation_endpoint_structured_errors_for_all_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
    endpoint: str,
) -> None:
    app = create_app(settings(tmp_path))

    def failing_generate(_payload: dict[str, Any]) -> dict[str, Any]:
        raise WorkerError("Entrée invalide", 422)

    monkeypatch.setattr(app.state.manager, "generate_image", failing_generate)
    client = TestClient(app)
    payload = _base_video_payload() if "VIDEO" in capability else _base_image_payload()
    response = client.post(
        endpoint,
        json=payload,
        headers={"X-VidioAI-Worker-Token": "test-token"},
    )
    assert response.status_code == 422
    assert response.json() == {"error": "Entrée invalide"}


@pytest.mark.parametrize("capability", sorted(CAPABILITY_ENDPOINTS))
def test_pipeline_registry_selects_adapter_for_each_capability(capability: str) -> None:
    registry = PipelineRegistry()
    metadata = {"capabilities": [capability], "pipeline_tag": "image-to-image"}
    adapter = registry.select_for_capability(metadata, capability)
    assert adapter is not None
    assert capability in adapter.capabilities()


def test_dynamic_metadata_detection_exposes_multiple_capabilities(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionImg2ImgPipeline",
                "pipeline_tag": "image-to-image",
                "config": {"model_type": "stable-diffusion"},
                "tags": ["image-to-image", "diffusers"],
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "stable-diffusion", "architectures": ["UNet2DConditionModel"]}),
        encoding="utf-8",
    )

    metadata = inspect_model_metadata(snapshot)
    assert "TEXT_TO_IMAGE" in metadata["capabilities"]
    assert "IMAGE_TO_IMAGE" in metadata["capabilities"]


def test_pipeline_registry_selects_an_adapter_from_metadata(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "LTXImageToVideoPipeline",
                "pipeline_tag": "image-to-video",
                "tags": ["image-to-video", "diffusers"],
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "config.json").write_text(json.dumps({"model_type": "ltx"}), encoding="utf-8")

    metadata = inspect_model_metadata(snapshot)
    registry = PipelineRegistry()
    adapter = registry.select_for_capability(metadata, "IMAGE_TO_VIDEO")
    assert adapter is not None
    assert "IMAGE_TO_VIDEO" in adapter.capabilities()


def test_i2v_adapter_defaults_to_single_image_for_unknown_models() -> None:
    adapter = ImageToVideoAdapter()
    metadata = {"pipeline_tag": "image-to-video", "class_name": "UnknownImageToVideoPipeline"}
    profile = adapter.input_profile(metadata)
    assert profile["min_input_images"] == 1
    assert profile["max_input_images"] == 1
    assert profile["supported_image_roles"] == []


def test_i2v_adapter_supports_start_and_end_frames_for_ltx_models() -> None:
    adapter = ImageToVideoAdapter()
    metadata = {"pipeline_tag": "image-to-video", "class_name": "LTXImageToVideoPipeline"}
    profile = adapter.input_profile(metadata)
    assert profile["min_input_images"] == 1
    assert profile["max_input_images"] == 2
    assert set(profile["supported_image_roles"]) == {
        "start",
        "end",
        "start_frame",
        "end_frame",
    }
    assert profile["supports_start_end_frames"] is True


def test_i2v_adapter_supports_reference_images_for_multiple_inputs() -> None:
    adapter = ImageToVideoAdapter()
    metadata = {"pipeline_tag": "image-to-video", "class_name": "CogVideoXImageToVideoPipeline"}
    profile = adapter.input_profile(metadata)
    assert profile["min_input_images"] == 1
    assert profile["max_input_images"] == 8
    assert profile["supports_reference_images"] is True
    assert profile["supports_keyframes"] is True


def test_i2v_adapter_preserves_input_order_and_roles() -> None:
    adapter = ImageToVideoAdapter()
    request = {
        "prompt": "demo",
        "input_images": [
            {"asset_id": "a", "order": 0, "role": "start_frame"},
            {"asset_id": "b", "order": 1, "role": "reference"},
        ],
    }
    payload = adapter.prepare_pipeline_inputs(request)
    assert payload["images"] == ["a", "b"]
    assert payload["roles"] == ["start_frame", "reference"]


def test_text_to_image_adapter_filters_kwargs_by_pipeline_signature() -> None:
    adapter = TextToImageAdapter()
    called: dict[str, Any] = {}

    class Pipeline:
        def __call__(self, prompt=None, width=None):
            called.update({"prompt": prompt, "width": width})
            return SimpleNamespace(images=["ok"])

    output = adapter.generate(
        Pipeline(),
        {"generator": object()},
        {
            "prompt": "hello",
            "width": 512,
            "height": 512,
            "negative_prompt": "bad",
            "steps": 6,
            "guidance_scale": 7.0,
        },
    )
    assert output["images"] == ["ok"]
    assert called == {"prompt": "hello", "width": 512}


def test_image_to_image_adapter_filters_kwargs_for_inpainting() -> None:
    adapter = ImageToImageAdapter()
    called: dict[str, Any] = {}

    class Pipeline:
        def __call__(
            self,
            prompt=None,
            image=None,
            mask_image=None,
            num_inference_steps=None,
            generator=None,
        ):
            called.update(
                {
                    "prompt": prompt,
                    "image": image,
                    "mask_image": mask_image,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                }
            )
            return SimpleNamespace(images=["ok"])

    output = adapter.generate(
        Pipeline(),
        {"generator": object()},
        {
            "prompt": "hello",
            "capability": "INPAINTING",
            "input_image": "img",
            "mask_image": "mask",
            "control_image": "control",
            "strength": 0.4,
            "steps": 9,
            "guidance_scale": 7.0,
        },
    )
    assert output["images"] == ["ok"]
    assert called["prompt"] == "hello"
    assert called["image"] == "img"
    assert called["mask_image"] == "mask"
    assert called["num_inference_steps"] == 9


def test_image_to_image_adapter_removes_strength_for_variation() -> None:
    adapter = ImageToImageAdapter()
    called: dict[str, Any] = {}

    class Pipeline:
        def __call__(self, prompt=None, image=None, strength=None):
            called.update({"prompt": prompt, "image": image, "strength": strength})
            return SimpleNamespace(images=["ok"])

    adapter.generate(
        Pipeline(),
        {"generator": object()},
        {
            "prompt": "hello",
            "capability": "IMAGE_VARIATION",
            "input_image": "img",
            "strength": 0.9,
        },
    )
    assert called["prompt"] == "hello"
    assert called["image"] == "img"
    assert called["strength"] is None


def test_text_to_video_adapter_filters_kwargs_by_signature() -> None:
    adapter = TextToVideoAdapter()
    called: dict[str, Any] = {}

    class Pipeline:
        def __call__(self, prompt=None, num_frames=None, fps=None):
            called.update({"prompt": prompt, "num_frames": num_frames, "fps": fps})
            return SimpleNamespace(frames=["f"])

    output = adapter.generate(
        Pipeline(),
        {"generator": object()},
        {
            "prompt": "hello",
            "frames": 8,
            "fps": 12,
            "negative_prompt": "bad",
            "steps": 10,
            "guidance_scale": 5,
        },
    )
    assert output["frames"] == ["f"]
    assert called == {"prompt": "hello", "num_frames": 8, "fps": 12}


def test_image_to_video_adapter_filters_kwargs_and_preserves_roles() -> None:
    adapter = ImageToVideoAdapter()
    called: dict[str, Any] = {}

    class Pipeline:
        def __call__(self, prompt=None, image=None, end_image=None, image_roles=None):
            called.update(
                {
                    "prompt": prompt,
                    "image": image,
                    "end_image": end_image,
                    "image_roles": image_roles,
                }
            )
            return SimpleNamespace(frames=["f"])

    output = adapter.generate(
        Pipeline(),
        {"generator": object()},
        {
            "prompt": "hello",
            "input_images": [
                {"asset_id": "start", "order": 0, "role": "start_frame"},
                {"asset_id": "end", "order": 1, "role": "end_frame"},
            ],
            "steps": 4,
        },
    )
    assert output["frames"] == ["f"]
    assert called["prompt"] == "hello"
    assert called["image"] == "start"
    assert called["end_image"] == "end"
    assert called["image_roles"] == ["start_frame", "end_frame"]


def test_video_to_video_adapter_filters_kwargs_by_signature() -> None:
    adapter = VideoToVideoAdapter()
    called: dict[str, Any] = {}

    class Pipeline:
        def __call__(self, prompt=None, video=None, mask_image=None):
            called.update({"prompt": prompt, "video": video, "mask_image": mask_image})
            return SimpleNamespace(frames=["f"])

    output = adapter.generate(
        Pipeline(),
        {"generator": object()},
        {
            "prompt": "hello",
            "input_video": "vid.mp4",
            "mask_image": "mask",
            "frames": 16,
            "fps": 8,
            "strength": 0.4,
        },
    )
    assert output["frames"] == ["f"]
    assert called == {"prompt": "hello", "video": "vid.mp4", "mask_image": "mask"}


def test_manifest_without_weights_is_rejected(tmp_path: Path) -> None:
    manager = RuntimeManager(settings(tmp_path))
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}),
        encoding="utf-8",
    )
    try:
        manager.validate_snapshot(snapshot)
    except WorkerError as error:
        assert "safetensors" in str(error)
    else:
        raise AssertionError("Un manifest sans poids ne doit jamais être accepté.")


def test_valid_l3_snapshot_is_reused_without_hugging_face_call(tmp_path: Path) -> None:
    configuration = settings(tmp_path)
    manager = RuntimeManager(configuration)
    model_root = configuration.models_dir / "stable-image-core"
    snapshot = model_root / "commit-sha"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"w" * 2048)
    (model_root / "active.json").write_text(
        json.dumps(
            {
                "model_id": "stable-image-core",
                "repository": "stabilityai/sd-turbo",
                "revision": "commit-sha",
            }
        ),
        encoding="utf-8",
    )

    status = manager.install_model(
        "stable-image-core",
        "stabilityai/sd-turbo",
        "commit-sha",
        ["TEXT_TO_IMAGE"],
    )
    assert status["state"] == "INSTALLED"
    assert status["weights_valid"] is True
    assert status["validation_test"] is False


def _write_fake_snapshot(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"}),
        encoding="utf-8",
    )
    (target / "model.safetensors").write_bytes(b"w" * 2048)


def _fake_torch(*, cuda_available: bool) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        version=SimpleNamespace(cuda="12.4" if cuda_available else None),
        __version__="2.9.0",
    )


def test_runtime_status_handles_runtime_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RuntimeManager(settings(tmp_path))

    def failing_imports() -> tuple[object, object]:
        raise WorkerError("Runtime IA indisponible: import error", 503)

    monkeypatch.setattr(manager, "_imports", failing_imports)
    status = manager.runtime_status()
    assert status["ready"] is False
    assert status["runtime_available"] is False
    assert status["cuda_available"] is False
    assert any("import error" in error for error in status["errors"])


def test_imports_contract_is_two_values(tmp_path: Path) -> None:
    manager = RuntimeManager(settings(tmp_path))
    manager._runtime_modules = (_fake_torch(cuda_available=False), (object(), object()))
    torch, hub = manager._imports()
    assert torch is not None
    assert isinstance(hub, tuple)
    assert len(hub) == 2


def test_imports_raises_worker_error_when_runtime_error_is_cached(tmp_path: Path) -> None:
    manager = RuntimeManager(settings(tmp_path))
    manager._runtime_error = "runtime cassé"
    with pytest.raises(WorkerError) as error:
        manager._imports()
    assert error.value.status_code == 503
    assert "runtime cassé" in str(error.value)


def test_runtime_status_cuda_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RuntimeManager(settings(tmp_path))

    fake_torch = _fake_torch(cuda_available=False)
    monkeypatch.setattr(manager, "_imports", lambda: (fake_torch, object()))

    status = manager.runtime_status()
    assert status["runtime_available"] is True
    assert status["cuda_available"] is False
    assert status["torch_version"] == "2.9.0"


def test_runtime_status_cuda_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = RuntimeManager(settings(tmp_path))

    fake_torch = _fake_torch(cuda_available=True)
    monkeypatch.setattr(manager, "_imports", lambda: (fake_torch, object()))

    status = manager.runtime_status()
    assert status["runtime_available"] is True
    assert status["cuda_available"] is True
    assert status["cuda_version"] == "12.4"
    assert status["ready"] is True


def test_ready_endpoint_exposes_runtime_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    application = create_app(settings(tmp_path))

    def failing_imports() -> tuple[object, object]:
        raise WorkerError("Runtime IA indisponible: import error", 503)

    monkeypatch.setattr(application.state.manager, "_imports", failing_imports)
    client = TestClient(application)
    response = client.get(
        "/ready", headers={"X-VidioAI-Worker-Token": "test-token"}
    )
    assert response.status_code == 503
    payload = response.json()
    assert payload["ready"] is False
    assert payload["runtime_available"] is False


@pytest.mark.parametrize("value", ["", "   "])
def test_install_model_never_sends_empty_bearer_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    manager = RuntimeManager(settings(tmp_path))
    observed = {"api_token": "UNSET", "download_token": "UNSET", "download_headers": None}

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            observed["api_token"] = token

        def model_info(self, repository: str, revision: str = "main") -> SimpleNamespace:
            return SimpleNamespace(sha="commit-sha")

    def fake_snapshot_download(**kwargs) -> None:
        observed["download_token"] = kwargs.get("token")
        observed["download_headers"] = kwargs.get("headers")
        _write_fake_snapshot(Path(kwargs["local_dir"]))

    monkeypatch.setattr(
        manager,
        "_imports",
        lambda: (_fake_torch(cuda_available=False), (FakeHfApi, fake_snapshot_download)),
    )
    monkeypatch.setenv("HF_TOKEN", value)

    status = manager.install_model(
        "stable-image-core",
        "stabilityai/sd-turbo",
        "main",
        ["TEXT_TO_IMAGE"],
    )
    assert status["state"] == "INSTALLED"
    assert observed["api_token"] is None
    assert observed["download_token"] is None
    assert observed["download_headers"] is None or "Authorization" not in observed["download_headers"]


def test_install_model_public_repository_works_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RuntimeManager(settings(tmp_path))
    observed = {"api_token": "UNSET", "download_token": "UNSET"}

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            observed["api_token"] = token

        def model_info(self, repository: str, revision: str = "main") -> SimpleNamespace:
            return SimpleNamespace(sha="commit-sha")

    def fake_snapshot_download(**kwargs) -> None:
        observed["download_token"] = kwargs.get("token")
        _write_fake_snapshot(Path(kwargs["local_dir"]))

    monkeypatch.setattr(
        manager,
        "_imports",
        lambda: (_fake_torch(cuda_available=False), (FakeHfApi, fake_snapshot_download)),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    status = manager.install_model(
        "stable-image-core",
        "stabilityai/sd-turbo",
        "main",
        ["TEXT_TO_IMAGE"],
    )
    assert status["state"] == "INSTALLED"
    assert observed["api_token"] is None
    assert observed["download_token"] is None


def test_install_model_public_repository_works_with_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RuntimeManager(settings(tmp_path))
    observed = {"api_token": None, "download_token": None}

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            observed["api_token"] = token

        def model_info(self, repository: str, revision: str = "main") -> SimpleNamespace:
            return SimpleNamespace(sha="commit-sha")

    def fake_snapshot_download(**kwargs) -> None:
        observed["download_token"] = kwargs.get("token")
        _write_fake_snapshot(Path(kwargs["local_dir"]))

    monkeypatch.setattr(
        manager,
        "_imports",
        lambda: (_fake_torch(cuda_available=False), (FakeHfApi, fake_snapshot_download)),
    )
    monkeypatch.setenv("HF_TOKEN", "hf_valid_token")

    status = manager.install_model(
        "stable-image-core",
        "stabilityai/sd-turbo",
        "main",
        ["TEXT_TO_IMAGE"],
    )
    assert status["state"] == "INSTALLED"
    assert observed["api_token"] == "hf_valid_token"
    assert observed["download_token"] == "hf_valid_token"


def test_install_model_gated_without_token_returns_access_required_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RuntimeManager(settings(tmp_path))

    class GatedRepoError(Exception):
        pass

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            del token

        def model_info(self, repository: str, revision: str = "main") -> SimpleNamespace:
            raise GatedRepoError("Repository is gated and requires authentication")

    def fake_snapshot_download(**kwargs) -> None:
        _write_fake_snapshot(Path(kwargs["local_dir"]))

    monkeypatch.setattr(
        manager,
        "_imports",
        lambda: (_fake_torch(cuda_available=False), (FakeHfApi, fake_snapshot_download)),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(WorkerError) as error:
        manager.install_model(
            "stable-image-core",
            "stabilityai/sd-turbo",
            "main",
            ["TEXT_TO_IMAGE"],
        )
    assert error.value.status_code == 403
    assert "Accès Hugging Face requis" in str(error.value)


def test_install_model_private_without_token_returns_access_required_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RuntimeManager(settings(tmp_path))

    class PrivateRepoError(Exception):
        pass

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            del token

        def model_info(self, repository: str, revision: str = "main") -> SimpleNamespace:
            raise PrivateRepoError("private repository: authentication required")

    def fake_snapshot_download(**kwargs) -> None:
        _write_fake_snapshot(Path(kwargs["local_dir"]))

    monkeypatch.setattr(
        manager,
        "_imports",
        lambda: (_fake_torch(cuda_available=False), (FakeHfApi, fake_snapshot_download)),
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(WorkerError) as error:
        manager.install_model(
            "stable-image-core",
            "stabilityai/sd-turbo",
            "main",
            ["TEXT_TO_IMAGE"],
        )
    assert error.value.status_code == 403
    assert "Accès Hugging Face requis" in str(error.value)
