from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.adapters.image_to_video import ImageToVideoAdapter
from app.adapters.registry import PipelineRegistry
from app.adapters.inspectors import inspect_model_metadata
from app.config import Settings
from app.main import create_app
from app.runtime import RuntimeManager, WorkerError


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
        "TEXT_TO_VIDEO",
        "IMAGE_TO_VIDEO",
        "VIDEO_TO_VIDEO",
    }


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
    assert adapter.capabilities() == ["IMAGE_TO_VIDEO"]


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
    assert profile["supported_image_roles"] == ["start_frame", "end_frame"]
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
