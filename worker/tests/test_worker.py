from __future__ import annotations

import json
from pathlib import Path

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
