from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

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


def test_unsupported_video_capability_is_explicit(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    response = client.post(
        "/v1/generate/text-to-video",
        headers={"X-VidioAI-Worker-Token": "test-token"},
        json={
            "job_id": "12345678",
            "model_id": "stable-image-core",
            "payload": {},
        },
    )
    assert response.status_code == 501


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
        "main",
        ["TEXT_TO_IMAGE"],
    )
    assert status["state"] == "INSTALLED"
    assert status["weights_valid"] is True
    assert status["validation_test"] is False
