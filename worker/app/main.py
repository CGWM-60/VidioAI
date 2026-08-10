"""API interne du worker GPU VidioAI."""

from __future__ import annotations

import asyncio
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from .config import Settings
from .runtime import RuntimeManager, WorkerError
from .schemas import (
    CancelRequest,
    GenerateImageRequest,
    InstallModelRequest,
    ModelRequest,
    UnsupportedGenerationRequest,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    worker_settings = settings or Settings.from_env()
    manager = RuntimeManager(worker_settings)
    application = FastAPI(
        title="VidioAI GPU Worker",
        version="0.1.0",
        docs_url=None if worker_settings.app_env == "GPU_PRODUCTION" else "/docs",
        redoc_url=None,
    )
    application.state.manager = manager
    application.state.settings = worker_settings

    async def authorize(
        token: str | None = Header(
            default=None, alias="X-VidioAI-Worker-Token"
        ),
    ) -> None:
        expected = worker_settings.worker_token
        if expected is not None and (
            token is None or not secrets.compare_digest(token, expected)
        ):
            raise HTTPException(status_code=401, detail="Worker token invalide.")

    @application.exception_handler(WorkerError)
    async def worker_error_handler(_request, error: WorkerError):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=error.status_code,
            content={"error": str(error)},
        )

    @application.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "service": "vidioai-gpu-worker"}

    @application.get("/ready")
    async def ready(
        response: Response, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = manager.runtime_status()
        if not payload["ready"]:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return payload

    @application.get("/v1/resources")
    async def resources(_auth: None = Depends(authorize)) -> dict[str, object]:
        return manager.resources()

    @application.get("/v1/capabilities")
    async def capabilities(_auth: None = Depends(authorize)) -> dict[str, object]:
        return {
            "supported": ["TEXT_TO_IMAGE"],
            "unsupported": [
                "IMAGE_TO_IMAGE",
                "TEXT_TO_VIDEO",
                "IMAGE_TO_VIDEO",
                "VIDEO_TO_VIDEO",
            ],
            "engine_type": "ai",
        }

    @application.post("/v1/models/install")
    async def install_model(
        request: InstallModelRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            manager.install_model,
            request.model_id,
            request.repository,
            request.revision,
            request.capabilities,
        )

    @application.post("/v1/models/load")
    async def load_model(
        request: ModelRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return await asyncio.to_thread(manager.load_model, request.model_id)

    @application.post("/v1/models/unload")
    async def unload_model(
        request: ModelRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return await asyncio.to_thread(manager.unload_model, request.model_id)

    @application.post("/v1/models/unload-all")
    async def unload_all(
        _auth: None = Depends(authorize),
    ) -> dict[str, object]:
        return await asyncio.to_thread(manager.unload_all)

    @application.get("/v1/models/{model_id}/status")
    async def model_status(
        model_id: str, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return manager.model_status(model_id)

    @application.post("/v1/generate/text-to-image")
    async def text_to_image(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        result = await asyncio.to_thread(manager.generate_image, request.model_dump())
        if result["state"] == "FAILED":
            raise WorkerError(str(result["error"]), 500)
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def unsupported(
        _request: UnsupportedGenerationRequest,
        _auth: None = Depends(authorize),
    ) -> None:
        raise HTTPException(
            status_code=501,
            detail="Cette capacité ne possède pas encore de runtime validé.",
        )

    application.post("/v1/generate/image-to-image")(unsupported)
    application.post("/v1/generate/text-to-video")(unsupported)
    application.post("/v1/generate/image-to-video")(unsupported)
    application.post("/v1/generate/video-to-video")(unsupported)

    @application.post("/v1/jobs/cancel")
    async def cancel_job(
        request: CancelRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return manager.cancel_job(request.job_id)

    @application.get("/v1/jobs/{job_id}")
    async def job_status(
        job_id: str, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return manager.job_status(job_id)

    return application


app = create_app()
