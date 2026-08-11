"""API interne du worker GPU VidioAI."""

from __future__ import annotations

import asyncio
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from .config import Settings
from .runtime import RuntimeManager, WorkerError
from .schemas import (
    CancelRequest,
    CompatibilityRequest,
    GenerateImageRequest,
    GenerateVideoRequest,
    InstallModelRequest,
    ModelRequest,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    worker_settings = settings or Settings.from_env()
    manager = RuntimeManager(worker_settings)
    manager.log_runtime_versions()
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
            content={
                "error": str(error),
                "code": error.code,
                "retryable": error.retryable,
            },
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
            "supported": [
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
            ],
            "unsupported": [],
            "engine_type": "ai",
        }

    @application.post("/v1/models/compatibility")
    async def model_compatibility(
        request: CompatibilityRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        return manager.check_compatibility(request.model_dump())

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
        payload = request.model_dump()
        payload["capability"] = payload.get("capability") or "TEXT_TO_IMAGE"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def image_to_image(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = payload.get("capability") or "IMAGE_TO_IMAGE"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def text_to_video(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = payload.get("capability") or "TEXT_TO_VIDEO"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def image_to_video(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = payload.get("capability") or "IMAGE_TO_VIDEO"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def video_to_video(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = payload.get("capability") or "VIDEO_TO_VIDEO"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def inpainting(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "INPAINTING"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def outpainting(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "OUTPAINTING"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def image_variation(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "IMAGE_VARIATION"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def image_upscale(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "IMAGE_UPSCALE"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def controlled_image_generation(
        request: GenerateImageRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "CONTROLLED_IMAGE_GENERATION"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def multi_image_to_video(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "MULTI_IMAGE_TO_VIDEO"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def start_end_image_to_video(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "START_END_IMAGE_TO_VIDEO"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def keyframes_to_video(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "KEYFRAMES_TO_VIDEO"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def video_inpainting(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "VIDEO_INPAINTING"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    async def video_upscale(
        request: GenerateVideoRequest, _auth: None = Depends(authorize)
    ) -> dict[str, object]:
        payload = request.model_dump()
        payload["capability"] = "VIDEO_UPSCALE"
        result = await asyncio.to_thread(manager.generate_image, payload)
        if result["state"] == "FAILED":
            raise WorkerError(
                str(result["error"]),
                500,
                code=str(result.get("error_code") or "GENERATION_FAILED"),
            )
        if result["state"] == "CANCELLED":
            raise WorkerError("La génération a été annulée.", 409)
        return result

    application.post("/v1/generate/image-to-image")(image_to_image)
    application.post("/v1/generate/inpainting")(inpainting)
    application.post("/v1/generate/outpainting")(outpainting)
    application.post("/v1/generate/image-variation")(image_variation)
    application.post("/v1/generate/image-upscale")(image_upscale)
    application.post("/v1/generate/controlled-image-generation")(controlled_image_generation)
    application.post("/v1/generate/text-to-video")(text_to_video)
    application.post("/v1/generate/image-to-video")(image_to_video)
    application.post("/v1/generate/multi-image-to-video")(multi_image_to_video)
    application.post("/v1/generate/start-end-image-to-video")(start_end_image_to_video)
    application.post("/v1/generate/keyframes-to-video")(keyframes_to_video)
    application.post("/v1/generate/video-to-video")(video_to_video)
    application.post("/v1/generate/video-inpainting")(video_inpainting)
    application.post("/v1/generate/video-upscale")(video_upscale)

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
