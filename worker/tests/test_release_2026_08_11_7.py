from __future__ import annotations

from app.capability_resolver import CapabilityResolver
from app.dtype_resolver import DTypeResolver


def test_bf16_gpu_support_does_not_force_bf16_without_weight_metadata() -> None:
    plan = DTypeResolver().resolve(
        {"model_index": {"_class_name": "DiffusionPipeline"}},
        cuda_available=True,
        bf16_supported=True,
    )
    assert plan.load_dtype == "float16"
    assert plan.precision == "FP16"


def test_image_and_num_frames_signature_is_i2v_not_t2i() -> None:
    class ImageToVideoPipeline:
        def __call__(self, image, num_frames):
            del image, num_frames

    capabilities = CapabilityResolver().runtime_capabilities(ImageToVideoPipeline)
    assert "IMAGE_TO_VIDEO" in capabilities
    assert "TEXT_TO_IMAGE" not in capabilities


def test_prompt_image_and_num_frames_signature_exposes_t2v_and_i2v() -> None:
    class DualVideoPipeline:
        def __call__(self, prompt, image, num_frames):
            del prompt, image, num_frames

    capabilities = CapabilityResolver().runtime_capabilities(DualVideoPipeline)
    assert "TEXT_TO_VIDEO" in capabilities
    assert "IMAGE_TO_VIDEO" in capabilities
