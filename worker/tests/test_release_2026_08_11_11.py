from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.adapters.generic_diffusers import GenericDiffusersAdapter
from app.model_profile import ModelRuntimeProfile
from app.normalizers import OutputNormalizer
from app.temporal_output_planner import TemporalOutputPlanner


def test_profile_uses_real_pipeline_defaults_instead_of_arbitrary_values() -> None:
    class Pipeline:
        def __call__(self, prompt, num_inference_steps=17, guidance_scale=1.25):
            del prompt, num_inference_steps, guidance_scale

    profile = ModelRuntimeProfile.from_metadata({}, Pipeline())
    assert profile.steps == 17
    assert profile.guidance_scale == 1.25


def test_generic_i2i_preserves_exact_prompt_and_source_dimensions() -> None:
    observed = {}
    source = object()

    class Pipeline:
        def __call__(
            self,
            prompt,
            image,
            num_inference_steps=12,
            guidance_scale=1.5,
        ):
            observed.update(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            return SimpleNamespace(images=["ok"])

    result = GenericDiffusersAdapter().generate(
        Pipeline(),
        {"metadata": {}, "capability": "IMAGE_TO_IMAGE", "generator": None},
        {
            "prompt": "EXACT PROMPT 123",
            "capability": "IMAGE_TO_IMAGE",
            "input_image": source,
        },
    )

    assert result["images"] == ["ok"]
    assert observed["prompt"] == "EXACT PROMPT 123"
    assert observed["image"] is source
    assert observed["num_inference_steps"] == 12
    assert observed["guidance_scale"] == 1.5


def test_temporal_output_plan_14_native_to_4_seconds_24fps() -> None:
    plan = TemporalOutputPlanner().plan(
        native_frames=14,
        requested_duration_seconds=4,
        requested_fps=24,
    )
    assert plan.delivery_frames == 96
    assert plan.delivery_fps == 24
    assert plan.target_duration_seconds == 4
    assert plan.strategy == "MOTION_INTERPOLATION"


def test_temporal_output_plan_25_native_to_6_seconds_24fps() -> None:
    plan = TemporalOutputPlanner().plan(
        native_frames=25,
        requested_duration_seconds=6,
        requested_fps=24,
    )
    assert plan.delivery_frames == 144
    assert plan.target_duration_seconds == 6


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe absent",
)
def test_real_mp4_respects_4_seconds_24fps(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    frames = []
    for index in range(14):
        image = Image.new("RGB", (96, 64), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (5 + index * 4, 20, 20 + index * 4, 40),
            fill="black",
        )
        frames.append(image)

    output = tmp_path / "result.mp4"
    probe = OutputNormalizer(tmp_path).write_video(
        frames,
        output,
        24,
        duration_seconds=4,
    )

    assert output.is_file()
    assert probe["frames"] == 96
    assert probe["fps"] == pytest.approx(24.0, abs=0.05)
    assert probe["duration"] == pytest.approx(4.0, abs=0.1)
    assert probe["native_frames"] == 14
