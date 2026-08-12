from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.adapters.minimax_h3 import MiniMaxH3Adapter
from app.audio_output import AudioOutputError, NativeAudioMuxer
from app.pipeline_resolver import PipelineResolver


def h3_metadata() -> dict:
    return {
        "library_name": "diffusers",
        "is_modular": True,
        "class_name": "MiniMaxH3ModularPipeline",
        "modular_model_index": {
            "_class_name": "MiniMaxH3ModularPipeline",
        },
        "architectures": ["MiniMaxH3ModularPipeline"],
        "config": {},
    }


def test_h3_adapter_is_architecture_based_not_repo_based() -> None:
    adapter = MiniMaxH3Adapter()
    metadata = h3_metadata()
    assert adapter.supports_model(metadata)
    metadata["repository"] = "somebody/completely-different-name"
    assert adapter.supports_model(metadata)


def test_h3_workflow_mapping() -> None:
    adapter = MiniMaxH3Adapter()
    assert adapter.workflow_for_capability("TEXT_TO_VIDEO") == "t2va"
    assert adapter.workflow_for_capability("IMAGE_TO_VIDEO") == "fl2va"
    assert (
        adapter.workflow_for_capability("START_END_IMAGE_TO_VIDEO")
        == "fl2va"
    )
    assert (
        adapter.workflow_for_capability("MULTI_IMAGE_TO_VIDEO")
        == "ref2va"
    )


def test_h3_frame_alignment_and_duration_contract() -> None:
    assert MiniMaxH3Adapter._aligned_frames(5.0) == 124
    six_seconds = MiniMaxH3Adapter._aligned_frames(6.0)
    assert six_seconds is not None
    assert six_seconds % 17 == 5
    assert six_seconds / 24 <= 15
    with pytest.raises(Exception):
        MiniMaxH3Adapter._aligned_frames(4.0)
    with pytest.raises(Exception):
        MiniMaxH3Adapter._aligned_frames(16.0)



def test_h3_precision_choice_uses_measured_component_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sizes = {
        "transformer": 62 * 1024**3,
        "text_encoder": 62 * 1024**3,
        "vae": 4 * 1024**3,
        "audio_vae": 1 * 1024**3,
        "scheduler": 0,
        "audio_scheduler": 0,
    }

    monkeypatch.setattr(
        MiniMaxH3Adapter,
        "_dir_bytes",
        staticmethod(lambda path: sizes.get(Path(path).name, 0)),
    )

    # 80 GiB: 62 + 12 GiB de réserve tiennent -> BF16.
    assert (
        MiniMaxH3Adapter._should_use_int8(
            tmp_path,
            "t2va",
            80 * 1024**3,
        )
        is False
    )
    # 64 GiB: le plus gros composant + réserve ne tient pas -> INT8.
    assert (
        MiniMaxH3Adapter._should_use_int8(
            tmp_path,
            "t2va",
            64 * 1024**3,
        )
        is True
    )

def test_pipeline_resolver_requires_exact_h3_modular_class() -> None:
    fake_old = SimpleNamespace(ModularPipeline=object)
    resolution = PipelineResolver().resolve_class(
        h3_metadata(),
        diffusers_module=fake_old,
    )
    assert resolution.runtime_supported is False
    assert resolution.class_name == "MiniMaxH3ModularPipeline"
    assert "DIFFUSERS_VERSION_TOO_OLD" in resolution.runtime_reason

    class H3:
        pass

    fake_current = SimpleNamespace(
        ModularPipeline=object,
        MiniMaxH3ModularPipeline=H3,
    )
    resolution = PipelineResolver().resolve_class(
        h3_metadata(),
        diffusers_module=fake_current,
    )
    assert resolution.runtime_supported is True
    assert resolution.pipeline_cls is H3


def test_h3_generate_requests_native_video_and_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MiniMaxH3Adapter()
    observed = {}

    class Pipeline:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            return {
                "videos": [["f1", "f2"]],
                "audio": [np.zeros((2, 32000), dtype=np.float32)],
                "sampling_rate": 32000,
            }

    monkeypatch.setattr(
        adapter,
        "_require_h3_diffusers",
        lambda: {},
    )
    output = adapter.generate(
        Pipeline(),
        {"generator": "seeded-generator"},
        {
            "capability": "TEXT_TO_VIDEO",
            "prompt": "A fox walks through snow.",
            "duration_seconds": 5,
            "requested_duration_seconds": 5,
            "width": 960,
            "height": 544,
        },
    )

    assert observed["prompt"] == "A fox walks through snow."
    assert observed["output"] == [
        "videos",
        "audio",
        "sampling_rate",
    ]
    assert observed["num_frames"] == 124
    assert observed["width"] % 32 == 0
    assert observed["height"] % 32 == 0
    assert "negative_prompt" not in observed
    assert "guidance_scale" not in observed
    assert output["frames"] == [["f1", "f2"]]
    assert output["audio_sample_rate"] == 32000
    assert output["native_audio"] is not None


def test_h3_fl2va_maps_first_and_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MiniMaxH3Adapter()
    observed = {}
    first = object()
    last = object()

    class Pipeline:
        def __call__(self, **kwargs):
            observed.update(kwargs)
            return {
                "videos": [["f"]],
                "audio": [np.zeros((2, 32000), dtype=np.float32)],
                "sampling_rate": 32000,
            }

    monkeypatch.setattr(adapter, "_require_h3_diffusers", lambda: {})
    adapter.generate(
        Pipeline(),
        {"generator": None},
        {
            "capability": "START_END_IMAGE_TO_VIDEO",
            "prompt": "Transition naturally.",
            "duration_seconds": 5,
            "resolved_input_images": [first, last],
            "input_images": [
                {"order": 0, "role": "start_frame"},
                {"order": 1, "role": "end_frame"},
            ],
        },
    )
    assert observed["image"] is first
    assert observed["last_image"] is last


def _create_h264(path: Path, duration: float = 1.0) -> None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=96x64:r=24",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"ffmpeg H264 indisponible: {result.stderr}")


def test_native_audio_mux_produces_h264_plus_aac(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    _create_h264(video)

    sample_rate = 32000
    samples = np.arange(sample_rate, dtype=np.float32)
    tone = 0.05 * np.sin(2 * np.pi * 440.0 * samples / sample_rate)
    stereo = np.stack([tone, tone], axis=0)

    probe = NativeAudioMuxer().mux(
        video_path=video,
        native_audio=[stereo],
        audio_sample_rate=sample_rate,
        duration_seconds=1.0,
    )

    assert probe["actual_audio"] is True
    assert probe["audio_codec"] == "aac"
    assert probe["audio_channels"] == 2
    assert probe["audio_sample_rate"] == sample_rate

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    assert any(
        stream["codec_type"] == "video"
        and stream["codec_name"] == "h264"
        for stream in streams
    )
    assert any(
        stream["codec_type"] == "audio"
        and stream["codec_name"] == "aac"
        for stream in streams
    )


def test_requested_native_audio_never_falls_back_to_silence(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mp4"
    _create_h264(video)
    with pytest.raises(AudioOutputError) as error:
        NativeAudioMuxer().mux(
            video_path=video,
            native_audio=None,
            audio_sample_rate=32000,
            duration_seconds=1.0,
        )
    assert error.value.code == "NATIVE_AUDIO_MISSING"
