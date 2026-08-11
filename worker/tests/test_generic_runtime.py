from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.adapters.generic_diffusers import GenericDiffusersAdapter
from app.adapters.inspectors import inspect_model_metadata
from app.adapters.registry import PipelineRegistry
from app.capability_resolver import CapabilityResolver
from app.model_profile import ModelRuntimeProfile
from app.normalizers import InputNormalizer, NormalizationError, OutputNormalizer
from app.pipeline_resolver import PipelineResolutionError, PipelineResolver
from app.resolution_resolver import ResolutionResolver
from app.config import Settings
from app.runtime import LoadedModel, RuntimeImports, RuntimeManager


def metadata(class_name: str = "FutureVideoPipeline") -> dict[str, object]:
    return {
        "library_name": "diffusers",
        "class_name": class_name,
        "model_index": {"_class_name": class_name},
        "config": {},
        "architectures": [],
        "raw_tags": [],
        "base_models": [],
    }


def test_pipeline_resolver_accepts_any_installed_diffusers_pipeline_class() -> None:
    class FutureVideoPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    result = PipelineResolver().resolve_class(
        metadata(), diffusers_module=SimpleNamespace(FutureVideoPipeline=FutureVideoPipeline)
    )
    assert result.runtime_supported is True
    assert result.class_name == "FutureVideoPipeline"
    assert result.strategy == "exact-class"


def test_real_public_metadata_resolves_multiple_diffusers_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures = Path(__file__).parent / "fixtures" / "public_diffusers"

    class TextVideoPipeline:
        def __call__(self, prompt, num_frames, height=None, width=None):
            del prompt, num_frames, height, width

    class ImageVideoPipeline:
        def __call__(self, prompt, image, num_frames, height=None, width=None):
            del prompt, image, num_frames, height, width

    class TextImagePipeline:
        def __call__(self, prompt, height=None, width=None):
            del prompt, height, width

    classes = {
        "StableDiffusionPipeline": TextImagePipeline,
        "StableDiffusionXLPipeline": TextImagePipeline,
        "FluxPipeline": TextImagePipeline,
        "WanPipeline": TextVideoPipeline,
        "CogVideoXPipeline": TextVideoPipeline,
        "LTXPipeline": ImageVideoPipeline,
        "HunyuanVideoPipeline": TextVideoPipeline,
    }
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(**classes))

    expected = {
        "stable_diffusion_v1_5": "TEXT_TO_IMAGE",
        "stable_diffusion_xl_base": "TEXT_TO_IMAGE",
        "flux_1_schnell": "TEXT_TO_IMAGE",
        "wan_2_2_ti2v": "TEXT_TO_VIDEO",
        "cogvideox_2b": "TEXT_TO_VIDEO",
        "ltx_video": "IMAGE_TO_VIDEO",
        "hunyuan_video": "TEXT_TO_VIDEO",
    }
    for family, capability in expected.items():
        root = fixtures / family
        hub = json.loads((root / "hub.json").read_text(encoding="utf-8"))
        detected = inspect_model_metadata(root)
        detected.update(
            pipeline_tag=hub.get("pipeline_tag"),
            library_name=hub.get("library_name"),
            raw_tags=hub.get("tags") or [],
            base_models=hub.get("base_models") or [],
        )
        resolution = PipelineResolver().resolve_class(detected)
        assert resolution.runtime_supported is True, hub["id"]
        assert resolution.class_name == detected["model_index"]["_class_name"]
        capabilities = CapabilityResolver().resolve(detected, resolution.pipeline_cls)
        assert capability in capabilities, hub["id"]
        adapter = PipelineRegistry().select_for_capability(detected, capability)
        assert isinstance(adapter, GenericDiffusersAdapter), hub["id"]


def test_pipeline_resolver_falls_back_to_diffusion_pipeline() -> None:
    class ExactPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise ValueError("configuration class mismatch")

    sentinel = object()

    class DiffusionPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return sentinel

    pipeline, result = PipelineResolver().load(
        "/snapshot",
        metadata("ExactPipeline"),
        "TEXT_TO_VIDEO",
        {"torch_dtype": None},
        diffusers_module=SimpleNamespace(
            ExactPipeline=ExactPipeline,
            DiffusionPipeline=DiffusionPipeline,
        ),
    )
    assert pipeline is sentinel
    assert result.strategy == "DiffusionPipeline"
    assert [attempt["result"] for attempt in result.attempted] == ["FAILED", "SUCCESS"]


def test_pipeline_resolver_switches_to_capability_specific_auto_pipeline() -> None:
    class TextPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, prompt, num_frames):
            del prompt, num_frames

    class ImagePipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, prompt, image, num_frames):
            del prompt, image, num_frames

    pipeline, result = PipelineResolver().load(
        "/snapshot",
        metadata("TextPipeline"),
        "IMAGE_TO_VIDEO",
        {},
        diffusers_module=SimpleNamespace(
            TextPipeline=TextPipeline,
            AutoPipelineForImage2Video=ImagePipeline,
        ),
    )
    assert isinstance(pipeline, ImagePipeline)
    assert result.strategy == "AutoPipelineForImage2Video"


def test_pipeline_resolver_classifies_missing_dependency() -> None:
    class ExactPipeline:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            error = ModuleNotFoundError("No module named 'sentencepiece'")
            error.name = "sentencepiece"
            raise error

    with pytest.raises(PipelineResolutionError) as caught:
        PipelineResolver().load(
            "/snapshot",
            metadata("ExactPipeline"),
            "TEXT_TO_IMAGE",
            {},
            diffusers_module=SimpleNamespace(ExactPipeline=ExactPipeline),
        )
    assert caught.value.code == "MISSING_DEPENDENCY"
    assert caught.value.dependency == "sentencepiece"


def test_pipeline_resolver_never_enables_remote_code() -> None:
    remote = metadata()
    remote["config"] = {"auto_map": {"DiffusionPipeline": "custom.Pipeline"}}
    with pytest.raises(PipelineResolutionError) as caught:
        PipelineResolver().load(
            "/snapshot", remote, "TEXT_TO_VIDEO", {}, diffusers_module=SimpleNamespace()
        )
    assert caught.value.code == "REMOTE_CODE_REQUIRED"


def test_capability_resolver_uses_signature_without_inventing_multi_modes() -> None:
    class ImageVideoPipeline:
        def __call__(self, prompt, image, num_frames, height=None, width=None):
            del prompt, image, num_frames, height, width

    capabilities = CapabilityResolver().resolve(metadata(), ImageVideoPipeline)
    assert "TEXT_TO_VIDEO" in capabilities
    assert "IMAGE_TO_VIDEO" in capabilities
    assert "MULTI_IMAGE_TO_VIDEO" not in capabilities
    assert "START_END_IMAGE_TO_VIDEO" not in capabilities
    assert "KEYFRAMES_TO_VIDEO" not in capabilities


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        ("images", "MULTI_IMAGE_TO_VIDEO"),
        ("last_image", "START_END_IMAGE_TO_VIDEO"),
        ("keyframes", "KEYFRAMES_TO_VIDEO"),
    ],
)
def test_capability_resolver_requires_explicit_multi_input_parameter(
    parameter: str, expected: str
) -> None:
    namespace: dict[str, object] = {}
    exec(
        f"def __call__(self, prompt, image, num_frames, {parameter}=None): pass",
        namespace,
    )
    pipeline = type("DynamicPipeline", (), {"__call__": namespace["__call__"]})
    assert expected in CapabilityResolver().resolve(metadata(), pipeline)


def test_generic_adapter_sends_only_supported_aliases() -> None:
    observed: dict[str, object] = {}

    class Pipeline:
        def __call__(self, prompt, reference_images, video_length, end_image=None):
            observed.update(
                prompt=prompt,
                reference_images=reference_images,
                video_length=video_length,
                end_image=end_image,
            )
            return SimpleNamespace(frames=[[Image.new("RGB", (16, 16))] * 3])

    images = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]
    result = GenericDiffusersAdapter().generate(
        Pipeline(),
        {"metadata": metadata(), "generator": None},
        {
            "capability": "START_END_IMAGE_TO_VIDEO",
            "prompt": "test",
            "frames": 9,
            "resolved_input_images": images,
            "input_images": [
                {"role": "start_frame"},
                {"role": "end_frame"},
            ],
        },
    )
    assert observed["reference_images"] == images
    assert observed["end_image"] is images[-1]
    assert observed["video_length"] == 9
    assert result["frames"]


def test_generic_adapter_never_passes_an_opaque_asset_id_to_a_pipeline() -> None:
    called = False

    class Pipeline:
        def __call__(self, prompt, image, num_frames):
            nonlocal called
            called = True
            del prompt, image, num_frames
            return SimpleNamespace(frames=[])

    with pytest.raises(NormalizationError) as caught:
        GenericDiffusersAdapter().generate(
            Pipeline(),
            {"metadata": metadata(), "generator": None},
            {
                "capability": "IMAGE_TO_VIDEO",
                "prompt": "test",
                "frames": 3,
                "input_images": [
                    {"asset_id": "opaque-asset-id", "order": 0, "role": "start_frame"}
                ],
            },
        )
    assert caught.value.code == "INVALID_INPUT_ASSET"
    assert called is False


def test_input_normalizer_produces_real_pil_images(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 12), "green").save(source)
    request = InputNormalizer(tmp_path).normalize(
        {
            "capability": "IMAGE_TO_VIDEO",
            "input_path": str(source),
            "input_images": [{"source": str(source), "role": "start_frame"}],
        },
        {"image"},
    )
    assert isinstance(request["input_image"], Image.Image)
    assert isinstance(request["resolved_input_images"][0], Image.Image)


def test_output_normalizer_encodes_and_probes_real_h264(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (32, 24), (index * 40, 0, 0)) for index in range(4)]
    output = tmp_path / "result.mp4"
    probe = OutputNormalizer(tmp_path).write_video(frames, output, 8)
    assert output.is_file()
    assert probe["codec"] == "h264"
    assert probe["frames"] == 4
    assert probe["duration"] > 0


def test_output_normalizer_accepts_batched_numpy_video_layouts() -> None:
    import numpy as np

    batch_frames_last = np.zeros((1, 5, 12, 20, 3), dtype=np.float32)
    batch_channels_first = np.zeros((1, 3, 5, 12, 20), dtype=np.float32)
    assert len(OutputNormalizer.normalize_frames(batch_frames_last)) == 5
    assert len(OutputNormalizer.normalize_frames(batch_channels_first)) == 5


def test_output_normalizer_accepts_torch_like_tensor_output() -> None:
    import numpy as np

    class TensorBoundary:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    tensor = TensorBoundary(np.zeros((1, 3, 6, 12, 20), dtype=np.float32))
    assert len(OutputNormalizer.normalize_frames(tensor)) == 6


def test_output_normalizer_rejects_png_for_video(tmp_path: Path) -> None:
    with pytest.raises(NormalizationError) as caught:
        OutputNormalizer(tmp_path).write_video(
            [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))],
            tmp_path / "result.png",
            8,
        )
    assert caught.value.code == "INVALID_OUTPUT_PATH"


def test_model_profile_uses_metadata_and_normalizes_constraints() -> None:
    profile = ModelRuntimeProfile.from_metadata(
        {
            "model_index": {
                "fps": 12,
                "num_inference_steps": 36,
                "guidance_scale": 4.5,
            },
            "config": {"dimension_multiple": 32, "temporal_compression_ratio": 4},
        }
    )
    values = profile.normalize(
        {"quality": "720p", "duration_seconds": 4},
        video=True,
    )
    assert values["fps"] == 12
    assert values["num_inference_steps"] == 36
    assert values["guidance_scale"] == 4.5
    assert values["width"] % 32 == 0
    assert values["height"] % 32 == 0
    assert (values["num_frames"] - 1) % 4 == 0
    assert values["num_frames"] == 49


@pytest.mark.parametrize(
    ("quality", "aspect_ratio", "landscape"),
    [
        ("480p", "16:9", True),
        ("720p", "16:9", True),
        ("480p", "9:16", False),
        ("720p", "1:1", None),
    ],
)
def test_resolution_resolver_supports_required_quality_and_ratios(
    quality: str,
    aspect_ratio: str,
    landscape: bool | None,
) -> None:
    result = ResolutionResolver().resolve(
        quality=quality,
        aspect_ratio=aspect_ratio,
        metadata={"config": {"dimension_multiple": 32}},
    )
    assert result.width % 32 == 0
    assert result.height % 32 == 0
    assert result.requested_quality == quality
    assert result.requested_aspect_ratio == aspect_ratio
    if landscape is True:
        assert result.width > result.height
    elif landscape is False:
        assert result.width < result.height
    else:
        assert result.width == result.height


def test_resolution_resolver_reads_pipeline_multiple_and_never_exceeds_limits() -> None:
    pipeline = SimpleNamespace(
        vae_scale_factor=8,
        transformer=SimpleNamespace(config=SimpleNamespace(patch_size=4)),
    )
    result = ResolutionResolver().resolve(
        quality="720p",
        aspect_ratio="16:9",
        pipeline=pipeline,
        metadata={"config": {"max_width": 1216, "max_height": 704}},
    )
    assert result.dimension_multiple == 32
    assert result.width <= 1216
    assert result.height <= 704
    assert result.width % 32 == 0
    assert result.height % 32 == 0


def test_resolution_resolver_prefers_closest_720p_size_over_overscaling() -> None:
    result = ResolutionResolver().resolve(
        quality="720p",
        aspect_ratio="16:9",
        metadata={"config": {"dimension_multiple": 32}},
    )
    assert (result.width, result.height) == (1280, 704)


def test_temporal_normalization_chooses_97_frames_for_four_seconds_at_24_fps() -> None:
    profile = ModelRuntimeProfile.from_metadata(
        {"config": {"temporal_compression_ratio": 4}}
    )
    values = profile.normalize(
        {"quality": "480p", "duration_seconds": 4, "fps": 24},
        video=True,
    )
    assert values["fps"] == 24
    assert values["num_frames"] == 97


def test_worker_compatibility_reports_installed_class_and_real_capabilities(tmp_path: Path) -> None:
    settings = Settings(
        app_env="TEST",
        worker_token=None,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
        hf_home=tmp_path / "hf",
        gpu_required=False,
        minimum_weights_bytes=1,
        default_model_id="test-model",
        default_repository="example/test-model",
    )
    result = RuntimeManager(settings).check_compatibility(
        {
            "pipeline_class": "WanPipeline",
            "library_name": "diffusers",
            "pipeline_tag": None,
            "tags": [],
            "architectures": [],
        }
    )
    assert result["runtime_supported"] is True
    assert result["pipeline_class"] == "WanPipeline"
    assert "TEXT_TO_VIDEO" in result["runtime_capabilities"]
    assert "IMAGE_TO_VIDEO" in result["runtime_capabilities"]


def test_worker_compatibility_distinguishes_old_diffusers_and_remote_code(tmp_path: Path) -> None:
    settings = Settings(
        app_env="TEST",
        worker_token=None,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
        hf_home=tmp_path / "hf",
        gpu_required=False,
        minimum_weights_bytes=1,
        default_model_id="test-model",
        default_repository="example/test-model",
    )
    manager = RuntimeManager(settings)
    missing = manager.check_compatibility(
        {"pipeline_class": "FutureMissingPipeline", "library_name": "diffusers"}
    )
    assert missing["runtime_supported"] is False
    assert missing["error_code"] == "DIFFUSERS_VERSION_TOO_OLD"

    remote = manager.check_compatibility(
        {
            "pipeline_class": "WanPipeline",
            "library_name": "diffusers",
            "trust_remote_code": True,
        }
    )
    assert remote["runtime_supported"] is False
    assert remote["error_code"] == "REMOTE_CODE_REQUIRED"


def test_worker_compatibility_keeps_incomplete_diffusers_metadata_unknown(
    tmp_path: Path,
) -> None:
    settings = Settings(
        app_env="TEST",
        worker_token=None,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
        hf_home=tmp_path / "hf",
        gpu_required=False,
        minimum_weights_bytes=1,
        default_model_id="test-model",
        default_repository="example/test-model",
    )
    result = RuntimeManager(settings).check_compatibility(
        {
            "pipeline_class": None,
            "library_name": "diffusers",
            "pipeline_tag": "text-to-video",
            "tags": ["diffusers", "text-to-video"],
            "architectures": ["FutureTransformer3DModel"],
        }
    )
    assert result["compatibility_status"] == "UNKNOWN"
    assert result["runtime_supported"] is False
    assert result["error_code"] is None


@pytest.mark.parametrize(
    ("capability", "quality", "with_image"),
    [
        ("TEXT_TO_VIDEO", "480p", False),
        ("IMAGE_TO_VIDEO", "480p", True),
        ("TEXT_TO_VIDEO", "720p", False),
    ],
)
def test_real_cpu_video_contract_traverses_runtime_and_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
    quality: str,
    with_image: bool,
) -> None:
    observed: dict[str, object] = {}

    class ContractVideoPipeline:
        def __call__(
            self,
            prompt,
            num_frames,
            height,
            width,
            image=None,
            generator=None,
        ):
            del generator
            observed.update(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                image=image,
            )
            frames = [
                Image.new("RGB", (width, height), (index * 30, 10, 20))
                for index in range(num_frames)
            ]
            return SimpleNamespace(frames=[frames])

    fake_diffusers = SimpleNamespace(ContractVideoPipeline=ContractVideoPipeline)
    monkeypatch.setitem(__import__("sys").modules, "diffusers", fake_diffusers)

    settings = Settings(
        app_env="TEST",
        worker_token=None,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
        hf_home=tmp_path / "hf",
        gpu_required=False,
        minimum_weights_bytes=1,
        default_model_id="contract-video",
        default_repository="example/contract-video",
    )
    manager = RuntimeManager(settings)

    class Generator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, _seed: int) -> "Generator":
            return self

    torch = SimpleNamespace(
        Generator=Generator,
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
        __version__="test",
    )
    manager._runtime_modules = RuntimeImports(
        torch=torch,
        hf_api=object(),
        snapshot_download=object(),
    )
    runtime_metadata = {
        **metadata("ContractVideoPipeline"),
        "pipeline_tag": "image-to-video" if with_image else "text-to-video",
        "config": {
            "dimension_multiple": 32,
            "temporal_compression_ratio": 1,
            "min_frames": 2,
        },
    }
    manager._loaded["contract-video"] = LoadedModel(
        model_id="contract-video",
        repository="example/contract-video",
        revision="fixture",
        device="cpu",
        loaded_at=0.0,
        validation_test=True,
        precision="float32",
        load_benchmark={},
        pipeline=ContractVideoPipeline(),
        capability=capability,
        metadata=runtime_metadata,
    )

    request: dict[str, object] = {
        "job_id": f"job-{capability.lower()}-{quality}",
        "model_id": "contract-video",
        "capability": capability,
        "prompt": "CPU contract",
        "output_relative_path": f"contracts/{capability.lower()}-{quality}.mp4",
        "quality": quality,
        "aspect_ratio": "16:9",
        "duration_seconds": 1,
        "fps": 2,
        "steps": 1,
    }
    if with_image:
        source = tmp_path / "source.png"
        Image.new("RGB", (64, 64), "green").save(source)
        request["input_path"] = str(source)
        request["input_images"] = [
            {"source": str(source), "order": 0, "role": "start_frame"}
        ]

    result = manager.generate_image(request)
    assert result["state"] == "COMPLETED", result
    assert result["requested_quality"] == quality
    assert result["requested_aspect_ratio"] == "16:9"
    assert result["actual_width"] > result["actual_height"] > 0
    assert result["actual_frames"] > 1
    assert result["actual_fps"] > 0
    assert observed["width"] == result["actual_width"]
    assert observed["height"] == result["actual_height"]
    if with_image:
        assert isinstance(observed["image"], Image.Image)


def test_generic_runtime_contract_uses_the_complete_vidioai_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {"loads": 0, "generations": 0}

    class ContractVideoPipeline:
        @classmethod
        def from_pretrained(cls, snapshot: str, **kwargs):
            observed["loads"] = int(observed["loads"]) + 1
            observed["snapshot"] = snapshot
            observed["load_kwargs"] = kwargs
            return cls()

        def __call__(
            self,
            prompt,
            num_frames,
            height,
            width,
            image=None,
            generator=None,
        ):
            del image, generator
            observed["generations"] = int(observed["generations"]) + 1
            observed["prompt"] = prompt
            observed["kwargs"] = {
                "num_frames": num_frames,
                "height": height,
                "width": width,
            }
            frames = [
                Image.new("RGB", (width, height), (20 * index, 40, 80))
                for index in range(num_frames)
            ]
            return SimpleNamespace(frames=[frames])

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            ContractVideoPipeline=ContractVideoPipeline,
            DiffusionPipeline=ContractVideoPipeline,
        ),
    )

    settings = Settings(
        app_env="TEST",
        worker_token=None,
        models_dir=tmp_path / "models",
        outputs_dir=tmp_path / "outputs",
        work_dir=tmp_path / "work",
        hf_home=tmp_path / "hf",
        gpu_required=False,
        minimum_weights_bytes=1,
        default_model_id="contract-video",
        default_repository="example/contract-video",
    )
    manager = RuntimeManager(settings)
    snapshot = settings.models_dir / "contract-video" / "fixture-revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ContractVideoPipeline",
                "_diffusers_version": "contract",
                "transformer": ["diffusers", "ContractTransformer3DModel"],
                "vae": ["diffusers", "ContractVAE"],
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "dimension_multiple": 32,
                "temporal_compression_ratio": 1,
                "min_frames": 2,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot.parent / "active.json").write_text(
        json.dumps(
            {
                "model_id": "contract-video",
                "repository": "example/contract-video",
                "revision": "fixture-revision",
            }
        ),
        encoding="utf-8",
    )

    class Generator:
        def __init__(self, device: str) -> None:
            self.device = device

        def manual_seed(self, _seed: int) -> "Generator":
            return self

    torch = SimpleNamespace(
        Generator=Generator,
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
        __version__="test",
    )
    manager._runtime_modules = RuntimeImports(
        torch=torch,
        hf_api=object(),
        snapshot_download=object(),
    )

    status = manager.load_model("contract-video")
    assert status["state"] == "READY"
    assert status["pipeline_class"] == "ContractVideoPipeline"
    assert status["capability"] == "TEXT_TO_VIDEO"
    assert observed["loads"] == 1
    assert observed["generations"] == 0

    result = manager.generate_image(
        {
            "job_id": "job-complete-generic-chain",
            "model_id": "contract-video",
            "capability": "TEXT_TO_VIDEO",
            "prompt": "Complete generic contract",
            "output_relative_path": "contracts/complete-generic.mp4",
            "quality": "480p",
            "aspect_ratio": "16:9",
            "duration_seconds": 1,
            "fps": 2,
            "steps": 1,
        }
    )
    assert result["state"] == "COMPLETED", result
    assert observed["generations"] == 1
    assert observed["prompt"] == "Complete generic contract"
    assert observed["kwargs"] == {
        "num_frames": 2,
        "height": 480,
        "width": 864,
    }
    probe = OutputNormalizer.probe_video(
        settings.outputs_dir / "contracts" / "complete-generic.mp4"
    )
    assert probe["codec"] == "h264"
    assert probe["frames"] == 2
