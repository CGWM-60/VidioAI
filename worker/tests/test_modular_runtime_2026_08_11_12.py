from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.adapters.modular_diffusers import ModularDiffusersAdapter
from app.capability_resolver import CapabilityResolver
from app.modular_runtime import ModularManifestResolver, ModularRuntimeError
from app.pipeline_resolver import PipelineResolver


def modular_metadata(**extra):
    metadata = {
        "library_name": "diffusers",
        "is_modular": True,
        "modular_model_index": {
            "_class_name": "ExampleVideoModularPipeline",
        },
        "pipeline_tag": "image-to-video",
        "raw_tags": [
            "image-to-video",
            "start-end-image-to-video",
            "multi-image-to-video",
        ],
        "architectures": [],
        "base_models": [],
        "config": {},
    }
    metadata.update(extra)
    return metadata


def test_modular_manifest_extracts_external_components() -> None:
    manifest = {
        "_class_name": "ExampleModularPipeline",
        "scheduler": [
            None,
            None,
            {
                "pretrained_model_name_or_path": "owner/base",
                "subfolder": "scheduler",
                "type_hint": [
                    "diffusers",
                    "Scheduler",
                ],
            },
        ],
        "transformer": [
            None,
            None,
            {
                "pretrained_model_name_or_path": "owner/int8-transformer",
                "revision": "abc",
                "subfolder": "transformer",
                "type_hint": [
                    "diffusers",
                    "VideoTransformer",
                ],
            },
        ],
        "vae": [
            None,
            None,
            {
                "subfolder": "vae",
                "type_hint": [
                    "diffusers",
                    "VideoVAE",
                ],
            },
        ],
    }

    external = ModularManifestResolver.external_components(
        manifest,
        base_repository="owner/base",
    )

    assert len(external) == 1
    assert external[0].name == "transformer"
    assert external[0].repository == "owner/int8-transformer"
    assert external[0].revision == "abc"
    assert external[0].subfolder == "transformer"


def test_pipeline_resolver_accepts_native_modular_diffusers() -> None:
    class FakeModularPipeline:
        pass

    fake_diffusers = SimpleNamespace(
        ModularPipeline=FakeModularPipeline
    )

    resolution = PipelineResolver().resolve_class(
        modular_metadata(),
        diffusers_module=fake_diffusers,
    )

    assert resolution.runtime_supported is True
    assert resolution.pipeline_cls is FakeModularPipeline
    assert resolution.strategy == "modular-pipeline"


def test_pipeline_resolver_rejects_modular_remote_code() -> None:
    metadata = modular_metadata(
        config={
            "auto_map": {
                "ModularPipelineBlocks": "block.CustomBlocks"
            }
        }
    )

    resolution = PipelineResolver().resolve_class(
        metadata,
        diffusers_module=SimpleNamespace(
            ModularPipeline=object
        ),
    )

    assert resolution.runtime_supported is False
    assert resolution.runtime_reason == "REMOTE_CODE_REQUIRED"


def test_capability_resolver_reads_modular_block_inputs() -> None:
    class Input:
        def __init__(self, name: str):
            self.name = name

    pipeline = SimpleNamespace(
        blocks=SimpleNamespace(
            inputs=[
                Input("prompt"),
                Input("first_image"),
                Input("last_image"),
                Input("reference_images"),
                Input("num_frames"),
            ]
        )
    )

    capabilities = CapabilityResolver().runtime_capabilities(
        pipeline
    )

    assert "TEXT_TO_VIDEO" in capabilities
    assert "IMAGE_TO_VIDEO" in capabilities
    assert "START_END_IMAGE_TO_VIDEO" in capabilities
    assert "MULTI_IMAGE_TO_VIDEO" in capabilities


def test_modular_adapter_forwards_first_last_and_references() -> None:
    class Input:
        def __init__(self, name: str):
            self.name = name

    observed = {}

    class Pipeline:
        blocks = SimpleNamespace(
            inputs=[
                Input("prompt"),
                Input("first_image"),
                Input("last_image"),
                Input("reference_images"),
                Input("num_frames"),
                Input("num_inference_steps"),
            ]
        )

        def __call__(self, **kwargs):
            observed.update(kwargs)
            return {
                "frames": [["f1", "f2", "f3"]],
            }

    start = object()
    end = object()
    ref = object()

    output = ModularDiffusersAdapter().generate(
        Pipeline(),
        {"generator": None},
        {
            "capability": "START_END_IMAGE_TO_VIDEO",
            "prompt": "The subject turns around.",
            "frames": 25,
            "steps": 8,
            "resolved_input_images": [
                start,
                end,
                ref,
            ],
            "input_images": [
                {
                    "order": 0,
                    "role": "start_frame",
                },
                {
                    "order": 1,
                    "role": "end_frame",
                },
                {
                    "order": 2,
                    "role": "reference",
                },
            ],
        },
    )

    assert observed["prompt"] == "The subject turns around."
    assert observed["first_image"] is start
    assert observed["last_image"] is end
    assert "reference_images" not in observed
    assert observed["num_frames"] == 25
    assert observed["num_inference_steps"] == 8
    assert output["frames"] == [["f1", "f2", "f3"]]


def test_modular_adapter_ref2v_forwards_reference_images() -> None:
    class Input:
        def __init__(self, name: str):
            self.name = name

    observed = {}

    class Pipeline:
        blocks = SimpleNamespace(
            inputs=[
                Input("prompt"),
                Input("image"),
                Input("reference_images"),
                Input("num_frames"),
            ]
        )

        def __call__(self, **kwargs):
            observed.update(kwargs)
            return {"frames": [["ok1", "ok2"]]}

    start = object()
    ref1 = object()
    ref2 = object()

    ModularDiffusersAdapter().generate(
        Pipeline(),
        {"generator": None},
        {
            "capability": "MULTI_IMAGE_TO_VIDEO",
            "prompt": "Use the references for appearance.",
            "frames": 17,
            "resolved_input_images": [
                start,
                ref1,
                ref2,
            ],
            "input_images": [
                {"order": 0, "role": "start_frame"},
                {"order": 1, "role": "reference"},
                {"order": 2, "role": "reference"},
            ],
        },
    )

    assert observed["image"] is start
    assert observed["reference_images"] == [ref1, ref2]


def test_modular_adapter_refuses_opaque_input_contract() -> None:
    class Pipeline:
        blocks = SimpleNamespace(inputs=[])

        def __call__(self, **kwargs):
            return {"frames": [["unused"]]}

    with pytest.raises(ModularRuntimeError) as error:
        ModularDiffusersAdapter().generate(
            Pipeline(),
            {"generator": None},
            {
                "capability": "TEXT_TO_VIDEO",
                "prompt": "test",
            },
        )

    assert error.value.code == "MODULAR_INPUT_CONTRACT_UNKNOWN"


def test_materialization_manifest_is_local_and_roundtrips(
    tmp_path: Path,
) -> None:
    component = (
        tmp_path
        / "vidioai"
        / "modular-components"
        / "abc"
        / "transformer"
    )
    component.mkdir(parents=True)
    (component / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "load_in_8bit": True
                }
            }
        ),
        encoding="utf-8",
    )

    ModularManifestResolver.write_materialization(
        tmp_path,
        [
            {
                "name": "transformer",
                "repository": "owner/variant",
                "resolved_revision": "sha",
                "local_root": (
                    "vidioai/modular-components/abc"
                ),
                "subfolder": "transformer",
            }
        ],
    )

    configs = (
        ModularManifestResolver.component_config_paths(
            tmp_path
        )
    )
    assert configs["transformer"].is_file()



def test_modular_adapter_prefers_observed_runtime_capabilities() -> None:
    adapter = ModularDiffusersAdapter()
    capabilities = adapter.supported_capabilities(
        modular_metadata(
            capabilities=[
                "TEXT_TO_VIDEO",
                "IMAGE_TO_VIDEO",
                "START_END_IMAGE_TO_VIDEO",
                "MULTI_IMAGE_TO_VIDEO",
            ],
            runtime_capabilities=[
                "TEXT_TO_VIDEO",
                "IMAGE_TO_VIDEO",
                "START_END_IMAGE_TO_VIDEO",
                "MULTI_IMAGE_TO_VIDEO",
            ],
        )
    )
    assert capabilities == [
        "TEXT_TO_VIDEO",
        "IMAGE_TO_VIDEO",
        "MULTI_IMAGE_TO_VIDEO",
        "START_END_IMAGE_TO_VIDEO",
    ]
