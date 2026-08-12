from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.adapters.image_to_image import ImageToImageAdapter
from app.adapters.image_to_video import ImageToVideoAdapter
from app.inference_recipe import InferenceRecipeResolver
from app.model_bundle import BundleError, ModelBundleManager
from app.schemas import InstallModelRequest


def test_recipe_request_overrides_bundle() -> None:
    class Pipeline:
        def __call__(
            self,
            prompt,
            width=1024,
            height=1024,
            guidance_scale=1.0,
        ):
            del prompt, width, height, guidance_scale

    plan = InferenceRecipeResolver().resolve(
        pipeline=Pipeline(),
        request={
            "quality": "quality",
            "guidance_scale": 3.5,
        },
        bundle={
            "recipe": {
                "quality_mode": "quality",
                "guidance_scale": 2.0,
                "width": 768,
            }
        },
    )

    assert plan.values["guidance_scale"] == 3.5
    assert plan.sources["guidance_scale"] == "request"
    assert plan.values["width"] == 768
    assert plan.sources["width"] == "bundle_recipe"
    assert plan.values["height"] == 1024
    assert plan.sources["height"] == "pipeline_signature"


def test_quality_uses_real_1024_signature_default() -> None:
    class Pipeline:
        def __call__(self, prompt, width=1024, height=1024):
            del prompt, width, height

    plan = InferenceRecipeResolver().resolve(
        pipeline=Pipeline(),
        request={"quality": "quality"},
        bundle={},
    )
    assert plan.values == {
        "width": 1024,
        "height": 1024,
    }


def test_quality_does_not_invent_resolution_when_pipeline_default_is_none() -> None:
    class Pipeline:
        def __call__(self, prompt, width=None, height=None):
            del prompt, width, height

    plan = InferenceRecipeResolver().resolve(
        pipeline=Pipeline(),
        request={"quality": "quality"},
        bundle={},
    )
    assert "width" not in plan.values
    assert "height" not in plan.values


def test_recipe_supports_true_cfg_and_max_sequence_length() -> None:
    class Pipeline:
        def __call__(
            self,
            prompt,
            true_cfg_scale=1.0,
            max_sequence_length=512,
        ):
            del prompt, true_cfg_scale, max_sequence_length

    plan = InferenceRecipeResolver().resolve(
        pipeline=Pipeline(),
        request={},
        bundle={
            "recipe": {
                "quality_mode": "quality",
                "true_cfg_scale": 4.0,
                "max_sequence_length": 1024,
            }
        },
    )
    assert plan.values["true_cfg_scale"] == 4.0
    assert plan.values["max_sequence_length"] == 1024




def test_bundle_without_lora_does_not_require_huggingface_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            raise AssertionError(
                "huggingface_hub ne doit pas être importé pour un bundle sans LoRA"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    bundle = ModelBundleManager().materialize(
        snapshot=snapshot,
        repository="owner/base",
        revision="commit-sha",
        loras=None,
        recipe={"quality_mode": "native"},
        token=None,
        cache_dir=tmp_path / "cache",
        preserve_existing=False,
    )

    assert bundle["base_model"] == {
        "repository": "owner/base",
        "revision": "commit-sha",
    }
    assert bundle["loras"] == []
    assert bundle["recipe"]["quality_mode"] == "native"

def test_bundle_apply_loras_uses_names_and_scales(tmp_path: Path) -> None:
    calls = []
    activated = []

    class Pipeline:
        def load_lora_weights(self, path, **kwargs):
            calls.append((path, kwargs))

        def set_adapters(self, names, adapter_weights=None):
            activated.append((names, adapter_weights))

    for name in ("adult-style", "detail"):
        directory = tmp_path / "vidioai" / "loras" / name
        directory.mkdir(parents=True)
        (directory / "adapter.safetensors").write_bytes(b"test")

    bundle = {
        "loras": [
            {
                "repository": "owner/a",
                "revision": "abc",
                "adapter_name": "adult-style",
                "weight_name": "adapter.safetensors",
                "local_path": "vidioai/loras/adult-style",
                "scale": 0.85,
                "enabled": True,
            },
            {
                "repository": "owner/b",
                "revision": "def",
                "adapter_name": "detail",
                "weight_name": "adapter.safetensors",
                "local_path": "vidioai/loras/detail",
                "scale": 0.4,
                "enabled": True,
            },
        ]
    }

    pipeline = Pipeline()
    ModelBundleManager.apply_loras(pipeline, tmp_path, bundle)

    assert len(calls) == 2
    assert calls[0][1]["adapter_name"] == "adult-style"
    assert calls[1][1]["adapter_name"] == "detail"
    assert activated == [
        (["adult-style", "detail"], [0.85, 0.4])
    ]


def test_disabled_lora_is_not_loaded(tmp_path: Path) -> None:
    class Pipeline:
        def __init__(self):
            self.calls = 0

        def load_lora_weights(self, path, **kwargs):
            del path, kwargs
            self.calls += 1

    pipeline = Pipeline()
    ModelBundleManager.apply_loras(
        pipeline,
        tmp_path,
        {
            "loras": [
                {
                    "enabled": False,
                    "adapter_name": "off",
                }
            ]
        },
    )
    assert pipeline.calls == 0


def test_pipeline_without_lora_support_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "vidioai" / "loras" / "x"
    directory.mkdir(parents=True)
    (directory / "x.safetensors").write_bytes(b"x")

    with pytest.raises(BundleError) as error:
        ModelBundleManager.apply_loras(
            object(),
            tmp_path,
            {
                "loras": [
                    {
                        "enabled": True,
                        "adapter_name": "x",
                        "weight_name": "x.safetensors",
                        "local_path": "vidioai/loras/x",
                        "scale": 1.0,
                    }
                ]
            },
        )
    assert error.value.code == "LORA_UNSUPPORTED"


def test_weight_selection_prefers_standard_lora_name() -> None:
    selected = ModelBundleManager.select_weight_name(
        [
            "other.safetensors",
            "pytorch_lora_weights.safetensors",
        ]
    )
    assert selected == "pytorch_lora_weights.safetensors"


def test_bundle_schema_roundtrip() -> None:
    request = InstallModelRequest.model_validate(
        {
            "model_id": "base",
            "repository": "owner/base",
            "revision": "main",
            "capabilities": ["IMAGE_TO_IMAGE"],
            "loras": [
                {
                    "repository": "owner/lora",
                    "scale": 0.8,
                }
            ],
            "recipe": {
                "quality_mode": "quality",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 40,
                "true_cfg_scale": 4.0,
            },
        }
    )
    payload = request.model_dump(exclude_none=True)
    assert payload["loras"][0]["repository"] == "owner/lora"
    assert payload["recipe"]["width"] == 1024
    assert payload["recipe"]["true_cfg_scale"] == 4.0


def test_i2i_prompt_is_preserved_and_recipe_values_are_forwarded() -> None:
    observed = {}
    image = object()

    class Pipeline:
        def __call__(
            self,
            prompt,
            image,
            width=None,
            height=None,
            num_inference_steps=None,
            true_cfg_scale=None,
        ):
            observed.update(
                prompt=prompt,
                image=image,
                width=width,
                height=height,
                steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
            )
            return SimpleNamespace(images=["ok"])

    output = ImageToImageAdapter().generate(
        Pipeline(),
        {"generator": None},
        {
            "capability": "IMAGE_TO_IMAGE",
            "prompt": "Turn the person into a medieval knight wearing bright red armor.",
            "input_image": image,
            "width": 1024,
            "height": 1024,
            "steps": 40,
            "true_cfg_scale": 4.0,
        },
    )

    assert output["images"] == ["ok"]
    assert observed["prompt"].startswith("Turn the person")
    assert observed["image"] is image
    assert observed["width"] == 1024
    assert observed["height"] == 1024
    assert observed["steps"] == 40
    assert observed["true_cfg_scale"] == 4.0


def test_i2v_delivery_fps_is_not_injected_as_inference_fps() -> None:
    observed = {}
    image = object()

    class Pipeline:
        def __call__(self, prompt, image, fps=None):
            observed.update(prompt=prompt, image=image, fps=fps)
            return SimpleNamespace(frames=[["f1", "f2"]])

    ImageToVideoAdapter().generate(
        Pipeline(),
        {"generator": None},
        {
            "capability": "IMAGE_TO_VIDEO",
            "prompt": "A person turns around and raises one hand.",
            "fps": 24,
            "resolved_input_images": [image],
            "input_images": [
                {
                    "asset_id": "a",
                    "order": 0,
                    "role": "start_frame",
                }
            ],
        },
    )
    assert observed["prompt"].startswith("A person")
    assert observed["image"] is image
    assert observed["fps"] is None
