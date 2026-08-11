"""Runtime Diffusers minimal pour les tests CPU sans dependances GPU."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class _ImagePipeline:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, prompt, height=None, width=None, **_kwargs):
        from PIL import Image

        del prompt
        return SimpleNamespace(images=[Image.new("RGB", (width or 64, height or 64))])

    def to(self, _device):
        return self


class _VideoPipeline:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, prompt, num_frames=3, image=None, height=None, width=None, **_kwargs):
        from PIL import Image

        del prompt, image
        frame = Image.new("RGB", (width or 64, height or 64))
        return SimpleNamespace(frames=[[frame] * max(2, int(num_frames or 3))])

    def to(self, _device):
        return self


class _ImageToImagePipeline(_ImagePipeline):
    def __call__(self, prompt, image, height=None, width=None, **_kwargs):
        del image
        return super().__call__(prompt, height=height, width=width)


@pytest.fixture(autouse=True)
def installed_diffusers_runtime(monkeypatch: pytest.MonkeyPatch):
    try:
        __import__("diffusers")
        yield
        return
    except ModuleNotFoundError:
        pass

    fake = SimpleNamespace(
        StableDiffusionPipeline=_ImagePipeline,
        StableDiffusionImg2ImgPipeline=_ImageToImagePipeline,
        FluxPipeline=_ImagePipeline,
        WanPipeline=_VideoPipeline,
        WanImageToVideoPipeline=_VideoPipeline,
        LTXImageToVideoPipeline=_VideoPipeline,
        CogVideoXImageToVideoPipeline=_VideoPipeline,
        DiffusionPipeline=_ImagePipeline,
        AutoPipelineForText2Image=_ImagePipeline,
        AutoPipelineForImage2Image=_ImagePipeline,
        AutoPipelineForText2Video=_VideoPipeline,
        AutoPipelineForImage2Video=_VideoPipeline,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    yield
