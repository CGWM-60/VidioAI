from __future__ import annotations

from app.engines.comfyui import ComfyUIEngine


class Response:
    def __init__(self, payload: bytes):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return None
    def read(self):
        return self.payload


def test_comfyui_node_types_comes_from_object_info():
    def opener(request, *, timeout):
        del timeout
        if request.full_url.endswith('/object_info'):
            return Response(b'{"KSampler": {}, "VAEDecode": {}}')
        raise AssertionError(request.full_url)

    engine = ComfyUIEngine('http://comfy.invalid', opener=opener)
    assert engine.node_types() == {'KSampler', 'VAEDecode'}
