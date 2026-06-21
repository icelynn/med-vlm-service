# -*- coding: utf-8 -*-
"""Offline tests for the demo HF-transformers inference path (_call_hf / _stream_hf).

The model/processor are mocked, so no real model download or GPU is needed — only
torch itself must be importable (already a project dependency; ~2.6s import cost,
which is why this is a separate file from the fast test_streaming.py suite).

Run:  ./.venv/Scripts/python.exe python/tests/test_hf_inference.py
"""
import asyncio
import base64
import os
import sys
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("ENV", "demo")
os.environ.setdefault("MODEL_ROLE", "main")
os.environ.setdefault("OLLAMA_API_URL", "http://upstream/api/chat")

from PIL import Image

import inference  # noqa: E402


def _tiny_png_b64(size=(4, 4)):
    img = Image.new("RGB", size, color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --- fakes: no real model weights involved -----------------------------------
class _FakeTensor:
    def __init__(self, n):
        self.shape = (1, n)


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    tokenizer = object()
    _next_output = ""

    def apply_chat_template(self, messages, **kw):
        assert messages[0]["content"][0]["text"], "system prompt missing from messages"
        assert messages[1]["content"][0]["type"] == "image", "image missing from messages"
        return _FakeInputs(input_ids=_FakeTensor(10))

    def decode(self, tokens, **kw):
        return self._next_output


class _FakeModel:
    device = "cpu"

    def generate(self, **kw):
        import torch
        return torch.zeros((1, 15), dtype=torch.long)


def _set_fake_model(output_text):
    fake_processor = _FakeProcessor()
    fake_processor._next_output = output_text
    inference._hf_cache[inference.config.model] = (fake_processor, _FakeModel())


# --- tests --------------------------------------------------------------------
async def test_resolve_backend_demo_is_hf():
    inference.config.ENV = "demo"
    assert inference._resolve_service_backend() == "hf"


async def test_resolve_backend_dev_still_blocked():
    """ENV=dev must still redirect to run_baseline.py, not become servable just because
    demo also resolved to 'hf' now."""
    inference.config.ENV = "dev"
    raised = False
    try:
        inference._resolve_service_backend()
    except ValueError as e:
        raised = "run_baseline.py" in str(e)
    finally:
        inference.config.ENV = "demo"
    assert raised, "expected ENV=dev to raise pointing at run_baseline.py"


async def test_call_hf_strips_thinking_block():
    _set_fake_model("<think>reasoning...</think>HEMORRHAGE: no")
    try:
        text = await inference.generate_medical_report(_tiny_png_b64(), "p", "s")
    finally:
        inference._hf_cache.clear()
    assert text == "HEMORRHAGE: no", repr(text)


async def test_call_hf_passthrough_when_no_thinking():
    _set_fake_model("HEMORRHAGE: yes\nSUBTYPES: IPH")
    try:
        text = await inference.generate_medical_report(_tiny_png_b64(), "p", "s")
    finally:
        inference._hf_cache.clear()
    assert text == "HEMORRHAGE: yes\nSUBTYPES: IPH", repr(text)


def test_strip_thinking_unit():
    assert inference._strip_thinking("<think>x</think>final") == "final"
    assert inference._strip_thinking("no tags here") == "no tags here"


def test_decode_image_caps_size():
    b64 = _tiny_png_b64(size=(2000, 1000))
    decoded = inference._decode_image(b64)
    assert max(decoded.size) <= 896, decoded.size


async def _main():
    async_tests = [
        test_resolve_backend_demo_is_hf,
        test_resolve_backend_dev_still_blocked,
        test_call_hf_strips_thinking_block,
        test_call_hf_passthrough_when_no_thinking,
    ]
    sync_tests = [test_strip_thinking_unit, test_decode_image_caps_size]
    failed = 0
    for t in async_tests:
        try:
            await t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    for t in sync_tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    total = len(async_tests) + len(sync_tests)
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
