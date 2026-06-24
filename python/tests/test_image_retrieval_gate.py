# -*- coding: utf-8 -*-
"""Offline tests for the R2 (image-retrieval) modality gate (_passes_head_ct_gate,
wired into _call_hf_image / _stream_hf_image). The model/processor and the
exemplar-retrieval step are mocked, so no real model weights, BiomedCLIP index,
or GPU is needed.

Run:  ./.venv/Scripts/python.exe python/tests/test_image_retrieval_gate.py
"""
import asyncio
import base64
import os
import sys
import tempfile
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))  # _build_fewshot_inputs imports feasibility_check
os.environ.setdefault("ENV", "demo")
os.environ.setdefault("MODEL_ROLE", "main")

from PIL import Image

import inference  # noqa: E402


def _tiny_png_b64(size=(4, 4)):
    img = Image.new("RGB", size, color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _tiny_png_file():
    """A real on-disk PNG, since _build_fewshot_inputs opens exemplar paths
    via PIL (feasibility_check._load_capped_image), not from bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.new("RGB", (4, 4), color=(5, 5, 5)).save(tmp.name)
    return tmp.name


# --- fakes: no real model weights involved -----------------------------------
class _FakeTensor:
    def __init__(self, n):
        self.shape = (1, n)


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    """decode() pops one canned answer per generate() call, in call order --
    e.g. [gate_answer, fewshot_answer] for a gate-pass round trip."""
    tokenizer = object()

    def __init__(self, outputs):
        self._outputs = list(outputs)

    def apply_chat_template(self, messages, **kw):
        assert messages[0]["content"][0]["text"], "system prompt missing from messages"
        return _FakeInputs(input_ids=_FakeTensor(10))

    def decode(self, tokens, **kw):
        return self._outputs.pop(0)


class _FakeModel:
    device = "cpu"

    def generate(self, **kw):
        import torch
        return torch.zeros((1, 15), dtype=torch.long)


def _set_fake_model(outputs):
    fake_processor = _FakeProcessor(outputs)
    inference._hf_cache[inference.config.model] = (fake_processor, _FakeModel())


def _patch_exemplars(exemplar_path):
    """Bypass real BiomedCLIP retrieval -- the gate logic doesn't care what the
    exemplars are, only whether retrieval was attempted at all."""
    calls = {"count": 0}

    async def _fake(image):
        calls["count"] += 1
        return [(exemplar_path, "HEMORRHAGE: no\nSUBTYPES: none")]

    original = inference._get_image_exemplars
    inference._get_image_exemplars = _fake
    return calls, original


# --- tests --------------------------------------------------------------------
async def test_gate_fail_blocks_before_retrieval():
    """A confident 'NO' from the gate must refuse without ever calling exemplar
    retrieval (the CPU/BiomedCLIP step) or the few-shot generate."""
    _set_fake_model(["NO"])
    calls, original = _patch_exemplars("unused.png")
    try:
        text = await inference._call_hf_image(_tiny_png_b64())
    finally:
        inference._hf_cache.clear()
        inference._get_image_exemplars = original
    assert text == inference._GATE_REFUSAL_MESSAGE, repr(text)
    assert calls["count"] == 0, "exemplar retrieval must not run when the gate fails"


async def test_gate_pass_reaches_fewshot_answer():
    """A 'YES' gate answer must proceed to retrieval + the few-shot generate,
    and return that generate's output, not the refusal message."""
    exemplar_path = _tiny_png_file()
    _set_fake_model(["YES", "HEMORRHAGE: no\nSUBTYPES: none"])
    calls, original = _patch_exemplars(exemplar_path)
    try:
        text = await inference._call_hf_image(_tiny_png_b64())
    finally:
        inference._hf_cache.clear()
        inference._get_image_exemplars = original
        os.unlink(exemplar_path)
    assert text == "HEMORRHAGE: no\nSUBTYPES: none", repr(text)
    assert calls["count"] == 1, "exemplar retrieval must run exactly once when the gate passes"


async def test_gate_fails_open_on_ambiguous_answer():
    """Anything not starting with 'NO' must pass through (fail-open), matching
    the project's established anti-over-refusal philosophy (core-challenges #9)."""
    exemplar_path = _tiny_png_file()
    _set_fake_model(["Yes, this is a head CT.", "HEMORRHAGE: yes\nSUBTYPES: IPH"])
    calls, original = _patch_exemplars(exemplar_path)
    try:
        text = await inference._call_hf_image(_tiny_png_b64())
    finally:
        inference._hf_cache.clear()
        inference._get_image_exemplars = original
        os.unlink(exemplar_path)
    assert text == "HEMORRHAGE: yes\nSUBTYPES: IPH", repr(text)
    assert calls["count"] == 1


async def test_stream_gate_fail_yields_refusal_only():
    """Streaming variant: gate failure must yield exactly the refusal message
    and stop, never touching exemplar retrieval or TextIteratorStreamer."""
    _set_fake_model(["NO"])
    calls, original = _patch_exemplars("unused.png")
    pieces = []
    try:
        async for piece in inference._stream_hf_image(_tiny_png_b64()):
            pieces.append(piece)
    finally:
        inference._hf_cache.clear()
        inference._get_image_exemplars = original
    assert pieces == [inference._GATE_REFUSAL_MESSAGE], pieces
    assert calls["count"] == 0


async def _main():
    async_tests = [
        test_gate_fail_blocks_before_retrieval,
        test_gate_pass_reaches_fewshot_answer,
        test_gate_fails_open_on_ambiguous_answer,
        test_stream_gate_fail_yields_refusal_only,
    ]
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
    print(f"\n{len(async_tests) - failed}/{len(async_tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
