# -*- coding: utf-8 -*-
"""Offline, zero-cost tests for the SSE streaming service.

No GPU, no Ollama, no paid API calls. The upstream model is faked at the httpx
layer (for the router-parsing tests) and at the dispatch layer (for the endpoint
framing tests), so these run anywhere and are deterministic.

Run:  ./.venv/Scripts/python.exe python/tests/test_streaming.py
"""
import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("ENV", "test")
os.environ.setdefault("MODEL_ROLE", "main")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy")
os.environ.setdefault("OPENROUTER_API_URL", "http://upstream/v1/chat")
os.environ.setdefault("OLLAMA_API_URL", "http://upstream/api/chat")

import inference  # noqa: E402
import main  # noqa: E402


# --- fake upstream at the httpx layer ---------------------------------------
class _FakeStream:
    """Stands in for both `client.stream(...)` ctx and the response object."""

    def __init__(self, lines, status=200):
        self._lines, self.status_code = lines, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b"upstream boom"


class _FakeClient:
    def __init__(self, lines, status=200):
        self._lines, self._status = lines, status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, *a, **k):
        return _FakeStream(self._lines, self._status)


def _patch_httpx(lines, status=200):
    inference.httpx.AsyncClient = lambda *a, **k: _FakeClient(lines, status)


_REAL_CLIENT = httpx.AsyncClient


def _restore_httpx():
    inference.httpx.AsyncClient = _REAL_CLIENT


# --- assertions --------------------------------------------------------------
async def test_dev_router_parses_ndjson():
    """Ollama NDJSON lines -> ordered token stream, stops at done=true."""
    lines = [
        '{"message":{"content":"No "},"done":false}',
        '{"message":{"content":"hemorrhage."},"done":false}',
        "",  # blank line must be skipped
        '{"message":{"content":""},"done":true}',
    ]
    _patch_httpx(lines)
    try:
        toks = [t async for t in inference._stream_ollama("b64", "p", "s")]
    finally:
        _restore_httpx()
    assert toks == ["No ", "hemorrhage."], toks


async def test_local_router_parses_sse():
    """OpenRouter SSE 'data:' lines -> ordered token stream, stops at [DONE]."""
    lines = [
        'data: {"choices":[{"delta":{"content":"No "}}]}',
        ": openrouter keep-alive comment",  # non-data line must be skipped
        'data: {"choices":[{"delta":{"content":"hemorrhage."}}]}',
        "data: [DONE]",
    ]
    _patch_httpx(lines)
    try:
        toks = [t async for t in inference._stream_openrouter("b64", "p", "s")]
    finally:
        _restore_httpx()
    assert toks == ["No ", "hemorrhage."], toks


async def test_upstream_error_raises():
    """Non-200 upstream surfaces as an exception (so the endpoint can report it)."""
    _patch_httpx([], status=500)
    try:
        try:
            async for _ in inference._stream_ollama("b64", "p", "s"):
                pass
            raised = False
        except Exception as e:
            raised = "500" in str(e)
    finally:
        _restore_httpx()
    assert raised, "expected a 500 error to propagate"


async def test_ollama_skips_malformed_json():
    """A truncated / non-JSON NDJSON line is skipped, not fatal to the stream."""
    lines = [
        '{"message":{"content":"No "},"done":false}',
        '{"message":{"content":"hemo',  # truncated JSON (TCP boundary) -> skipped
        '{"message":{"content":"rrhage."},"done":false}',
        '{"message":{"content":""},"done":true}',
    ]
    _patch_httpx(lines)
    try:
        toks = [t async for t in inference._stream_ollama("b64", "p", "s")]
    finally:
        _restore_httpx()
    assert toks == ["No ", "rrhage."], toks


async def test_openrouter_skips_malformed_and_choiceless():
    """SSE chunks that are malformed JSON or lack 'choices' are skipped, not fatal."""
    lines = [
        'data: {"choices":[{"delta":{"content":"No "}}]}',
        'data: {"id":"x","choices":[]}',          # heartbeat / empty choices -> skipped
        'data: {"truncated',                       # malformed JSON -> skipped
        'data: {"choices":[{"delta":{"content":"hemorrhage."}}]}',
        "data: [DONE]",
    ]
    _patch_httpx(lines)
    try:
        toks = [t async for t in inference._stream_openrouter("b64", "p", "s")]
    finally:
        _restore_httpx()
    assert toks == ["No ", "hemorrhage."], toks


async def test_missing_openrouter_key_clear_error():
    """ENV=test with an empty OPENROUTER_API_KEY raises a clear local error, not Bearer ''."""
    orig_key = inference.config.OPENROUTER_API_KEY
    inference.config.OPENROUTER_API_KEY = ""
    try:
        raised = False
        try:
            await inference.generate_medical_report("b64", "p", "s")
        except ValueError as e:
            raised = "OPENROUTER_API_KEY" in str(e)
    finally:
        inference.config.OPENROUTER_API_KEY = orig_key
    assert raised, "expected a clear OPENROUTER_API_KEY-not-set ValueError"


async def _post_stream():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.post(
            "/analyze/stream",
            data={"prompt": "hi"},
            files={"image": ("x.png", b"x", "image/png")},
        )
        return r


async def test_endpoint_sse_framing():
    """Endpoint wraps each token as `data: {"delta": ...}` and ends with [DONE]."""
    async def fake_gen(*a, **k):
        for t in ["A", "B"]:
            yield t

    orig = main.generate_medical_report_stream
    main.generate_medical_report_stream = fake_gen
    try:
        r = await _post_stream()
    finally:
        main.generate_medical_report_stream = orig
    assert r.headers["content-type"].startswith("text/event-stream"), r.headers
    assert r.text == (
        'data: {"delta": "A"}\n\n'
        'data: {"delta": "B"}\n\n'
        "data: [DONE]\n\n"
    ), repr(r.text)


async def test_endpoint_error_then_done():
    """A mid-stream upstream failure is emitted as {"error": ...} then [DONE]."""
    async def boom_gen(*a, **k):
        raise Exception("upstream down")
        yield  # pragma: no cover

    orig = main.generate_medical_report_stream
    main.generate_medical_report_stream = boom_gen
    try:
        r = await _post_stream()
    finally:
        main.generate_medical_report_stream = orig
    assert '"error"' in r.text and "upstream down" in r.text, repr(r.text)
    assert r.text.endswith("data: [DONE]\n\n"), repr(r.text)


async def _main():
    tests = [
        test_dev_router_parses_ndjson,
        test_local_router_parses_sse,
        test_upstream_error_raises,
        test_ollama_skips_malformed_json,
        test_openrouter_skips_malformed_and_choiceless,
        test_missing_openrouter_key_clear_error,
        test_endpoint_sse_framing,
        test_endpoint_error_then_done,
    ]
    failed = 0
    for t in tests:
        try:
            await t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
