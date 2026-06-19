# -*- coding: utf-8 -*-
"""Mock Ollama /api/chat streaming upstream for verifying the service's SSE forwarding.

Emits NDJSON (one JSON object per line) with a small per-token delay so we can observe
that the real service flushes tokens incrementally rather than buffering the whole reply.
Run: uvicorn mock_ollama_stream:app --port 11434
"""
import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

TOKENS = ["No ", "acute ", "intracranial ", "hemorrhage ", "is ", "identified.", ""]


@app.post("/api/chat")
async def chat(request: Request):
    await request.json()  # consume body like the real endpoint

    async def gen():
        for i, tok in enumerate(TOKENS):
            done = i == len(TOKENS) - 1
            yield json.dumps({"message": {"role": "assistant", "content": tok}, "done": done}) + "\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(gen(), media_type="application/x-ndjson")
