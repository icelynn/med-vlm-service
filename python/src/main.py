# -*- coding: utf-8 -*-
"""
Project: Multimodal Medical AI Microservice - End-to-End Inference Engine
Tech Stack: FastAPI + HTTPX + HF transformers / OpenRouter (vision-language models)

This service provides a secure RESTful API endpoint that accepts user-uploaded medical
images (e.g., X-ray, CT, MRI) and diagnostic prompts, and asynchronously interacts with the
inference backend to produce multimodal diagnostic reports. Backends: HF transformers
(demo, in-process on GPU) or OpenRouter (test). Ollama was retired 2026-06-21.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from inference import generate_medical_report, generate_medical_report_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-vlm-service")

app = FastAPI(
	title="Multimodal Medical AI Image Dialogue Microservice (MVP)",
	description="Low-latency multimodal medical image diagnosis API (Qwen3-VL / MedGemma via HF transformers or OpenRouter)",
	version="1.0.0"
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # In production, domains should be strictly restricted
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Medical-specific system prompt (constrains model behavior); kept in its own file so it
# can be edited/versioned without touching code.
SYSTEM_PROMPT = (Path(__file__).resolve().parent / "system_prompt.txt").read_text(encoding="utf-8")

@app.post("/analyze")
async def analyze_medical_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    two_stage: Optional[bool] = Form(None),
    image_retrieval: Optional[bool] = Form(None),
):
    try:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        logger.info("Dispatching multimodal core for medical image report generation...")
        report_content = await generate_medical_report(base64_image, prompt, SYSTEM_PROMPT,
                                                        two_stage=two_stage,
                                                        image_retrieval=image_retrieval)
        return {"report": report_content}

    except Exception as e:
        logger.error(f"Unexpected error in inference pipeline: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/analyze/stream")
async def analyze_medical_image_stream(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    two_stage: Optional[bool] = Form(None),
    image_retrieval: Optional[bool] = Form(None),
):
    """SSE variant of /analyze: streams the report token-by-token as text/event-stream.

    Each event is a JSON payload {"delta": "<text>"}; the stream ends with [DONE].
    Errors mid-stream are emitted as {"error": "..."} (HTTP status is already 200 by then).
    """
    image_bytes = await image.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    async def event_generator():
        logger.info("Dispatching streaming multimodal inference...")
        try:
            async for token in generate_medical_report_stream(base64_image, prompt, SYSTEM_PROMPT,
                                                               two_stage=two_stage,
                                                               image_retrieval=image_retrieval):
                yield f"data: {json.dumps({'delta': token})}\n\n"
        except Exception as e:
            logger.error(f"Streaming inference failed: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
