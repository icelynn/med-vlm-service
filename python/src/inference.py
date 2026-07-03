import asyncio
import base64
import json
import logging
import re
import sys
import tempfile
import threading
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from config import config

logger = logging.getLogger("medical-vlm-service")

_HF_MAX_IMAGE_SIZE = 896  # vision-token-blowup cap, same lesson as feasibility_check.py
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TWO_STAGE_TEXT_K = 2  # matches run_baseline.py's TEXT_TWOSTAGE_K (< 5 = whole KB)

# R2 (image-retrieval few-shot) constrained prompts -- mirrored verbatim from
# run_baseline.py's SYSTEM_PROMPT/USER_PROMPT (source of truth). R2's measured
# F1=0.667/0.575 used exactly this constrained format, not free-text reports,
# so the demo path reproduces it rather than reusing the user's own prompt.
_R2_SYSTEM_PROMPT = (
    "You are a neuroradiologist reading a single axial head CT slice shown in a "
    "brain window. Judge only what is visible on this one slice."
)
_R2_USER_PROMPT = (
    "Assess this head CT slice for acute intracranial hemorrhage. "
    "Answer in EXACTLY this format and nothing else:\n"
    "HEMORRHAGE: <yes or no>\n"
    "SUBTYPES: <comma-separated list of those present, from IPH, IVH, SAH, EDH, "
    "SDH; or 'none'>\n\n"
    "Where IPH=intraparenchymal, IVH=intraventricular, SAH=subarachnoid, "
    "EDH=epidural, SDH=subdural."
)
_R2_MAX_NEW_TOKENS = 64  # matches run_baseline.py: the constrained answer is short

# R2 modality gate (opt-in path's only safety check -- the constrained R2 prompts
# above have zero refusal logic by design, since that's the exact configuration
# that was measured at F1=0.575/0.667). This is a SEPARATE, minimal generate call,
# not a canned escape-hatch folded into the same prompt as the substantive task --
# a canned-tag refusal channel (`[Error]`) inside the *same* generation was found to
# over-trigger even on valid images. A plain yes/no question in its own call doesn't
# carry that failure mode, but to stay safe it still fails OPEN (treats anything not
# starting with "NO" as a pass) rather than blocking on an ambiguous answer.
_GATE_SYSTEM_PROMPT = "You are a radiology image triage assistant."
_GATE_USER_PROMPT = (
    "Is this image a single axial head CT slice (a brain CT)? "
    "Default to YES: judge only the imaging modality and rough anatomical "
    "region, not image quality, contrast, or how much brain tissue is "
    "visible. A dim or low-contrast slice is still YES. A slice near the top "
    "of the skull (vertex, mostly skull/scalp) or near the skull base "
    "(sinuses, orbits, mastoid bone) is still YES -- any axial CT slice "
    "anywhere within the head, however little brain tissue it shows, is "
    "YES. Answer NO only if you are confident the image is a different body "
    "part, a different imaging modality (e.g. a photo, X-ray, or MRI), or "
    "not a medical scan at all. Answer with exactly one word: YES or NO."
)
_GATE_MAX_NEW_TOKENS = 8
_GATE_REFUSAL_MESSAGE = (
    "This image does not appear to be a head CT slice, so the image-retrieval "
    "hemorrhage assessment was not performed."
)

_hf_cache = {}  # model_id -> (processor, model); loaded lazily, kept resident
_hf_lock = asyncio.Lock()  # one GPU, one generate() at a time
_retrieval_lock = asyncio.Lock()  # providers.py/biomedclip_embed.py's lazy singletons
                                  # (BiomedCLIP model, image index) are plain module
                                  # globals with no thread-safety; concurrent requests
                                  # hitting the cold cache via asyncio.to_thread would
                                  # race to populate them. Separate from _hf_lock so
                                  # CPU retrieval never blocks GPU generation.


def _decode_image(base64_image: str) -> Image.Image:
    """Base64 -> PIL, long edge capped to 896px to avoid a vision-token VRAM blowup."""
    image = Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")
    image.thumbnail((_HF_MAX_IMAGE_SIZE, _HF_MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
    return image


def _strip_thinking(text: str) -> str:
    """Defensive: strip <think>...</think> if the checkpoint actually emits it,
    keeping only the final report."""
    return _THINK_TAG_RE.sub("", text, count=1).strip()


async def _load_hf_model():
    """Lazily load the HF model for the current MODEL_ROLE, kept resident
    (loaded once per process).

    torch/transformers are imported here so environments that don't need HF
    (e.g. test/dev) don't pay that import cost (measured locally at ~2.6s).
    Reuses feasibility_check.py's already-validated T4 loading logic.
    """
    model_id = config.model
    if model_id not in _hf_cache:
        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from feasibility_check import load_model, resolve_dtype

        dtype = resolve_dtype(model_id, "auto")
        processor, model, _ = await asyncio.to_thread(load_model, model_id, "none", dtype)
        _hf_cache[model_id] = (processor, model)
    return _hf_cache[model_id]


def _build_hf_inputs(processor, model, image: Image.Image, prompt: str, system_prompt: str):
    """Same chat-template logic as feasibility_check.run_inference (shared by both models)."""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


async def _call_hf(base64_image: str, prompt: str, system_prompt: str) -> str:
    """demo environment: runs directly via HF transformers (reuses eval's
    validated load/inference logic)."""
    image = _decode_image(base64_image)
    async with _hf_lock:
        processor, model = await _load_hf_model()
        inputs = _build_hf_inputs(processor, model, image, prompt, system_prompt)
        input_len = inputs["input_ids"].shape[-1]

        def _generate():
            import torch
            with torch.inference_mode():
                return model.generate(**inputs, max_new_tokens=1536, do_sample=False)

        out = await asyncio.to_thread(_generate)

    new_tokens = out[0][input_len:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return _strip_thinking(text)


def _retrieve_text_context(findings: str) -> str:
    """Stage-2 retrieval: condition build_context's query on stage-1 findings
    instead of the fixed DEFAULT_QUERY (see python/rag/providers.py)."""
    rag_dir = Path(__file__).resolve().parent.parent / "rag"
    if str(rag_dir) not in sys.path:
        sys.path.insert(0, str(rag_dir))
    from providers import build_context

    return build_context("text", query=findings, text_k=_TWO_STAGE_TEXT_K)


async def _call_hf_twostage(base64_image: str, prompt: str, system_prompt: str) -> str:
    """Findings-conditioned two-stage RAG (opt-in, see config.TWO_STAGE_RAG):
    stage 1 runs the normal single-stage call to get findings, stage 2 retrieves
    against those findings and reruns with the enhanced system prompt."""
    findings = await _call_hf(base64_image, prompt, system_prompt)
    context = await asyncio.to_thread(_retrieve_text_context, findings)
    enhanced_system_prompt = f"{system_prompt}\n\n{context}" if context else system_prompt
    logger.info(f"[two-stage] stage-1 findings: {findings[:200]!r}")
    logger.info(f"[two-stage] retrieved context ({len(context)} chars): {context[:200]!r}")
    return await _call_hf(base64_image, prompt, enhanced_system_prompt)


def _retrieve_image_exemplars(query_image_path: str):
    """R2 retrieval: BiomedCLIP top-3 visually-similar pool exemplars for the
    query image (see python/rag/providers.py). Pure CPU (BiomedCLIP + numpy),
    called before the GPU lock is acquired so it never blocks generation."""
    rag_dir = Path(__file__).resolve().parent.parent / "rag"
    if str(rag_dir) not in sys.path:
        sys.path.insert(0, str(rag_dir))
    from providers import build_context

    return build_context("image", image_path=query_image_path,
                         image_index_dir=config.R2_INDEX_DIR,
                         image_pool_dir=config.R2_POOL_DIR)


def _build_fewshot_inputs(processor, model, query_image: Image.Image, exemplars):
    """Mirrors feasibility_check.run_inference_fewshot's message structure:
    each exemplar becomes a user-image-turn + assistant-answer-turn
    demonstrating the answer grammar, then the real query as a final user
    turn. Always uses the R2 constrained prompts (not the caller's prompt) --
    see _R2_SYSTEM_PROMPT/_R2_USER_PROMPT docstring for why."""
    from feasibility_check import _load_capped_image

    messages = [
        {"role": "system", "content": [{"type": "text", "text": _R2_SYSTEM_PROMPT}]},
    ]
    for ex_path, ex_label in exemplars:
        ex_image = _load_capped_image(ex_path, _HF_MAX_IMAGE_SIZE)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": ex_image},
                {"type": "text", "text": _R2_USER_PROMPT},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex_label}],
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": query_image},
            {"type": "text", "text": _R2_USER_PROMPT},
        ],
    })
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


async def _get_image_exemplars(image: Image.Image):
    """Write the query image to a temp file (retrieval needs a path, not a
    PIL object -- see biomedclip_embed.embed_image), retrieve, then clean up
    immediately; the temp file's only job is the retrieval step."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        async with _retrieval_lock:
            exemplars = await asyncio.to_thread(_retrieve_image_exemplars, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    logger.info(f"[image-retrieval] exemplars: {[Path(p).name for p, _ in exemplars]}")
    return exemplars


async def _passes_head_ct_gate(processor, model, image: Image.Image) -> bool:
    """Lightweight modality pre-check for the R2 (image-retrieval) path, which
    otherwise has no gate at all (its measured 0.575/0.667 F1 used a fully
    constrained prompt with zero refusal logic, see _R2_SYSTEM_PROMPT above).
    Caller must already hold _hf_lock and have a loaded processor/model."""
    inputs = _build_hf_inputs(processor, model, image, _GATE_USER_PROMPT, _GATE_SYSTEM_PROMPT)
    input_len = inputs["input_ids"].shape[-1]

    def _generate():
        import torch
        with torch.inference_mode():
            return model.generate(**inputs, max_new_tokens=_GATE_MAX_NEW_TOKENS, do_sample=False)

    out = await asyncio.to_thread(_generate)
    answer = processor.decode(out[0][input_len:], skip_special_tokens=True).strip().upper()
    return not answer.startswith("NO")


async def _call_hf_image(base64_image: str) -> str:
    """Image-retrieval few-shot RAG (R2, opt-in, see config.IMAGE_RETRIEVAL_RAG):
    reproduces the eval-measured R2-B config (constrained prompt, harmonized
    RSNA pool) for the substantive judgment itself. A separate modality gate
    (_passes_head_ct_gate) runs first so this mode doesn't have to choose
    between reproducing the measured config exactly and refusing non-head-CT
    input -- the gate is a distinct call, decoupled from the constrained
    few-shot prompt that produced the measured F1."""
    image = _decode_image(base64_image)
    async with _hf_lock:
        processor, model = await _load_hf_model()
        if not await _passes_head_ct_gate(processor, model, image):
            return _GATE_REFUSAL_MESSAGE

    exemplars = await _get_image_exemplars(image)

    async with _hf_lock:
        processor, model = await _load_hf_model()
        inputs = _build_fewshot_inputs(processor, model, image, exemplars)
        input_len = inputs["input_ids"].shape[-1]

        def _generate():
            import torch
            with torch.inference_mode():
                return model.generate(**inputs, max_new_tokens=_R2_MAX_NEW_TOKENS,
                                      do_sample=False)

        out = await asyncio.to_thread(_generate)

    new_tokens = out[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


async def _call_openrouter(base64_image: str, prompt: str, system_prompt: str) -> str:
    """test environment: calls the OpenRouter cloud API."""
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    # OpenRouter's multimodal vision format
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    }

    # Exponential backoff retry for transient 429/5xx errors (up to 3 attempts)
    max_retries = 3
    response = None
    for attempt in range(1, max_retries + 1):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config.OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=60.0
            )

        # Success -> break out of the retry loop
        if response.status_code == 200:
            break

        # 429/5xx -> transient error, retryable
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            wait_seconds = 2 ** attempt  # 2s, 4s, 8s
            print(f"[warning] OpenRouter transient error (HTTP {response.status_code}), "
                  f"retry {attempt}/{max_retries}, waiting {wait_seconds}s ...")
            await asyncio.sleep(wait_seconds)
            continue

        # Non-retryable error (e.g. 400) or max retries reached
        error_detail = response.text
        raise Exception(f"OpenRouter API error (HTTP {response.status_code}): {error_detail}")

    result = response.json()

    # Validate response shape
    if "choices" not in result:
        raise Exception(f"OpenRouter API response missing 'choices' field: {result}")

    return result["choices"][0]["message"]["content"]


def _resolve_service_backend() -> str:
    """Return the service backend for the current ENV, rejecting environments
    this proxy shouldn't serve."""
    if config.ENV == "dev":
        raise ValueError(
            "[error] ENV=dev is the HF+transformers eval pipeline (use run_baseline.py); "
            "this service only serves test (OpenRouter) and demo (HF transformers)."
        )
    backend = config.backend
    if backend not in ("openrouter", "hf"):
        raise ValueError(f"[error] unsupported ENV setting: {config.ENV}")
    if config.model is None:
        raise ValueError(
            f"[error] MODEL_ROLE={config.MODEL_ROLE} has no model on the {backend} backend "
            "(e.g. MedGemma isn't on OpenRouter). Use demo (HF transformers) or dev (HF eval) instead."
        )
    if backend == "openrouter" and not config.OPENROUTER_API_KEY:
        raise ValueError(
            "[error] ENV=test (OpenRouter) requires OPENROUTER_API_KEY, which is not set. "
            "Set OPENROUTER_API_KEY in .env."
        )
    return backend


def _resolve_retrieval_mode(two_stage: bool | None, image_retrieval: bool | None) -> str:
    """Resolve which HF-backend retrieval mode applies. Each flag falls back
    to its own config default when not explicitly given (None); if both
    resolve to True, fail fast instead of silently picking a precedence --
    the two modes are mutually exclusive (different prompts, different
    message structure)."""
    use_two_stage = config.TWO_STAGE_RAG if two_stage is None else two_stage
    use_image = config.IMAGE_RETRIEVAL_RAG if image_retrieval is None else image_retrieval
    if use_two_stage and use_image:
        raise ValueError(
            "[error] two_stage and image_retrieval cannot both be enabled -- "
            "they are mutually exclusive retrieval modes."
        )
    if use_image:
        return "image"
    if use_two_stage:
        return "text-twostage"
    return "none"


async def generate_medical_report(base64_image: str, prompt: str, system_prompt: str,
                                  two_stage: bool | None = None,
                                  image_retrieval: bool | None = None) -> str:
    """Strategy entry point: dispatches to the right environment flow based on config.

    two_stage/image_retrieval fall back to their own config defaults when None
    (both default off); only take effect on the HF backend -- an OpenRouter
    (test) mirror is optional and not implemented yet, so both flags are
    no-ops on that backend.
    """
    backend = _resolve_service_backend()
    if backend == "openrouter":
        return await _call_openrouter(base64_image, prompt, system_prompt)
    mode = _resolve_retrieval_mode(two_stage, image_retrieval)
    if mode == "image":
        return await _call_hf_image(base64_image)
    if mode == "text-twostage":
        return await _call_hf_twostage(base64_image, prompt, system_prompt)
    return await _call_hf(base64_image, prompt, system_prompt)


async def _stream_hf(base64_image: str, prompt: str, system_prompt: str):
    """demo environment (direct HF run, streaming): runs generate(streamer=...)
    on a background thread, bridged to async yield via asyncio.Queue."""
    from transformers import TextIteratorStreamer  # deferred import so non-demo environments don't pay the transformers cost

    image = _decode_image(base64_image)
    async with _hf_lock:
        processor, model = await _load_hf_model()
        inputs = _build_hf_inputs(processor, model, image, prompt, system_prompt)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def _generate():
            import torch
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=1536, do_sample=False, streamer=streamer)

        threading.Thread(target=_generate, daemon=True).start()

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def _pump():
            for piece in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, piece)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # end-of-stream sentinel

        threading.Thread(target=_pump, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield piece


async def _stream_hf_twostage(base64_image: str, prompt: str, system_prompt: str):
    """Streaming two-stage: stage 1 runs non-streamed to get findings, stage 2
    streams as usual."""
    findings = await _call_hf(base64_image, prompt, system_prompt)
    context = await asyncio.to_thread(_retrieve_text_context, findings)
    enhanced_system_prompt = f"{system_prompt}\n\n{context}" if context else system_prompt
    logger.info(f"[two-stage] stage-1 findings: {findings[:200]!r}")
    logger.info(f"[two-stage] retrieved context ({len(context)} chars): {context[:200]!r}")
    async for token in _stream_hf(base64_image, prompt, enhanced_system_prompt):
        yield token


async def _stream_hf_image(base64_image: str):
    """Streaming R2 (image-retrieval few-shot). Same modality gate as
    _call_hf_image (a failed gate yields the refusal message directly,
    skipping retrieval/generation); once the gate passes, retrieval runs
    outside the lock and streaming generation runs inside it -- same
    retrieval/build logic as _call_hf_image, just generate swapped for a
    streamer."""
    from transformers import TextIteratorStreamer

    image = _decode_image(base64_image)
    async with _hf_lock:
        processor, model = await _load_hf_model()
        if not await _passes_head_ct_gate(processor, model, image):
            yield _GATE_REFUSAL_MESSAGE
            return

    exemplars = await _get_image_exemplars(image)

    async with _hf_lock:
        processor, model = await _load_hf_model()
        inputs = _build_fewshot_inputs(processor, model, image, exemplars)
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)

        def _generate():
            import torch
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=_R2_MAX_NEW_TOKENS,
                               do_sample=False, streamer=streamer)

        threading.Thread(target=_generate, daemon=True).start()

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def _pump():
            for piece in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, piece)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # end-of-stream sentinel

        threading.Thread(target=_pump, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield piece


async def _stream_openrouter(base64_image: str, prompt: str, system_prompt: str):
    """test environment (streaming): reads OpenRouter's SSE response token by token."""
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "stream": True
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        async with client.stream("POST", config.OPENROUTER_API_URL, headers=headers, json=payload) as response:
            if response.status_code != 200:
                error_detail = (await response.aread()).decode("utf-8", "replace")
                raise Exception(f"OpenRouter API error (HTTP {response.status_code}): {error_detail}")

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # skip malformed lines (truncation/comment/keep-alive) without breaking the stream
                choices = chunk.get("choices")
                if not choices:
                    continue  # heartbeat/content-filter chunks may have no choices; skip rather than crash
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token


async def generate_medical_report_stream(base64_image: str, prompt: str, system_prompt: str,
                                         two_stage: bool | None = None,
                                         image_retrieval: bool | None = None):
    """Streaming strategy entry point: dispatches by config, yielding token by token."""
    backend = _resolve_service_backend()
    if backend == "openrouter":
        gen = _stream_openrouter(base64_image, prompt, system_prompt)
    else:
        mode = _resolve_retrieval_mode(two_stage, image_retrieval)
        if mode == "image":
            gen = _stream_hf_image(base64_image)
        elif mode == "text-twostage":
            gen = _stream_hf_twostage(base64_image, prompt, system_prompt)
        else:
            gen = _stream_hf(base64_image, prompt, system_prompt)
    async for token in gen:
        yield token
