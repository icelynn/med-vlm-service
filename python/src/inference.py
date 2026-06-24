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

_hf_cache = {}  # model_id -> (processor, model); loaded lazily, kept resident
_hf_lock = asyncio.Lock()  # one GPU, one generate() at a time


def _decode_image(base64_image: str) -> Image.Image:
    """Base64 → PIL,長邊縮到 896px 避免 vision-token 爆 VRAM。"""
    image = Image.open(BytesIO(base64.b64decode(base64_image))).convert("RGB")
    image.thumbnail((_HF_MAX_IMAGE_SIZE, _HF_MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
    return image


def _strip_thinking(text: str) -> str:
    """防禦性處理：若 checkpoint 真的吐出 <think>...</think>,濾掉只留最終報告。"""
    return _THINK_TAG_RE.sub("", text, count=1).strip()


async def _load_hf_model():
    """惰性載入目前 MODEL_ROLE 對應的 HF 模型,常駐記憶體（同一個 process 只载一次）。

    torch/transformers 在這裡才 import,讓 test/dev 等不需要 HF 的環境不用付這個 import 成本
    (本機量測 torch+transformers import ≈2.6s)。重用 feasibility_check.py 已驗證過的 T4 載入邏輯。
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
    """跟 feasibility_check.run_inference 相同的 chat-template 邏輯（兩個模型通用）。"""
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
    """demo 環境：直接用 HF transformers 跑（重用 eval 已驗證過的載入/推論邏輯）"""
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
        exemplars = await asyncio.to_thread(_retrieve_image_exemplars, tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    logger.info(f"[image-retrieval] exemplars: {[Path(p).name for p, _ in exemplars]}")
    return exemplars


async def _call_hf_image(base64_image: str) -> str:
    """Image-retrieval few-shot RAG (R2, opt-in, see config.IMAGE_RETRIEVAL_RAG):
    reproduces the eval-measured R2-B config (constrained prompt, harmonized
    RSNA pool) -- NOT the demo's free-report format, and does NOT apply the
    report endpoint's modality-gating system prompt. A non-brain image in
    this mode will get a constrained yes/no judgment, not a refusal -- that
    is the documented tradeoff of reproducing the measured config exactly."""
    image = _decode_image(base64_image)
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
    """test 環境：呼叫 OpenRouter 雲端 API"""
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    # OpenRouter 的多模態 Vision 格式
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

    # 針對 429/5xx 等暫時性錯誤，採取指數退避重試 (最多 3 次)
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

        # 成功 → 跳出重試迴圈
        if response.status_code == 200:
            break

        # 429/5xx → 暫時性錯誤，可重試
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            wait_seconds = 2 ** attempt  # 2s, 4s, 8s
            print(f"[警告] OpenRouter 暫時性錯誤 (HTTP {response.status_code})，"
                  f"第 {attempt}/{max_retries} 次重試，等待 {wait_seconds}s ...")
            await asyncio.sleep(wait_seconds)
            continue

        # 不可重試的錯誤 (例如 400) 或已達最大重試次數
        error_detail = response.text
        raise Exception(f"OpenRouter API 錯誤 (HTTP {response.status_code}): {error_detail}")

    result = response.json()

    # 檢查回應格式
    if "choices" not in result:
        raise Exception(f"OpenRouter API 回應格式異常，缺少 'choices' 欄位: {result}")

    return result["choices"][0]["message"]["content"]


def _resolve_service_backend() -> str:
    """回傳目前 ENV 對應的服務後端，並擋掉不該由此 proxy 服務的環境。"""
    if config.ENV == "dev":
        raise ValueError(
            "[錯誤] ENV=dev 是 HF+transformers 評估管線（請用 run_baseline.py），"
            "本服務只服務 test(OpenRouter) 與 demo(HF transformers)。"
        )
    backend = config.backend
    if backend not in ("openrouter", "hf"):
        raise ValueError(f"[錯誤] 不支援的 ENV 設定: {config.ENV}")
    if config.model is None:
        raise ValueError(
            f"[錯誤] 角色 MODEL_ROLE={config.MODEL_ROLE} 在 {backend} 後端上沒有可用模型"
            "（例如 MedGemma 未上架 OpenRouter）。請改用 demo(HF transformers) 或 dev(HF eval)。"
        )
    if backend == "openrouter" and not config.OPENROUTER_API_KEY:
        raise ValueError(
            "[錯誤] ENV=test（OpenRouter）需要 OPENROUTER_API_KEY，但目前未設定。"
            "請在 .env 設定 OPENROUTER_API_KEY。"
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
            "[錯誤] two_stage 與 image_retrieval 不能同時啟用——兩者是互斥的檢索模式。"
        )
    if use_image:
        return "image"
    if use_two_stage:
        return "text-twostage"
    return "none"


async def generate_medical_report(base64_image: str, prompt: str, system_prompt: str,
                                  two_stage: bool | None = None,
                                  image_retrieval: bool | None = None) -> str:
    """策略進入點：依據 config 決定派發哪一個環境流程

    two_stage/image_retrieval 為 None 時讀各自的 config 預設(皆預設 off);
    只在 HF 後端生效——OpenRouter(test)鏡像為選配,目前未實作,兩個旗標在
    該後端下都不生效。
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
    """demo 環境（HF 直跑、串流版）：背景 thread 跑 generate(streamer=...),用 asyncio.Queue 橋接成 async yield"""
    from transformers import TextIteratorStreamer  # 延後 import,避免非 demo 環境也要拉 transformers

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
            loop.call_soon_threadsafe(queue.put_nowait, None)  # 結束哨兵

        threading.Thread(target=_pump, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield piece


async def _stream_hf_twostage(base64_image: str, prompt: str, system_prompt: str):
    """串流版兩階段:stage 1 非串流跑完拿 findings,stage 2 照舊串流。"""
    findings = await _call_hf(base64_image, prompt, system_prompt)
    context = await asyncio.to_thread(_retrieve_text_context, findings)
    enhanced_system_prompt = f"{system_prompt}\n\n{context}" if context else system_prompt
    logger.info(f"[two-stage] stage-1 findings: {findings[:200]!r}")
    logger.info(f"[two-stage] retrieved context ({len(context)} chars): {context[:200]!r}")
    async for token in _stream_hf(base64_image, prompt, enhanced_system_prompt):
        yield token


async def _stream_hf_image(base64_image: str):
    """串流版 R2(影像檢索 few-shot)。檢索在鎖外做完,串流生成在鎖內,
    跟 _call_hf_image 同一套檢索/建構邏輯,只是 generate 換成 streamer。"""
    from transformers import TextIteratorStreamer

    image = _decode_image(base64_image)
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
            loop.call_soon_threadsafe(queue.put_nowait, None)  # 結束哨兵

        threading.Thread(target=_pump, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            yield piece


async def _stream_openrouter(base64_image: str, prompt: str, system_prompt: str):
    """test 環境（串流版）：逐 token 讀取 OpenRouter 的 SSE 回應"""
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
                raise Exception(f"OpenRouter API 錯誤 (HTTP {response.status_code}): {error_detail}")

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # 跳過壞行（截斷/註解/keep-alive），不中斷整串
                choices = chunk.get("choices")
                if not choices:
                    continue  # 心跳/內容過濾 chunk 可能無 choices，跳過而非崩潰
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    yield token


async def generate_medical_report_stream(base64_image: str, prompt: str, system_prompt: str,
                                         two_stage: bool | None = None,
                                         image_retrieval: bool | None = None):
    """串流版策略進入點：依據 config 決定派發哪一個環境流程，逐 token yield"""
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
