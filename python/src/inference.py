import asyncio
import json
import httpx
from config import config

async def _call_ollama(base64_image: str, prompt: str, system_prompt: str) -> str:
    """demo 環境：呼叫 EC2 上的 Ollama 引擎"""
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt, "images": [base64_image]}
        ],
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(config.OLLAMA_API_URL, json=payload, timeout=60.0)

        # 檢查 HTTP 狀態碼
        if response.status_code != 200:
            error_detail = response.text
            raise Exception(f"Ollama API 錯誤 (HTTP {response.status_code}): {error_detail}")

        result = response.json()

        # 檢查回應格式
        if "message" not in result:
            raise Exception(f"Ollama API 回應格式異常，缺少 'message' 欄位: {result}")

        return result["message"]["content"]


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
    backend = config.backend
    if backend == "hf":
        raise ValueError(
            f"[錯誤] ENV={config.ENV} 是 HF+transformers 評估管線（請用 run_baseline.py），"
            "本服務只服務 test(OpenRouter) 與 demo(Ollama)。"
        )
    if backend not in ("openrouter", "ollama"):
        raise ValueError(f"[錯誤] 不支援的 ENV 設定: {config.ENV}")
    if config.model is None:
        raise ValueError(
            f"[錯誤] 角色 MODEL_ROLE={config.MODEL_ROLE} 在 {backend} 後端上沒有可用模型"
            "（例如 MedGemma 未上架 OpenRouter）。請改用 demo(Ollama) 或 dev(HF eval)。"
        )
    return backend


async def generate_medical_report(base64_image: str, prompt: str, system_prompt: str) -> str:
    """策略進入點：依據 config 決定派發哪一個環境流程"""
    backend = _resolve_service_backend()
    if backend == "openrouter":
        return await _call_openrouter(base64_image, prompt, system_prompt)
    return await _call_ollama(base64_image, prompt, system_prompt)


async def _stream_ollama(base64_image: str, prompt: str, system_prompt: str):
    """demo 環境（串流版）：逐 token 讀取 Ollama 的 NDJSON 回應"""
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt, "images": [base64_image]}
        ],
        "stream": True
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=None)) as client:
        async with client.stream("POST", config.OLLAMA_API_URL, json=payload) as response:
            if response.status_code != 200:
                error_detail = (await response.aread()).decode("utf-8", "replace")
                raise Exception(f"Ollama API 錯誤 (HTTP {response.status_code}): {error_detail}")

            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break


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
                chunk = json.loads(data)
                token = chunk["choices"][0].get("delta", {}).get("content", "")
                if token:
                    yield token


async def generate_medical_report_stream(base64_image: str, prompt: str, system_prompt: str):
    """串流版策略進入點：依據 config 決定派發哪一個環境流程，逐 token yield"""
    backend = _resolve_service_backend()
    if backend == "openrouter":
        gen = _stream_openrouter(base64_image, prompt, system_prompt)
    else:
        gen = _stream_ollama(base64_image, prompt, system_prompt)
    async for token in gen:
        yield token
