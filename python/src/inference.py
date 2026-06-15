import asyncio
import httpx
from config import config

async def _call_dev_model_router(base64_image: str, prompt: str, system_prompt: str) -> str:
    """Dev environment: call local or AWS Ollama engine"""
    if not config.DEV_API_KEY:
        raise Exception(
            "DEV_API_KEY environment variable is not set. "
            "Please set it to your Ollama API key (e.g., export DEV_API_KEY='...')."
        )

    headers = {
        "Authorization": f"Bearer {config.DEV_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.DEV_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt, "images": [base64_image]}
        ],
        "stream": False
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(config.DEV_API_URL, headers=headers, json=payload, timeout=60.0)

        # Check HTTP status code
        if response.status_code != 200:
            error_detail = response.text
            raise Exception(f"Ollama API error (HTTP {response.status_code}): {error_detail}")

        result = response.json()

        # Check response format
        if "message" not in result:
            raise Exception(f"Ollama API response format error, missing 'message' field: {result}")

        return result["message"]["content"]


async def _call_local_model_router(base64_image: str, prompt: str, system_prompt: str) -> str:
    """Local environment: call OpenRouter cloud API to drive sandbox"""
    if not config.LOCAL_API_KEY:
        raise Exception(
            "LOCAL_API_KEY environment variable is not set. "
            "Please set it to your OpenRouter API key (e.g., export LOCAL_API_KEY='sk-or-...')."
        )

    headers = {
        "Authorization": f"Bearer {config.LOCAL_API_KEY}",
        "Content-Type": "application/json"
    }
    # OpenRouter multimodal Vision format
    payload = {
        "model": config.LOCAL_MODEL,
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

    # Implement exponential backoff retry for transient errors like 429/5xx (max 3 attempts)
    max_retries = 3
    response = None
    for attempt in range(1, max_retries + 1):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                config.LOCAL_API_URL,
                headers=headers,
                json=payload,
                timeout=60.0
            )

        # Success → exit retry loop
        if response.status_code == 200:
            break

        # 429/5xx → transient error, retryable
        if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            wait_seconds = 2 ** attempt  # 2s, 4s, 8s
            print(f"[Warning] OpenRouter transient error (HTTP {response.status_code}), "
                  f"retry {attempt}/{max_retries}, waiting {wait_seconds}s ...")
            await asyncio.sleep(wait_seconds)
            continue

        # Non-retryable error (e.g., 400) or max retries reached
        error_detail = response.text
        raise Exception(f"OpenRouter API error (HTTP {response.status_code}): {error_detail}")

    if response is None:
        raise Exception("OpenRouter API error: no response received")

    result = response.json()

    # Check response format
    if "choices" not in result:
        raise Exception(f"OpenRouter API response format error, missing 'choices' field: {result}")

    return result["choices"][0]["message"]["content"]


async def generate_medical_report(base64_image: str, prompt: str, system_prompt: str) -> str:
    if config.ENV == "local":
        return await _call_local_model_router(base64_image, prompt, system_prompt)
    elif config.ENV == "dev":
        return await _call_dev_model_router(base64_image, prompt, system_prompt)
    else:
        raise ValueError(f"[ERROR] Unsupported ENV setting: {config.ENV}")
