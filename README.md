<div align="center">

# Medical vLM Microservice

[![License](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker Sandbox](https://img.shields.io/badge/Sandbox-Docker-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

**Production-Ready Asynchronous Multimodal Medical AI Serving Engine with Strategy Dispatch**

[Architecture Overview](#architecture-overview) | [Quick Start](#quick-start)

</div>

## About The Project

**Medical vLM Microservice** is a high-performance, asynchronous RESTful API tailored for automated and systematic medical image analysis (e.g., Chest X-Rays). Built upon a decoupled microservice paradigm, the engine features a pipeline to enforce absolute input modal integrity and clinical-grade image quality bottom-lines before orchestrating vision-language models (vLMs).

### Environment & Processing Matrix (Sample)

<div align="center">
<table style="width:100%">
  <thead>
    <tr>
      <th style="width:20%">Environment (`ENV`)</th>
      <th style="width:25%">Backend Inference Engine</th>
      <th style="width:30%">Target Multi-Modal Models</th>
      <th style="width:25%">Compute Context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>dev</code> / <code>prod</code></td>
      <td>Ollama / vLLM Engine</td>
      <td><code>llama3.2-vision</code> (Native Multi-modal)</td>
      <td>AWS GPU Instances (CUDA Accelerated)</td>
    </tr>
  </tbody>
</table>
</div>

## Architecture Overview

The system architecture is engineered to adhere to enterprise-grade asynchronous design, completely neutralizing blocking I/O overhead during massive graphic payloads:

1. **Ingress & Parsing Layer (`FastAPI` + `python-multipart`)**: Consumes medical binaries seamlessly via non-blocking multi-part form parameters.
2. **Dynamic Strategy Dispatch (`config.py` + `inference.py`)**: Conditionally routes Base64-encoded spatial tensor streams based on unified state configurations.
3. **Clinical Role Binding (Expert System Prompt)**: Constrains LLM autoregressive tokens under a strict 4-tier structured radiology sequence:
   - `Image Quality Assessment` -> `Objective Findings` -> `Impression (Differential Diagnosis)` -> `Recommendations & Disclaimer`.

## Open-Source Software Stack

This project stands on the shoulders of giants within the open-source GenAI ecosystem:
- **[FastAPI](https://github.com/tiangolo/fastapi)** - High-performance, low-latency ASGI web framework for Python.
- **[HTTPX](https://github.com/encode/httpx)** - Next-generation, fully asynchronous HTTP client utilized for internal multi-modal relay communications.
- **[Docker](https://github.com/docker)** - OS-level virtualization to guarantee reproducible sandbox runs.
- **[OpenRouter Labs / Ollama](https://openrouter.ai/)** - Advanced orchestration abstraction layers for unified Large Language Model execution.

## Quick Start

### 1. Environment Configuration
Create a `.env` file in the project root folder to register your runtime credentials:

```ini
ENV=local

# Dev/Prod Architecture (Ollama)
DEV_API_URL=[OLLAMA_API_URL]
DEV_MODEL=[OLLAMA_MODEL]

# Local Sandbox Architecture (OpenRouter)
LOCAL_API_KEY=[your_sk_or_v1_openrouter_api_key_here]
LOCAL_MODEL=openrouter/owl-alpha
```

### 2. Sandbox Deployment (Docker Isolation)
Build and spin up the complete isolated microservice environment natively without affecting host storage parameters:

```bash
# Clone the repository
git clone [https://github.com/your-username/med-vlm-service.git](https://github.com/your-username/med-vlm-service.git)
cd med-vlm-service

# Build the container image securely skipping local cash layers
docker build --no-cache -t med-vlm-sandbox .

# Execute the runtime sandboxed container routing bound ports
docker run -d -p 8000:8000 --env-file .env --name medical_service_agent med-vlm-sandbox
```

### 3. Clinical Inference Verification (curl)
Test the endpoint via terminal to evaluate the system prompt constraints and runtime capabilities:

```bash
curl -X POST [http://127.0.0.1:8000/analyze](http://127.0.0.1:8000/analyze) \
  -F "prompt=Please systematically evaluate this chest radiography for clinical anomalies." \
  -F "image=@./data/test_images/normal_xray.jpg"
```
