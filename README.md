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
      <td><code>test</code></td>
      <td>OpenRouter (Cloud API)</td>
      <td>Main vLM (vision slug for local testing)</td>
      <td>Local dev machine</td>
    </tr>
    <tr>
      <td><code>dev</code></td>
      <td>HF + Transformers (eval pipeline)</td>
      <td><code>Qwen3-VL-4B</code> (main) / <code>MedGemma-4B</code> (medical baseline)</td>
      <td>AWS GPU Instance (CUDA Accelerated)</td>
    </tr>
    <tr>
      <td><code>demo</code></td>
      <td>HF + Transformers (same engine as <code>dev</code>)</td>
      <td><code>Qwen3-VL-4B</code> (main) / <code>MedGemma-4B</code> (medical baseline)</td>
      <td>AWS GPU Instance (CUDA Accelerated)</td>
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
- **[OpenRouter Labs](https://openrouter.ai/)** - Unified cloud API orchestration layer used by the `test` environment.
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Local GPU inference engine shared by the `dev` (evaluation) and `demo` (served) environments.

## Quick Start

### 1. Environment Configuration
Create a `.env` file in the project root folder to register your runtime credentials:

```ini
ENV=test                 # test | dev | demo
MODEL_ROLE=main          # main | medical_baseline

# test — OpenRouter (cloud API, local dev machine)
OPENROUTER_API_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_API_KEY=[your_sk_or_v1_openrouter_api_key_here]
# OPENROUTER_MAIN_MODEL defaults in config.py

# demo (served) and dev (eval pipeline) — both run HF + Transformers on the
# AWS GPU instance; HF model ids default in config.py (HF_MAIN_MODEL /
# HF_MEDICAL_MODEL override Qwen3-VL-4B / MedGemma-4B if set)
```

> Backends are split by purpose: `dev` (HF + Transformers) is the **evaluation** track (`python/eval/run_baseline.py`) and produces all reported metrics; `test`/`demo` are the **served** proxy. Model identities live in one place — `python/src/config.py` (`MODELS` registry).

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

## Evaluation Results

The `dev` track (`python/eval/`) runs a controlled, reproducible evaluation protocol on the [CT-ICH](https://physionet.org/content/ct-ich/1.3.1/) dataset (150 stratified head-CT slices, intracranial hemorrhage detection). Every row below uses the same manifest, the same constrained prompt, and the same scoring code — only one factor changes per row (the model, or whether a retrieval context is injected), so differences are attributable to that one factor.

| Dataset | Method | n | Any-hem F1 | 95% CI | Precision | Recall | Hemorrhage slices missed entirely |
|---|---|---|---|---|---|---|---|
| CT-ICH | No-RAG (Qwen3-VL-4B) | 150 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 100.0% (105/105) |
| CT-ICH | Text-RAG | 150 | 0.018 | [0.000, 0.056] | 0.250 | 0.010 | 99.0% (104/105) |
| CT-ICH | MedGemma-4B | 150 | 0.336 | [0.233, 0.439] | 0.846 | 0.210 | 79.0% (83/105) |

*CI = 95% bootstrap percentile interval (2000 resamples). A CI that does not cross 0 means the F1 is statistically distinguishable from a no-effect floor; a CI that hugs 0 means it isn't, regardless of the point estimate.*

Reproduce any row:
```bash
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --out results.jsonl              # No-RAG
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --context text --out results.jsonl  # Text-RAG
HF_HOME=/mnt/hf MODEL_ROLE=medical_baseline python python/eval/run_baseline.py --out results.jsonl          # MedGemma-4B
python python/eval/score_baseline.py --results results.jsonl
python python/eval/compare_runs.py   # regenerate the full comparison table from all summaries
```

### Honest limitations

- **General-purpose VLMs floor out at this task.** A general vision-language model with no domain pretraining (Qwen3-VL-4B) detects essentially zero hemorrhages — not a tuning failure, but a perceptual ceiling: it cannot see what it was never trained to recognize.
- **Domain pretraining helps, unevenly.** A medically pretrained model (MedGemma-4B) lifts recall to 0.21, but two of five hemorrhage subtypes (epidural, subdural) are still missed 100% of the time. "Domain pretraining helps" is true; "domain pretraining solves this" is not.
- **Text knowledge cannot substitute for visual training.** Injecting textbook descriptions of each hemorrhage subtype (retrieval-augmented generation) left the floor-level result essentially unchanged (F1 0.000 -> 0.018, CI still hugging 0). The gap here is perceptual, not a missing-knowledge problem — which is why a text-knowledge fix doesn't move it.
- **Single dataset, small n.** 150 slices from one source cohort; the bootstrap CIs above are the honest expression of how much that limits precision of the point estimates. Cross-dataset external validation (RSNA) is in progress.
- **Rule-based answer parsing, not a learned clinical labeler.** Free-text model output is parsed with a constrained-format-first, keyword-fallback parser (`python/eval/parse_answer.py`); parse-failure and refusal rates are reported as first-class metrics precisely so a low score can't be hand-waved away as "the parser didn't understand it" (both rates are 0% across all rows above).
