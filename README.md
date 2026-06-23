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

The `dev` track (`python/eval/`) runs a controlled, reproducible evaluation protocol across two independent head-CT hemorrhage datasets: [CT-ICH](https://physionet.org/content/ct-ich/1.3.1/) (75-patient cohort, PhysioNet) and [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) (multi-institutional Kaggle challenge). 150 slices per dataset, drawn via multi-label stratified sampling so each sample's subtype prevalence and co-occurrence rate track the true population (not an artificially balanced subset). Every row uses the same manifest within its dataset, the same constrained prompt, and the same scoring code — only one factor changes per row (the model, or whether a retrieval context is injected).

| Dataset | Method | n | Any-hem F1 | 95% CI | Precision | Recall | Hemorrhage slices missed entirely |
|---|---|---|---|---|---|---|---|
| CT-ICH | No-RAG (Qwen3-VL-4B) | 150 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 100.0% (105/105) |
| CT-ICH | Text-RAG | 150 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 100.0% (105/105) |
| CT-ICH | MedGemma-4B | 150 | 0.242 | [0.143, 0.345] | 0.789 | 0.143 | 85.7% (90/105) |
| RSNA | No-RAG (Qwen3-VL-4B) | 150 | 0.158 | [0.073, 0.252] | 1.000 | 0.086 | 91.4% (96/105) |
| RSNA | Text-RAG | 150 | 0.202 | [0.107, 0.298] | 0.857 | 0.114 | 88.6% (93/105) |
| RSNA | MedGemma-4B | 150 | 0.568 | [0.464, 0.658] | 0.977 | 0.400 | 60.0% (63/105) |
| RSNA | Image-Retrieval (R2) | 150 | 0.556 | [0.456, 0.650] | 0.913 | 0.400 | 60.0% (63/105) |

*CI = 95% bootstrap percentile interval (2000 resamples). A CI that does not cross 0 means the F1 is statistically distinguishable from a no-effect floor; a CI that hugs 0 means it isn't, regardless of the point estimate.*

Image-Retrieval (R2) gives the same general model (Qwen3-VL-4B, no domain pretraining) a handful of visually-similar reference slices with known labels instead of text — a few-shot analogue built from a study-disjoint pool of RSNA images (verified zero overlap with the eval set by construction, see `python/rag/build_rsna_pool.py`). Paired bootstrap against the other RSNA rows on the same 150 slices: **+0.398 F1 vs No-RAG** (p<0.0001) and **+0.355 F1 vs Text-RAG** (p<0.0001) — both decisive — versus **-0.011 vs MedGemma-4B** (p=0.85, not distinguishable from zero). Showing the model relevant images closes nearly all of the gap that describing them in words could not.

Reproduce any row:
```bash
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --out results.jsonl              # No-RAG
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --context text --out results.jsonl  # Text-RAG
HF_HOME=/mnt/hf MODEL_ROLE=medical_baseline python python/eval/run_baseline.py --out results.jsonl          # MedGemma-4B
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --context image --out results.jsonl # Image-Retrieval (RSNA only)
# add --manifest data/rsna/manifest.csv --images data/rsna/images for the RSNA rows
python python/eval/score_baseline.py --results results.jsonl
python python/eval/compare_runs.py   # regenerate the full comparison table from all summaries
```

### Honest limitations

- **General-purpose VLMs floor out at this task, on both datasets.** A general vision-language model with no domain pretraining (Qwen3-VL-4B) detects essentially zero hemorrhages on CT-ICH (F1=0.000) and barely more on RSNA (F1=0.158, recall=0.086) — not a tuning failure, but a perceptual ceiling: it cannot reliably see what it was never trained to recognize.
- **Domain pretraining helps, unevenly — and the size of the help is dataset-dependent.** MedGemma-4B scores F1=0.242 on CT-ICH but F1=0.568 on RSNA, with non-overlapping confidence intervals — the *same model, same prompt* performs very differently depending on which dataset it's looking at. We do not have a confirmed explanation for this gap; three candidate factors are plausible and not mutually exclusive: (1) RSNA's higher co-occurrence rate gives the binary "any hemorrhage" metric more chances to be right via any one of several simultaneous findings; (2) the two datasets' source images differ in lesion severity and/or windowing/post-processing pipeline — a controlled test (same 150 RSNA images, only swapping a 128px source for a 512px one) found the *higher-resolution* source scored *worse* (F1=0.331 vs 0.568), with a directly visible loss of hyperdensity contrast on the same case across the two sources, pointing at windowing rather than resolution as the operative variable; (3) RSNA is one of the most widely discussed public medical-imaging benchmarks since 2019 and we cannot rule out the model's pretraining corpus having had some exposure to it, versus the far more obscure, access-gated CT-ICH. We report all three candidates rather than picking one — the uncertainty itself is the honest finding.
- **Text knowledge cannot substitute for visual training.** Injecting textbook descriptions of each hemorrhage subtype (retrieval-augmented generation) left the floor-level result unchanged on CT-ICH (F1=0.000 either way) and within the same confidence interval as no-RAG on RSNA (0.202 vs 0.158, CIs overlap). The gap is perceptual, not a missing-knowledge problem — which is why a text-knowledge fix doesn't move it.
- **...but visual examples can.** Swapping the retrieval content from text descriptions to a handful of visually-similar reference images with known labels (Image-Retrieval, RSNA) recovers most of that gap: F1=0.556, decisively above both No-RAG and Text-RAG on the same 150 slices (paired bootstrap p<0.0001 for both) and statistically indistinguishable from the domain-pretrained MedGemma-4B (p=0.85). A general model with no medical pretraining can approach a specialized one's performance if shown what the answer looks like, even though being told what it looks like in words does nothing — the bottleneck is genuinely perceptual, not informational, and visual few-shot examples are a more direct fix for a visual problem than retrieval-augmented text ever could be.
- **Small n on both datasets.** 150 slices per dataset; the bootstrap CIs above are the honest expression of how much that limits precision of the point estimates, especially for rarer subtypes.
- **Rule-based answer parsing, not a learned clinical labeler.** Free-text model output is parsed with a constrained-format-first, keyword-fallback parser (`python/eval/parse_answer.py`); parse-failure and refusal rates are reported as first-class metrics precisely so a low score can't be hand-waved away as "the parser didn't understand it" (both rates are 0% across all rows above).
