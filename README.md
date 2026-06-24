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

`/analyze` and `/analyze/stream` (HF backend only) also accept two opt-in, mutually-exclusive form fields (both default off via `TWO_STAGE_RAG`/`IMAGE_RETRIEVAL_RAG` env vars; passing both `true` raises an error):

- `two_stage=true` — findings-conditioned two-stage text-RAG (see Evaluation Results: this is a *null* result, kept for architecture-completeness demonstration, not for a quality improvement).
- `image_retrieval=true` — R2 image-retrieval few-shot, reproducing the eval-measured R2-B config (constrained `HEMORRHAGE`/`SUBTYPES` judgment, not a free-text report). **Caveat**: this mode uses the eval's constrained prompts, not `system_prompt.txt`'s modality-gating system prompt — a non-brain image will get a constrained yes/no judgment instead of the report endpoint's validated refusal. This is the accepted tradeoff of reproducing the measured 0.575/0.667 F1 configuration exactly, not a regression.

## Evaluation Results

The `dev` track (`python/eval/`) runs a controlled, reproducible evaluation protocol across two independent head-CT hemorrhage datasets: [CT-ICH](https://physionet.org/content/ct-ich/1.3.1/) (75-patient cohort, PhysioNet) and [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) (multi-institutional Kaggle challenge). 150 slices per dataset, drawn via multi-label stratified sampling so each sample's subtype prevalence and co-occurrence rate track the true population (not an artificially balanced subset). Every row uses the same manifest within its dataset, the same constrained prompt, and the same scoring code — only one factor changes per row (the model, or whether a retrieval context is injected).

| Dataset | Method | n | Any-hem F1 | 95% CI | Precision | Recall | Hemorrhage slices missed entirely |
|---|---|---|---|---|---|---|---|
| CT-ICH | No-RAG (Qwen3-VL-4B) | 150 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 100.0% (105/105) |
| CT-ICH | Text-RAG | 150 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 100.0% (105/105) |
| CT-ICH | Text-RAG (two-stage, findings-conditioned) | 150 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 100.0% (105/105) |
| CT-ICH | MedGemma-4B | 150 | 0.242 | [0.143, 0.345] | 0.789 | 0.143 | 85.7% (90/105) |
| CT-ICH | Image-Retrieval (R2-B) | 150 | 0.575 | [0.479, 0.663] | 0.836 | 0.438 | 56.2% (59/105) |
| CT-ICH | Image-Retrieval (R2-B, random-exemplar control) | 150 | 0.427 | [0.326, 0.521] | 0.711 | 0.305 | 69.5% (73/105) |
| RSNA | No-RAG (Qwen3-VL-4B) | 150 | 0.158 | [0.073, 0.252] | 1.000 | 0.086 | 91.4% (96/105) |
| RSNA | Text-RAG | 150 | 0.202 | [0.107, 0.298] | 0.857 | 0.114 | 88.6% (93/105) |
| RSNA | Text-RAG (two-stage, findings-conditioned) | 150 | 0.202 | [0.108, 0.300] | 0.857 | 0.114 | 88.6% (93/105) |
| RSNA | MedGemma-4B | 150 | 0.568 | [0.464, 0.658] | 0.977 | 0.400 | 60.0% (63/105) |
| RSNA | Image-Retrieval (R2-C) | 150 | 0.667 | [0.575, 0.745] | 0.917 | 0.524 | 47.6% (50/105) |
| RSNA | Image-Retrieval (R2-C, random-exemplar control) | 150 | 0.497 | [0.388, 0.591] | 0.841 | 0.352 | 64.8% (68/105) |

*CI = 95% bootstrap percentile interval (2000 resamples). A CI that does not cross 0 means the F1 is statistically distinguishable from a no-effect floor; a CI that hugs 0 means it isn't, regardless of the point estimate. The R2-C row reflects a resolution-matched retrieval pool — see below for why, and for the original (resolution-mismatched) measurement.*

Image-Retrieval gives the same general model (Qwen3-VL-4B, no domain pretraining) a handful of visually-similar reference slices with known labels instead of text — a few-shot analogue, retrieved from a study-disjoint pool of RSNA images (verified zero overlap with any eval set by construction, see `python/rag/build_rsna_pool.py`). Two variants: **R2-C** (RSNA pool → RSNA eval) and **R2-B** (RSNA pool → CT-ICH eval, pool histogram-matched to CT-ICH's intensity distribution first, `python/rag/harmonize_pool.py`).

We ran two additional control experiments per variant before trusting these numbers — see *Honest limitations* for what they found and why the headline claim below is narrower than our first pass at this writeup claimed:

**The claim that survives both controls**: same model, same dataset, same eval images — only the context provider changes. R2 beats No-RAG and Text-RAG decisively on both datasets (all four paired-bootstrap comparisons p<0.0001), **and** genuine visual retrieval beats a same-pool random-exemplar control once the query/pool domain is matched (R2-B: +0.148 F1, p=0.001; R2-C resolution-matched: +0.170 F1, p<0.0001). Retrieval quality is not just "having an exemplar to copy the format from" — it measurably matters, but only once the underlying visual space is consistent between pool and query.

Reproduce any row:
```bash
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --out results.jsonl              # No-RAG
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --context text --out results.jsonl  # Text-RAG
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --context text-twostage --out results.jsonl  # Text-RAG two-stage
HF_HOME=/mnt/hf MODEL_ROLE=medical_baseline python python/eval/run_baseline.py --out results.jsonl          # MedGemma-4B
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --manifest data/rsna/manifest.csv --images data/rsna/images \
  --context image --image-index-dir data/rag/image_index_c_128matched --image-pool-dir data/rsna/pool_images_128 \
  --out results.jsonl                                                                                     # Image-Retrieval R2-C (RSNA, resolution-matched)
HF_HOME=/mnt/hf MODEL_ROLE=main      python python/eval/run_baseline.py --manifest data/ct_ich/manifest.csv --images data/ct_ich/images \
  --context image --image-index-dir data/rag/image_index_b_harmonized --image-pool-dir data/rsna/pool_images_harmonized \
  --out results.jsonl                                                                                     # Image-Retrieval R2-B (CT-ICH)
# add --image-random to either command to reproduce the random-exemplar control
python python/eval/score_baseline.py --results results.jsonl
python python/eval/compare_runs.py   # regenerate the full comparison table from all summaries

# every paired-bootstrap p-value quoted above/below is reproducible from one command, e.g.:
python python/eval/significance_report.py --manifest data/rsna/manifest.csv \
  --a data/rsna/results_rag.jsonl --label-a "Text-RAG (R1)" \
  --b data/rsna/results_rag_twostage.jsonl --label-b "Text-RAG two-stage"
python python/eval/significance_report.py --self-test   # offline sanity check, no GPU needed
```

### Honest limitations

- **General-purpose VLMs floor out at this task, on both datasets.** A general vision-language model with no domain pretraining (Qwen3-VL-4B) detects essentially zero hemorrhages on CT-ICH (F1=0.000) and barely more on RSNA (F1=0.158, recall=0.086) — not a tuning failure, but a perceptual ceiling: it cannot reliably see what it was never trained to recognize.
- **Domain pretraining helps, unevenly — and the size of the help is dataset-dependent.** MedGemma-4B scores F1=0.242 on CT-ICH but F1=0.568 on RSNA, with non-overlapping confidence intervals — the *same model, same prompt* performs very differently depending on which dataset it's looking at. We do not have a confirmed explanation for this gap; three candidate factors are plausible and not mutually exclusive: (1) RSNA's higher co-occurrence rate gives the binary "any hemorrhage" metric more chances to be right via any one of several simultaneous findings; (2) the two datasets' source images differ in lesion severity and/or windowing/post-processing pipeline — a controlled test (same 150 RSNA images, only swapping a 128px source for a 512px one) found the *higher-resolution* source scored *worse* (F1=0.331 vs 0.568), with a directly visible loss of hyperdensity contrast on the same case across the two sources, pointing at windowing rather than resolution as the operative variable; (3) RSNA is one of the most widely discussed public medical-imaging benchmarks since 2019 and we cannot rule out the model's pretraining corpus having had some exposure to it, versus the far more obscure, access-gated CT-ICH. We report all three candidates rather than picking one — the uncertainty itself is the honest finding.
- **Text knowledge cannot substitute for visual training.** Injecting textbook descriptions of each hemorrhage subtype (retrieval-augmented generation) left the floor-level result unchanged on CT-ICH (F1=0.000 either way) and within the same confidence interval as no-RAG on RSNA (0.202 vs 0.158, CIs overlap). The gap is perceptual, not a missing-knowledge problem — which is why a text-knowledge fix doesn't move it.
- **A smarter retrieval query doesn't change that conclusion either.** Text-RAG's fixed query (`DEFAULT_QUERY`, retrieving the whole 5-document KB) could be dismissed as "too naive a query" rather than evidence that text knowledge itself can't help. We tested that alternative explanation directly: a two-stage variant has the model describe its own findings first, then retrieves only the top-2 KB entries conditioned on that description, before answering. Result: statistically indistinguishable from the original fixed-query Text-RAG on both datasets (paired-bootstrap diff=0.000, p=1.0 on CT-ICH and RSNA alike — 147–148/150 predictions were literally identical to Text-RAG's). This rules out "the query was too naive" as the reason text retrieval doesn't fix the perception gap.
- **...but visual examples can, on both datasets, by a wide margin.** Swapping the retrieval content from text descriptions to a handful of visually-similar reference images with known labels recovers nearly all of the No-RAG/Text-RAG gap on RSNA (R2-C: F1=0.667) and CT-ICH (R2-B: F1=0.575) — both p<0.0001 against their own dataset's No-RAG and Text-RAG rows. The bottleneck is genuinely perceptual, not informational: visual examples fix what describing the same content in words could not.
- **R2 vs MedGemma-4B is not a clean comparison — we do not claim "a general model beats a specialist."** It compares Qwen3-VL+few-shot-exemplars against MedGemma *zero-shot*; MedGemma was never tested with the same exemplars, so whether a domain-pretrained model would benefit just as much (or more) from the same few-shot images is untested and left as an open question, not a result. Separately, R2-B's and R2-C's point estimates should not be ranked against each other either — they're F1 on two *different* datasets (different 150 images each), and their 95% CIs overlap; cross-dataset ranking isn't meaningful here, the same reason No-RAG/MedGemma themselves swing by dataset (bullet above).
- **R2-C originally had an undisclosed resolution mismatch, and fixing it changed the result — but the mechanism is genuinely not what we first assumed.** The retrieval pool is the 512px original-resolution RSNA source (`vaillant/rsna-ich-png`); the first version of this experiment queried it with the 128px RSNA eval source (`guiferviz`) used throughout this project, an unintentional mismatch we didn't catch until auditing the experiment setup itself. Downscaling the pool to 128px raised F1 from 0.556 to 0.667 (p<0.0001), and a follow-up sweep (96/128/192/512px, all re-run end-to-end on the GPU) confirmed this is a real, reproducible pattern, not a one-off: pools at ≤128px score 0.617–0.667, pools at ≥192px score exactly 0.556 (192px and the original 512px landed on the *identical* F1), with a clear break somewhere between 128px and 192px. So the effect is real, but it is **not** "exactly matching the query's 128px resolution" — 96px (which doesn't match) scores statistically indistinguishably from 128px (p=0.12) and significantly above 512px (p=0.031). It's closer to "below some downscaling threshold" than "equal to the query." We were unable to explain *why* via the retrieval mechanism itself: a BiomedCLIP top-k same-subtype hit-rate sweep across the same resolution range came out essentially flat (0.54–0.59 throughout, no resolution dependence) — yet the *specific* images retrieved differ substantially by pool resolution (only ~54% top-3 overlap between the 512px- and 128px-embedded versions of the same pool). A random-exemplar control independently showed pool image resolution alone doesn't affect the VLM's classification when exemplar *identity* is held fixed (F1 was identical, 0.497, whether the same randomly-chosen images were served at 512px or 128px). Put together: resolution changes *which* images BiomedCLIP retrieves, and that changes the downstream result, but neither "retrieval gets more discriminating" nor "the VLM prefers a given file resolution" explains it on its own — the mechanism remains open.
- **Retrieval quality only shows a measurable edge over random exemplars once the domain is matched.** We compared real (similarity-retrieved) exemplars against a same-pool *random*-exemplar control (k=3 random images, not nearest-neighbor) to separate "the model benefits from seeing any few-shot example" (format demonstration) from "the model benefits from a *visually relevant* one." Under R2-C's original resolution mismatch, retrieved exemplars were *not* significantly better than random (+0.060 F1, p=0.163). Once resolution was matched, retrieval pulled significantly ahead of random (+0.170 F1, p<0.0001), and R2-B (already domain-matched via histogram matching) showed the same pattern (+0.148 F1, p=0.001). Even the random-exemplar control beats No-RAG/Text-RAG decisively on both datasets (p<0.0001) — so *some* exemplar, any exemplar, helps a lot; genuine retrieval adds a further, real increment on top, conditional on the visual domain being consistent (see the bullet above for why "domain matching" is more subtle than it sounds).
- **More domain-matching is not always better.** We also tried stacking intensity (histogram) harmonization on top of R2-C's resolution match, targeting RSNA eval's own aggregate intensity distribution (`python/rag/harmonize_pool.py --reference-images data/rsna/images`). That dropped F1 from 0.667 to 0.599 (diff -0.068, p=0.05 — right at the conventional significance boundary) instead of improving it further, though retrieval still beat its own random-exemplar control in this setting (+0.116 F1, p=0.002). Histogram matching trades off local contrast to chase a global distribution match, and a literature search earlier in this work already flagged it as the weakest of the harmonization techniques we reviewed (see `python/rag/retrieval_sentinel.py`'s docstring) — this result is consistent with that limitation rather than a fluke. We report the resolution-only-matched number (0.667) as R2-C's primary result, not the further-harmonized one.
- **Small n on both datasets.** 150 slices per dataset; the bootstrap CIs above are the honest expression of how much that limits precision of the point estimates, especially for rarer subtypes.
- **Rule-based answer parsing, not a learned clinical labeler.** Free-text model output is parsed with a constrained-format-first, keyword-fallback parser (`python/eval/parse_answer.py`); parse-failure and refusal rates are reported as first-class metrics precisely so a low score can't be hand-waved away as "the parser didn't understand it" (both rates are 0% across all rows above).
