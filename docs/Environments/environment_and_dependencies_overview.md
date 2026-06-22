# Environment & Dependency Overview

This document is the entry point for the **med-vlm-service** environment setup — a multimodal medical-imaging microservice running on cloud GPU. It consolidates the full set of requirements for building the GPU development environment on AWS EC2. Per-service setup, troubleshooting, and verification steps are split into the documents linked below.

## Document Index

| Document | Scope |
|---|---|
| [AWS Setup Guide](./aws_setup_guide.md) | EC2 instance provisioning, SSH access, security-group ingress, CloudWatch idle auto-stop |
| [Docker Setup Guide](./docker_setup_guide.md) | Docker Engine installation, image build, FastAPI service deployment |
| [NVIDIA Driver Setup Guide](./nvidia_driver_setup_guide.md) | GPU driver installation, DKMS kernel-module troubleshooting |
| [Model Deployment Notes](./model_deployment_notes.md) | Model loading, CUDA compatibility, VRAM, gated-repo access, and dtype issues at the inference layer |
| [Ollama Setup Journey](./ollama_setup_journey.md) | Timeline and decisions from the demo backend's attempt at Ollama: clean install → T4 CUDA bug → two mitigation paths → retired in favor of HF transformers |

## 1. Architecture

The stack spans four layers, from underlying hardware up to the API surface:

```
[Hardware]   AWS g4dn.xlarge (NVIDIA T4 / 16 GB VRAM)
   │
[Driver]     NVIDIA Driver 550-server + DKMS
   │
[Application] FastAPI (8000) + transformers
```
> Note: an earlier design also ran an Ollama server on port 11434 for the demo backend. **Ollama was retired (2026-06-21)** — demo and eval now share a single transformers engine. Ollama mentions below are struck through and kept only as historical context.

## 2. Cloud Hardware Specification

| Item | Specification | Notes |
|---|---|---|
| AMI | Ubuntu Server 22.04 LTS (HVM), SSD | Standard long-term-support cloud image |
| Instance type | `g4dn.xlarge` | 1× NVIDIA T4 GPU (16 GB VRAM); best price/performance for inference |
| Root storage | ≥ 50 GB (gp3 SSD) | Multimodal models and Docker images are large |
| Instance store | ~116 GB NVMe (included with g4dn) | High-speed local disk for re-downloadable model caches (wiped on stop) |

## 3. Software & Package Stack

| Layer | Packages / Components | Purpose |
|---|---|---|
| OS base tools | `build-essential` `curl` `git` `python3-pip` `python3-venv` `ca-certificates` `gnupg` `lsb-release` | Compilation, downloads, version control, Python virtual environments |
| GPU driver | `nvidia-driver-550-server`, `nvidia-dkms-550-server`, `ubuntu-drivers-common` | Detect and drive the T4 GPU |
| Container engine | `docker-ce` `docker-ce-cli` `containerd.io` `docker-buildx-plugin` `docker-compose-plugin` | Containerized deployment of the FastAPI proxy (CPU-only; no GPU access from containers) |
| Inference engine | Direct model loading via transformers ~~/ Ollama~~ | Multimodal model inference, run natively on the host (~~Ollama~~ retired 2026-06-21; transformers only) |
| Application framework | FastAPI (port 8000), `httpx` | Serves the `/analyze` image-analysis API |
| Python ML | `torch` / `torchvision` (**must match the CUDA version — use cu121**) | Deep-learning tensor operations |

## 4. Inbound Ports (Security Group)

| Port | Service | Recommended exposure |
|---|---|---|
| 22 | SSH | Restrict to your own public IP |
| 8000 | FastAPI microservice | Public (`0.0.0.0/0`) or scoped as needed |
| ~~11434~~ | ~~Ollama API~~ | ~~Initially host-only / intra-security-group only~~ — **retired 2026-06-21**, no longer used (demo runs in-process via transformers) |

## 5. Environment Variables (`.env`)

`.env` is listed in `.gitignore` and must be recreated manually on the EC2 host.

```bash
# Mode A: ENV=local — call the cloud OpenRouter API (recommended for first-stage testing; no local GPU dependency)
ENV=local
OPENROUTER_API_KEY=your_full_OpenRouter_API_key
OPENROUTER_MODEL=google/gemini-2.5-flash

# Mode B (RETIRED 2026-06-21): ENV=dev/demo formerly called a local Ollama engine on EC2.
# Ollama was dropped (T4 CUDA kernel bug); dev (eval) and demo (served) now both load
# models directly via HF transformers. The block below is kept only as historical context.
# ENV=dev
# OLLAMA_API_URL=http://172.17.0.1:11434
# OLLAMA_MODEL=llama3.2-vision
#
# Current dev/demo: no OLLAMA_* vars; HF model ids default in python/src/config.py
# (HF_MAIN_MODEL=Qwen/Qwen3-VL-4B-Instruct, HF_MEDICAL_MODEL=google/medgemma-4b-it).
```

## 6. Outcome Summary

- **Infrastructure:** a reproducible AWS g4dn.xlarge (T4) environment covering the driver and Docker layers.
- **Deployment robustness:** systematic diagnosis and resolution of real-world issues — VRAM limits, CUDA/driver compatibility, disk and storage planning, gated-model access governance, and the effect of numerical precision (dtype) on correctness.
- **Cost control:** a combined "CloudWatch sentinel + cron heartbeat" mechanism that auto-stops the instance after 30 minutes of idleness, balancing development flexibility against GPU spend.
- **End-to-end validation:** FastAPI `/analyze` accepts multimodal requests from external clients and returns standard JSON; the service is live.
