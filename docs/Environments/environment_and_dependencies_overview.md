# Environment & Dependency Overview

This document is the entry point for the **med-vlm-service** environment setup — a multimodal medical-imaging microservice running on cloud GPU. It consolidates the full set of requirements for building the GPU development environment on AWS EC2. Per-service setup, troubleshooting, and verification steps are split into the documents linked below.

## Document Index

| Document | Scope |
|---|---|
| [AWS Setup Guide](./aws_setup_guide.md) | EC2 instance provisioning, SSH access, security-group ingress, CloudWatch idle auto-stop |
| [Docker Setup Guide](./docker_setup_guide.md) | Docker Engine installation, image build, FastAPI / Ollama service deployment |
| [NVIDIA Container Toolkit & Driver](./nvidia_container_toolkit_and_driver.md) | GPU driver installation, Container Toolkit bridging, DKMS kernel-module troubleshooting |
| [Model Deployment Notes](./model_deployment_notes.md) | Model loading, CUDA compatibility, VRAM, gated-repo access, and dtype issues at the inference layer |

## 1. Architecture

The stack spans four layers, from underlying hardware up to the API surface:

```
[Hardware]   AWS g4dn.xlarge (NVIDIA T4 / 16 GB VRAM)
   │
[Driver]     NVIDIA Driver 550-server + DKMS
   │
[Container]  Docker Engine + NVIDIA Container Toolkit
   │
[Application] FastAPI (8000) + Ollama (11434) / transformers
```

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
| Container engine | `docker-ce` `docker-ce-cli` `containerd.io` `docker-buildx-plugin` `docker-compose-plugin` | Containerized deployment |
| GPU container bridge | `nvidia-container-toolkit` | Expose host GPU to containers |
| Inference engine | Ollama (containerized) / direct model loading via transformers | Multimodal model inference |
| Application framework | FastAPI (port 8000), `httpx` | Serves the `/analyze` image-analysis API |
| Python ML | `torch` / `torchvision` (**must match the CUDA version — use cu121**) | Deep-learning tensor operations |

## 4. Inbound Ports (Security Group)

| Port | Service | Recommended exposure |
|---|---|---|
| 22 | SSH | Restrict to your own public IP |
| 8000 | FastAPI microservice | Public (`0.0.0.0/0`) or scoped as needed |
| 11434 | Ollama API | Initially host-only / intra-security-group only |

## 5. Environment Variables (`.env`)

`.env` is listed in `.gitignore` and must be recreated manually on the EC2 host.

```bash
# Mode A: ENV=local — call the cloud OpenRouter API (recommended for first-stage testing; no local GPU dependency)
ENV=local
OPENROUTER_API_KEY=your_full_OpenRouter_API_key
OPENROUTER_MODEL=google/gemini-2.5-flash

# Mode B: ENV=dev — call the Ollama engine deployed locally on EC2
# Note: to bypass Docker network isolation, the IP must point to the host gateway 172.17.0.1
ENV=dev
OLLAMA_API_URL=http://172.17.0.1:11434
OLLAMA_MODEL=llama3.2-vision
```

## 6. Outcome Summary

- **Infrastructure:** a reproducible AWS g4dn.xlarge (T4) environment covering the driver, Docker, and GPU container-bridge layers.
- **Deployment robustness:** systematic diagnosis and resolution of real-world issues — VRAM limits, CUDA/driver compatibility, disk and storage planning, gated-model access governance, and the effect of numerical precision (dtype) on correctness.
- **Cost control:** a combined "CloudWatch sentinel + cron heartbeat" mechanism that auto-stops the instance after 30 minutes of idleness, balancing development flexibility against GPU spend.
- **End-to-end validation:** FastAPI `/analyze` accepts multimodal requests from external clients and returns standard JSON; the service is live.
