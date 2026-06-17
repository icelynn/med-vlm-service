# NVIDIA Container Toolkit & Driver

Covers GPU driver installation, Container Toolkit bridging, and DKMS / kernel-module troubleshooting.
Back to overview: [Environment & Dependency Overview](./environment_and_dependencies_overview.md)

## 1. Setup

**Install the NVIDIA driver (server branch)**
```bash
# The cloud minimal AMI lacks this tool by default; install it first
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common

# Scan the PCIe bus for the attached GPU model (T4 / A10G / H100, etc.)
sudo ubuntu-drivers devices

# Install the official server driver + DKMS build package (550-server recommended)
sudo apt-get install -y nvidia-driver-550-server nvidia-dkms-550-server
sudo reboot   # reboot to load the kernel module
```

> **Driver branch selection (headless servers must use the `-server` suffix)**
>
> | Branch | Native CUDA | Recommended use |
> |---|---|---|
> | 535-server | 12.2+ | The most widely deployed, longest-maintained LTS branch in the cloud; choose for maximum conservatism |
> | **550-server** | 12.4+ | **Recommended branch**; optimized for PyTorch / vLLM / NCCL |
> | 565/570-server | 12.6+ | Cutting-edge branch; mainly for Hopper (H100) / Blackwell |

**Install the NVIDIA Container Toolkit (the bridge for containers to call the GPU)**
```bash
# 1. Configure the official key and repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure the Docker runtime and restart
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2. Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `ubuntu-drivers: command not found` | The AWS minimal AMI does not include hardware-detection tools by default | `sudo apt-get install -y ubuntu-drivers-common` |
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | The driver is installed but the `nvidia.ko` kernel module is not loaded (common after install before a reboot) | In order: (1) `sudo reboot`; (2) check with `dkms status`; (3) `sudo modprobe nvidia` to force-load |
| `dkms status` returns no output (blank) | No NVIDIA kernel source is registered; usually because matching kernel headers were missing during a prior install and the DKMS build aborted | Full cleanup and reinstall (see SOP below) |
| `torch.cuda.is_available()` = False | `pip install torch` resolves to cu130 (CUDA 13), but the driver supports only CUDA 12.2 — a major-version mismatch | Reinstall the cu121 torch build (see [Model Deployment Notes](./model_deployment_notes.md)) |

**Kernel module not loaded — three-stage SOP**
```bash
# Stage A: reboot (fastest, recommended)
sudo reboot   # after ~60 s, reconnect and run nvidia-smi

# Stage B: check DKMS build status
dkms status
#  → "nvidia/xxx, ..., installed" = built; go to Stage C
#  → completely blank = not registered; go to the reinstall SOP

# Stage C: force-load the module manually
sudo modprobe nvidia
nvidia-smi
```

**Blank DKMS — full cleanup and standard reinstall**
```bash
# 1. Purge all leftover / broken NVIDIA packages
sudo apt-get purge -y nvidia*
sudo apt-get autoremove -y

# 2. Install headers matching the current kernel exactly
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)

# 3. Reinstall the official server driver and DKMS package (background build ~2–3 min)
sudo apt-get install -y nvidia-driver-550-server nvidia-dkms-550-server
# Then run dkms status again and confirm it ends with "installed"
```

## 3. Verification

```bash
# 1. Host driver and low-level communication (printing the GPU table means success)
nvidia-smi

# 2. DKMS module registered correctly
dkms status        # should show nvidia/xxx, ..., installed

# 3. Container GPU access (Toolkit verification)
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```
When `nvidia-smi` prints the T4 and CUDA version (typically 12.x) and the container also sees the GPU, the infrastructure is fully ready.
