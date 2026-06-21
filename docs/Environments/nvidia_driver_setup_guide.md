# NVIDIA Driver Setup Guide

Covers GPU driver installation and DKMS / kernel-module troubleshooting.
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

## 2. Troubleshooting

| Symptom | Root cause | Fix |
|---|---|---|
| `ubuntu-drivers: command not found` | The AWS minimal AMI does not include hardware-detection tools by default | `sudo apt-get install -y ubuntu-drivers-common` |
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | The driver is installed but the `nvidia.ko` kernel module is not loaded (common after install before a reboot) | In order: (1) `sudo reboot`; (2) check with `dkms status`; (3) `sudo modprobe nvidia` to force-load |
| `dkms status` returns no output (blank) | No NVIDIA kernel source is registered; usually because matching kernel headers were missing during a prior install and the DKMS build aborted | Full cleanup and reinstall (see steps below) |
| `torch.cuda.is_available()` = False | `pip install torch` resolves to cu130 (CUDA 13), but the driver supports only CUDA 12.2 — a major-version mismatch | Reinstall the cu121 torch build (see [Model Deployment Notes](./model_deployment_notes.md)) |
| `PermissionError … /mnt/models/hf` during model download | `HF_HOME` points to a path that does not exist or is not writable (the instance store is not mounted, so `/mnt/models/hf` is absent) | Run the Instance Store mounting steps below, then set `export HF_HOME=/mnt/hf` |
| Model download aborts with `No space left on device` | The root disk (`/dev/root`, ~49 GB) has only 5–7 GB free after OS packages — not enough for Qwen3-VL-4B (~8.88 GB) | Same as above — redirect the HF model cache to the instance store |

**Instance Store mounting steps**

The g4dn.xlarge includes a ~116 GB NVMe instance store that is **unformatted and unmounted by default**. It must be remounted after each instance start (the instance store is wiped on stop/start).

> ⚠️ **Identify the device by size + state, NOT by a hard-coded name.** NVMe device names are not guaranteed stable across instances/AMIs. On this instance `lsblk` showed the instance store as **`nvme0n1` (116.4 GB, no partitions, unmounted)** and the EBS root as **`nvme1n1` (50 GB, partitioned, mounted at `/`)** — the *opposite* of the naming some AWS docs assume. **Before `mkfs`, confirm the target is the large, unpartitioned, unmounted disk — never format the disk mounted at `/`.**

```bash
# 1. Identify the instance store: the large (~116 GB), unpartitioned, UNMOUNTED disk
lsblk
#  → on this instance: nvme1n1 = 50 GB, mounted at /  (EBS root — DO NOT touch)
#                      nvme0n1 = 116.4 GB, no mountpoint (instance store — use this)
NVME=/dev/nvme0n1   # adjust to whatever lsblk shows as the unmounted ~116 GB disk

# 2. Check remaining space on the root disk and /mnt
df -h /        # root disk; typically only 5–7 GB free
df -h /mnt     # if /mnt has no separate mount, this shows the same number as root

# 3. Format (first time only; after a stop/start the data is wiped but no reformat needed)
sudo mkfs.ext4 "$NVME"

# 4. Mount and grant ownership
sudo mount "$NVME" /mnt
sudo chown ubuntu:ubuntu /mnt

# 5. Confirm available space (should show ~108 GB)
df -h /mnt

# 6. Point the HF model cache at the instance store
export HF_HOME=/mnt/hf
```

> **Note**: `export HF_HOME=…` applies to the current shell session only. Re-run it after reconnecting via SSH, or add it to `~/.bashrc` to make it persistent.

**Kernel module not loaded — three-stage procedure**
```bash
# Stage A: reboot (fastest, recommended)
sudo reboot   # after ~60 s, reconnect and run nvidia-smi

# Stage B: check DKMS build status
dkms status
#  → "nvidia/xxx, ..., installed" = built; go to Stage C
#  → completely blank = not registered; go to the reinstall steps

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
```
When `nvidia-smi` prints the T4 and CUDA version (typically 12.x), the driver is fully ready — PyTorch and any other GPU process on the host can use it directly.
