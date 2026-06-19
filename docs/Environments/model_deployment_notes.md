# Model Deployment Notes — Running Chest X-ray VLMs on an NVIDIA T4

Engineering notes from bringing up three open vision-language models — Qwen3-VL-4B,
Qwen3-VL-8B, and MedGemma-4B — for single-image chest X-ray inference on a single
NVIDIA T4 (16 GB, Turing, `sm_75`) on AWS EC2 `g4dn.xlarge`. Each issue below lists
the symptom, root cause, fix, and the general lesson.

Back to overview: [Environment & Dependency Overview](./environment_and_dependencies_overview.md)

## Summary

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `python: command not found` | Ubuntu 22.04 ships only `python3` | Activate the venv / auto-detect `python3` in the runner |
| 2 | `torch.cuda.is_available()` = False | Installed `torch ...+cu130` (CUDA 13); driver only supports CUDA 12.2 (major mismatch) | Install the cu121 torch build |
| 3 | Qwen3-VL-4B `CUDA OOM` | Input image not resized → vision-token explosion | Cap image long edge to 896 px before inference |
| 4 | Qwen3-VL-8B `No space left on device` | Root EBS 98 % full (1.2 GB free); model needs 17.5 GB | Mount the idle 116 GB instance store, relocate the HF cache |
| 5 | MedGemma `403 gated repo` | HF account not yet granted access | Accept the license on the model page |
| 6 | MedGemma `401 ... Please log in` | Instance not authenticated to HF | `huggingface-cli login` |
| 7 | MedGemma `couldn't connect to huggingface.co` | Misleading message; real cause is a fine-grained token missing gated-repo permission | Use a Read token, or enable gated-repo access on the token |
| 8 | MedGemma emits only `<pad>` | Gemma-family activations overflow in fp16 | Load in bfloat16 |
| 9 | `PermissionError … /mnt/models/hf` during model download | `HF_HOME` points into the instance store, but the store is unmounted after every stop/start | Remount the instance store (the unmounted ~116 GB disk — `nvme0n1` on this instance), then set `export HF_HOME=/mnt/hf` |

---

## 1. `python` not found

**Symptom:** `python: command not found`; all models skipped.

**Root cause:** Ubuntu 22.04 intentionally ships only `python3` and does not create a
`python` alias (to avoid confusion with the retired Python 2).

**Fix:** Activate the project venv (`source .venv/bin/activate`), which provides
`python`. The runner script also auto-detects the interpreter:
`PY="$(command -v python3 || command -v python)"`.

**Lesson:** On Ubuntu, a missing `python` is expected. Do not `apt install python`
(that pulls in Python 2).

## 2. torch cannot see the GPU (CUDA version mismatch)

**Symptom:**
```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)...
2.12.0+cu130 False NO GPU
```

**Root cause:** `pip install torch` resolved to a **cu130 (CUDA 13) build**, but the
driver supports only **CUDA 12.2** (`12020`). CUDA 12 → 13 is a **major** version
change and is not backward-compatible, so torch sees no usable GPU.

**Fix:** install a CUDA 12.x torch build:
```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
`cu121` (CUDA 12.1) runs fine on a 12.2 driver (CUDA 12 is backward-compatible within
the major version).

**Lesson:** torch's `+cuXXX` tag must match the CUDA version the GPU driver supports.
Check `nvidia-smi` for the supported CUDA version, then pick the matching wheel.

## 3. Qwen3-VL-4B out of memory (OOM)

**Symptom:** a 4B model (~8 GB of fp16 weights) OOMs on a 14.58 GB T4.

**Root cause:** Qwen-VL's **vision-token count scales with input image resolution**.
An un-resized high-resolution X-ray produces a large activation spike in the vision
encoder / prefill that exhausts VRAM.

**Fix:** resize the image long edge to 896 px before inference (`PIL.Image.thumbnail`,
preserves aspect ratio, only shrinks). Peak VRAM then drops to 8.75 GB.

**Lesson:** multimodal memory is not just weight size — **input image resolution
matters just as much**.

## 4. Qwen3-VL-8B out of disk space

**Symptom:** `No space left on device`, `File reconstruction error`; the 8B model
(17.5 GB) fails to download.

**Root cause:** `df -h` showed the root volume (50 GB EBS) 98 % full, 1.2 GB free.

**Key observation:** `lsblk` revealed a **116 GB instance-store NVMe that was
unmounted** — local high-speed storage included with `g4dn`. (On this instance it
appeared as `nvme0n1`, with the EBS root as `nvme1n1` — the opposite of what some AWS
docs assume. Device names vary: identify the store by size + unmounted state, and
**never `mkfs` the disk mounted at `/`**.)

**Fix:** format, mount, and relocate the Hugging Face cache (`NVME` = the unmounted ~116 GB disk per `lsblk`):
```bash
NVME=/dev/nvme0n1   # confirm with lsblk: the large, unpartitioned, unmounted disk
sudo mkfs.ext4 "$NVME"
sudo mkdir -p /mnt/models && sudo mount "$NVME" /mnt/models
sudo chown $USER:$USER /mnt/models
mkdir -p /mnt/models/hf && mv ~/.cache/huggingface/hub /mnt/models/hf/
export HF_HOME=/mnt/models/hf
```

**Lesson:** instance-store volumes are large, free, and fast but **ephemeral (wiped on
stop)**. Good for re-downloadable model caches; for persistence, expand the root EBS
volume instead.

## 5–7. Three gates to accessing a gated model (MedGemma)

MedGemma is a **gated** model; the error message evolved at each gate:

**5) `403 gated repo`** — the account had not been granted access.
Fix: accept the terms on the model page.

**6) `401 ... Please log in`** — access granted, but the instance was not authenticated.
Fix: `huggingface-cli login`.

**7) `couldn't connect to huggingface.co` (misleading)** — the trickiest.
- It looked like a network problem, but `curl https://huggingface.co` returned 200 and
  `HfApi().model_info()` succeeded — proving network and hub access were fine.
- The HF Xet download backend (different CDN hosts) was suspected; disabling it
  (`HF_HUB_DISABLE_XET=1`) still failed — a **red herring**.
- An explicit CLI download exposed the real cause:
  ```
  403 Forbidden: Please enable access to public gated repositories
  in your fine-grained token settings...
  ```
- **Root cause:** a **fine-grained token** without the "access public gated repos"
  permission. `transformers` surfaced this 403 as a misleading "couldn't connect" error.

Fix: use a **classic Read token** (which can read gated repos you've been granted), or
enable "Read access to contents of all public gated repos" on the fine-grained token.

**Lesson:** (1) a `transformers` "couldn't connect" error is not always a network
problem — `huggingface-cli download` surfaces the underlying error. (2) Gated access
requires three things at once: license acceptance, instance login, and token permission.

## 8. MedGemma emits only `<pad>` (fp16 numerical overflow)

**Symptom:** in fp16, MedGemma loads and "generates" 256 tokens, but the decoded text
is empty; `skip_special_tokens=False` shows all `<pad>`.

**Root cause:** the **Gemma family (incl. MedGemma) is trained in bfloat16** and has a
wide activation range. In fp16 those activations **overflow to NaN**, and the model
degenerates into emitting padding. (The HF model card lists Tensor type = BF16.)

**Fix:** load in **bfloat16**. The T4 has no native bf16 acceleration (so it is slower:
6.37 vs 14+ tok/s) but is numerically correct, and output returns to coherent radiology
text. The harness now auto-selects bf16 for any model id containing "gemma".

**Lesson:** **dtype is not a free choice.** fp16 has a small range and overflows easily;
some models (Gemma family) require bf16. On older GPUs without native bf16 (T4), bf16
still runs — "correct but slower" beats "fast but wrong". Qwen3-VL is unaffected and
runs fine in fp16.

## 9. HF_HOME path not writable (instance store not mounted)

**Symptom:** model download fails immediately with:
```
Could not cache non-existence of file. Error: [Errno 13] Permission denied: '/mnt/models/hf'
PermissionError: [Errno 13] Permission denied: '/mnt/models/hf'
OSError: PermissionError at /mnt/models/hf when downloading ...
```

**Root cause:** `HF_HOME` points to a path inside the instance store (e.g. `/mnt/models/hf` or `/mnt/hf`), but the instance store (the ~116 GB ephemeral disk — `nvme0n1` on this instance) is wiped and unmounted on every stop/start — the path no longer exists.

**How to diagnose:**
```bash
lsblk          # the unmounted ~116 GB disk (here nvme0n1) is the instance store
df -h /mnt     # if this shows the same numbers as root (/), /mnt has no separate mount
```

**Fix:** remount after each instance start (no reformat needed, but previously cached models are gone and must be re-downloaded):
```bash
NVME=/dev/nvme0n1   # confirm with lsblk: the unmounted ~116 GB disk (NOT the one mounted at /)
sudo mount "$NVME" /mnt
sudo chown ubuntu:ubuntu /mnt
export HF_HOME=/mnt/hf
```

To persist across SSH sessions:
```bash
echo 'export HF_HOME=/mnt/hf' >> ~/.bashrc
```

**Lesson:** the instance store is **wiped and unmounted on every stop/start** — unlike EBS which survives restarts. `export` only applies to the current shell; add it to `~/.bashrc` for persistence across sessions.

---

## 10. External corroboration (literature cross-check)

> The following observations come from a MIDL 2026 paper (LLaMA32-Med, Dong et al.,
> PEFT fine-tuning of LLaMA 3.2 Vision for medical VQA). They independently corroborate
> problems we hit during bring-up and are recorded here purely as engineering
> cross-references, not as part of our own results.

- **An 11B Vision model is genuinely too heavy for a 16 GB-class GPU.** Even with QLoRA
  (4-bit NF4), fine-tuning LLaMA 3.2 Vision 11B required ~20.4 GB on an A6000 — already
  above our T4's 14.58 GB, and that is the *training* footprint. Independent support for
  dropping the 11B and moving to 4B-class models.
- **Image resolution is a primary OOM lever (cf. issue 3).** That work resizes medical
  images to 512×512 before training specifically to "reduce OOM risk." Same lever as our
  inference-side fix (resize to 896 px to contain vision-token blow-up) — confirmed across
  both training and inference.
- **4-bit quantization costs accuracy, not just speed.** Their ablation shows LoRA beats
  QLoRA (4-bit NF4) by ~3% accuracy on SLAKE. This adds a second dimension to our
  "8B-4bit fits but is ~3× slower" observation: 4-bit can also trade away quality. On a
  T4, **4B fp16 generally beats 8B 4-bit** on both speed and numerical fidelity.
- **(Context) general VLMs are weak zero-shot on medical tasks.** Their zero-shot numbers
  put general VLMs (LLaMA 3.2 / Qwen2.5-VL / Gemma3) at ~30–44% on medical VQA — a useful
  expectation-setter: a general model is not clinically reliable out of the box, which is
  why factual quality should be measured with a structured labeler rather than fluency.

Source: LLaMA32-Med, MIDL 2026 — https://openreview.net/forum?id=qGgZZwEeef

---

## Takeaways

Getting these models to run was less about the models themselves and more about the
surrounding deployment surface: GPU VRAM limits, CUDA/driver compatibility, disk and
storage planning on cloud instances, gated-model access governance, and the effect of
numerical precision (dtype) on output correctness. Each is a common, reproducible
failure mode when serving vision-language models on constrained hardware.
