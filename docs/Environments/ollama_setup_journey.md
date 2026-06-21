# Ollama Setup Journey (Demo Backend: Attempt and Retirement)

> This is the **full journey of attempting Ollama as the demo backend** — from a clean
> install, through hitting an upstream CUDA bug, to two mitigation attempts, to finally
> retiring Ollama in favor of HF transformers.
> It is a **timeline / decision log**, deliberately not duplicating three other documents
> (follow the links for deeper content):
> - **Operational steps and the full CUDA-bug log/root-cause**: see the "Ollama" section of
>   the [AWS Setup Guide](./aws_setup_guide.md) (install commands, systemd override,
>   instance-store rebuild steps, verification checklist).
> - **Architecture decisions** (two-track separation, why HF won in the end): see
>   `核心難題_遭遇與處理紀錄.md` (Chinese only), challenges ⑧ and ⑨.
> - **HF/transformers inference troubleshooting itself** (dtype, OOM, gated repos, etc.):
>   see [Model Deployment Notes](./model_deployment_notes.md).
>
> Back to overview: [Environment & Dependency Overview](./environment_and_dependencies_overview.md). Updated: 2026-06-21.

---

## 1. Background: why demo needed Ollama at all

Under the two-track separation (challenge ⑧), the **eval pipeline** runs HF transformers
(numerically controlled, the sole source of every comparison-table number), while the
**demo service** was originally planned to run Ollama — serving convenience, built-in
streaming, just `ollama pull` and you're live. This document records what actually
happened once "demo = Ollama" was carried all the way through.

---

## 2. Timeline

### 1. Clean install (official script, localhost-only)

- Installed via the official script: `curl -fsSL https://ollama.com/install.sh | sh`,
  default-binds `127.0.0.1:11434`, **`OLLAMA_HOST` left unset**.
- **Security model = network boundary, not app-level auth**: Ollama's own API has no
  authentication mechanism — security comes entirely from "not reachable from outside
  the instance" — localhost bind + the security group never opening 11434. The proxy and
  Ollama share the same instance, talking over `127.0.0.1`.
- Verified against the post-install checklist (service active, localhost-only bind, reachable
  locally, **confirmed unreachable from an external machine**, no 11434 security-group rule).
- Full commands and pass criteria: see the AWS Setup Guide.

### 2. Models relocated to the instance store (root disk too small)

- The second `ollama pull` hit `no space left on device`: the root EBS volume (~49 GB)
  was already close to full from the pre-existing `dev`-track venv / dataset / containerd
  footprint.
- Fix: redirect the model storage to the g4dn's own ~116 GB instance store via a systemd
  drop-in setting `OLLAMA_MODELS=/mnt/...` — models land on the instance store, root stays
  untouched.
- **Cost (ephemeral)**: the instance store is wiped on every stop/start — the mount, its
  filesystem, and every pulled model disappear; the next boot needs a fresh
  format-and-mount plus a re-pull. Full rebuild steps: see the AWS Setup Guide.

### 3. Hit a CUDA kernel crash (Turing / sm_75)

- Every model load crashed immediately: `CUDA error: device kernel image is invalid`
  (`ggml_cuda_compute_forward: PAD failed`).
- **Root cause (upstream packaging bug)**: Ollama's `0.30.x` line ships a `cuda_v12`
  library missing the correct kernels for compute capability 7.5 (Turing, including the
  T4) — `cuobjdump` showed only `sm_50` kernels bundled. Drivers reporting CUDA 12.x get
  routed to this broken library. Unrelated to our model choice or configuration.
- Full log, `cuobjdump` evidence, and issue links: see the "Known issue" section of the
  AWS Setup Guide.

### 4. Two mitigation paths — neither good enough

| Path | What we did | Result | Verdict |
|---|---|---|---|
| **A. Force `cuda_v11`** | Set `OLLAMA_LLM_LIBRARY=cuda_v11` on 0.30.10 (the older library still has correct sm_75 kernels) | **No crash, correct answers** (full `/analyze` verified end-to-end, returned a real structured report); **but extremely slow** — one CT slice took **9m4s** (~2.8 tok/s; image encoding alone ~2 min) | Correct but unusable for demo |
| **B. Downgrade to 0.24.0** | Installed the pre-engine-rewrite release (logs confirmed native `cuda_v12`) | **No crash, ~20x faster** (~64 tok/s on a text-only test); **but `qwen3-vl:4b` repeatedly refused the same valid CT image (`050_016.png`, from the CT-ICH dataset)**, while `medgemma:4b` on the identical install/image answered correctly in 2.46s — so the GPU/vision pipeline itself was fine; the failure was specific to this version's qwen3-vl support | Fast but the primary model gets it wrong |

> Note: the refusal on 0.24.0 was originally attributed to a "thinking-mode template"
> theory. That theory was later disproven — the **same refusal reproduced on the
> non-thinking HF `-Instruct` checkpoint**, which showed the real cause is the demo
> service's `system_prompt.txt` wording (challenge ⑨), not thinking-mode or backend
> choice. Kept here to document how that reasoning evolved.

### 5. Decision: demo drops Ollama, moves to HF transformers

- Both Ollama paths were stuck on either "too slow" or "wrong answers," and **upgrading
  the driver to route to `cuda_v13`** would be a far bigger version jump than anything
  else documented in this repo (535 → 580+), and would touch the GPU environment shared
  with the `dev` track — the highest-risk option.
- Final decision (2026-06-21): **the demo backend now calls HF transformers directly** —
  same engine as eval, reusing already-validated loading logic, fully sidestepping
  Ollama's T4 compatibility problem. Full decision and "two-track purpose split is
  unchanged" reasoning: see challenge ⑧.

---

## 3. Takeaways

1. **Serving convenience ≠ it will run on an old GPU.** Ollama's `pull` + built-in
   streaming genuinely saves effort, but it bundles CUDA kernels inside its own library —
   if a given release packages the wrong kernel for your GPU architecture (here, Turing
   sm_75), there is no application-level fix; only a version or driver change. On a
   niche/older GPU architecture, that "black-box convenience" is itself a risk.
2. **Version numbers are not linear — don't assume "newer = better" or "older = safer."**
   0.30.x is newer than 0.24.0 yet broken; 0.24.0 fixes the crash but under-supports the
   newer qwen3-vl. Version choice here was a three-way tradeoff (crash / speed / model
   support) with no single best answer.
3. **The instance store's ephemeral nature is extra overhead for serving specifically.**
   Eval runs in batches and is done; serving is expected to stay resident, so every
   stop/start means re-mounting and re-pulling — eating into the "convenience" Ollama was
   chosen for.
4. **Serving eval and demo off the same engine (HF) lowers overall complexity.** After
   the full detour, demo ended up reusing eval's already-validated HF loading path
   directly — avoiding the cost of maintaining a second backend (Ollama's version matrix,
   quantization-induced numerical differences). This was itself part of the engineering
   case for returning to HF.

---

*First pass. To be extended if Ollama is re-evaluated later or a driver upgrade is actually tested.*
