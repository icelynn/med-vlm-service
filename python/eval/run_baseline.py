# -*- coding: utf-8 -*-
"""
No-RAG baseline inference harness — Qwen3-VL over the CT-ICH eval set.
=====================================================================

Reads data/ct_ich/manifest.csv, runs ONE constrained inference per slice on
Qwen3-VL-4B, parses the answer into a 6-dim hemorrhage vector, and appends one
JSON line per slice to results.jsonl. Run this ON the T4 (GPU required).

This is the No-RAG baseline: the model sees only the image + a constrained
prompt, no retrieval. Scoring is a separate, CPU-only step
(score_baseline.py) so you can re-score under different uncertain policies
without re-running the model.

Resume / checkpoint
-------------------
results.jsonl is the state. Each slice is appended immediately, so a crash or
spot instance reclaim loses at most the in-flight slice. Re-running skips every
image_file already present in the JSONL.

Loading reuses feasibility_check.load_model / run_inference (fp16, sdpa, T4-safe
image cap).

Usage
-----
    python run_baseline.py                         # full manifest, default model
    python run_baseline.py --limit 5               # smoke test (first 5 slices)
    python run_baseline.py --out results_v2.jsonl  # fresh run
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python" / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "python" / "rag"))
sys.path.insert(0, str(EVAL_DIR))

from feasibility_check import load_model, run_inference, resolve_dtype  # noqa: E402
from parse_answer import parse_answer, SUBTYPES  # noqa: E402
from config import MODELS, config as svc_config  # noqa: E402
from providers import build_context  # noqa: E402

# The eval pipeline is the ENV=dev (HF + transformers) track. The default model comes
# from the shared registry so model identities live in one place (config.py). Resolved
# lazily (only when --model is not given) so a bad MODEL_ROLE never blocks an explicit
# --model override, and produces a clear error instead of a raw KeyError.
def _default_eval_model():
    role_models = MODELS.get(svc_config.MODEL_ROLE)
    if role_models is None:
        raise SystemExit(
            f"[error] MODEL_ROLE='{svc_config.MODEL_ROLE}' is not one of {list(MODELS)}. "
            "Set MODEL_ROLE to 'main' or 'medical_baseline' (or pass --model explicitly)."
        )
    return role_models["hf"]

DATA_DIR = PROJECT_ROOT / "data" / "ct_ich"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST = DATA_DIR / "manifest.csv"

# Constrained prompt: two fixed lines so parse_answer's structured path fires.
SYSTEM_PROMPT = (
    "You are a neuroradiologist reading a single axial head CT slice shown in a "
    "brain window. Judge only what is visible on this one slice."
)
USER_PROMPT = (
    "Assess this head CT slice for acute intracranial hemorrhage. "
    "Answer in EXACTLY this format and nothing else:\n"
    "HEMORRHAGE: <yes or no>\n"
    "SUBTYPES: <comma-separated list of those present, from IPH, IVH, SAH, EDH, "
    "SDH; or 'none'>\n\n"
    "Where IPH=intraparenchymal, IVH=intraventricular, SAH=subarachnoid, "
    "EDH=epidural, SDH=subdural."
)


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(out_path):
    """Return the set of image_file values already in the JSONL (resume)."""
    done = set()
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["image_file"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(description="CT-ICH No-RAG baseline inference")
    ap.add_argument("--model", default=None,
                    help="HF repo id; defaults to config.MODELS[MODEL_ROLE]['hf'] "
                         "(MODEL_ROLE=medical_baseline selects MedGemma)")
    ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--images", default=str(IMAGES_DIR))
    ap.add_argument("--out", default=str(DATA_DIR / "results.jsonl"))
    ap.add_argument("--context", default="none", choices=["none", "text", "image"],
                    help="retrieval-injection provider (R1=text, R2=image, planned); "
                         "default 'none' keeps the No-RAG baseline unchanged")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="constrained answer is short; 64 is plenty")
    ap.add_argument("--max-image-size", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N")
    args = ap.parse_args()
    if args.model is None:
        args.model = _default_eval_model()

    import torch
    if not torch.cuda.is_available():
        print("[FAIL] CUDA not available — run this ON the T4 instance.")
        return

    rows = load_manifest(Path(args.manifest))
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    done = already_done(out_path)
    todo = [r for r in rows if r["image_file"] not in done]
    print(f"[info] manifest={len(rows)}  done={len(done)}  todo={len(todo)}  "
          f"model={args.model}")
    if not todo:
        print("[done] nothing to do — all slices already in", out_path.name)
        return

    dtype = resolve_dtype(args.model, args.dtype)
    print(f"[info] loading {args.model} (dtype={dtype}) ...")
    processor, model, load_s = load_model(args.model, args.quant, dtype)
    print(f"[ok  ] model loaded in {load_s:.1f}s")

    context = build_context(args.context)
    system_prompt = f"{SYSTEM_PROMPT}\n\n{context}" if context else SYSTEM_PROMPT
    if context:
        print(f"[info] context provider={args.context} ({len(context)} chars injected)")

    images_dir = Path(args.images)
    n_parsed = n_refused = 0
    t_start = time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, row in enumerate(todo, 1):
            img_path = images_dir / row["image_file"]
            if not img_path.is_file():
                print(f"[warn] missing image {row['image_file']} — skipped")
                continue
            text, gen_s, n_tok = run_inference(
                processor, model, str(img_path), USER_PROMPT, system_prompt,
                args.max_new_tokens, args.max_image_size)
            parsed = parse_answer(text)
            n_parsed += int(parsed["parsed"])
            n_refused += int(parsed["refused"])

            rec = {
                "image_file": row["image_file"],
                "patient": int(row["patient"]),
                "slice": int(row["slice"]),
                "pred_any_hem": parsed["any_hem"],
                **{f"pred_{k}": parsed[k] for k in SUBTYPES},
                "refused": parsed["refused"],
                "parsed": parsed["parsed"],
                "method": parsed["method"],
                "raw_text": text,
                "gen_seconds": round(gen_s, 2),
                "n_tokens": n_tok,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t_start)
                print(f"[{i:>3}/{len(todo)}] {row['image_file']}  "
                      f"any_hem={parsed['any_hem']} method={parsed['method']}  "
                      f"({rate:.2f} slice/s)")

    print(f"\n[done] wrote {len(todo)} results -> {out_path}")
    print(f"       parsed={n_parsed}/{len(todo)}  refused={n_refused}/{len(todo)}")
    print(f"       next: python3 python/eval/score_baseline.py --results {out_path}")


if __name__ == "__main__":
    main()
