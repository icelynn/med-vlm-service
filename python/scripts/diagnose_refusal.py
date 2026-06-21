# -*- coding: utf-8 -*-
"""One-off diagnostic: how often does the DEMO system prompt (system_prompt.txt, with its
strict modality/quality gating rules) make Qwen3-VL-4B-Instruct refuse a REAL, valid
CT-ICH slice? Run over the full manifest, classify each output, report the refusal rate
and whether it correlates with the ground-truth hemorrhage label.

This is diagnosis only — it does not modify system_prompt.txt or any service code.

Usage: python3 diagnose_refusal.py [--limit N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from feasibility_check import load_model, run_inference, resolve_dtype  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "ct_ich"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST = DATA_DIR / "manifest.csv"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "python" / "src" / "system_prompt.txt"
FEWSHOT_SYSTEM_PROMPT_FILE = SCRIPTS_DIR / "test_prompt_A_no_defense_rule.txt"
FEWSHOT_EXAMPLES_FILE = SCRIPTS_DIR / "fewshot_examples.json"
USER_PROMPT = "Please analyze this head CT image and provide a diagnostic report."


def run_inference_fewshot(processor, model, image_path, prompt, system_prompt, examples,
                          max_new_tokens, max_image_size=896):
    """Variant 3: only show 'should-answer' example (image, ideal report) pairs as
    multi-turn few-shot demonstrations — the system prompt itself never mentions a
    refusal rule or output format at all (see python/scripts/test_prompt_A_no_defense_rule.txt).
    Mirrors feasibility_check.run_inference but builds a multi-turn chat instead of one turn.
    """
    import torch
    from PIL import Image

    def _load(path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
        return img

    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    for ex in examples:
        messages.append({"role": "user", "content": [
            {"type": "image", "image": _load(ex["image"])},
            {"type": "text", "text": ex["prompt"]},
        ]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]})
    messages.append({"role": "user", "content": [
        {"type": "image", "image": _load(image_path)},
        {"type": "text", "text": prompt},
    ]})

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - t0

    new_tokens = out[0][input_len:]
    n_new = int(new_tokens.shape[-1])
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text, gen_seconds, n_new

MODALITY_REFUSAL = "does not comply with head CT imaging standards"
QUALITY_REFUSAL = "Image clarity is insufficient"

# A real report follows the 4-section structure (Image Quality Assessment / Objective
# Findings / Impression / Recommendations). Checking for the structure itself, rather than
# guessing every possible decline phrasing, is what correctly catches natural-language
# declines like "This image is not a head CT..." (the B-prompt's decline style).
REPORT_SECTION_MARKERS = ("Objective Findings", "Impression", "Recommendations")


def classify(text):
    if MODALITY_REFUSAL in text:
        return "refused_modality"
    if QUALITY_REFUSAL in text:
        return "refused_quality"
    if "[Error]" in text:
        return "refused_other"
    if sum(1 for m in REPORT_SECTION_MARKERS if m in text) < 2:
        return "refused_other"  # short decline, natural language, no real report structure
    return "answered"


def _load_manifest_items(limit):
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return [
        {"image_path": str(IMAGES_DIR / r["image_file"]), "image_file": r["image_file"],
         "any_hemorrhage": r["any_hemorrhage"], "expect": "answered"}
        for r in rows
    ]


def _load_defensive_items(images_dir, label):
    paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    return [
        {"image_path": str(p), "image_file": f"{label}/{p.name}",
         "any_hemorrhage": None, "expect": "refused"}
        for p in paths
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all 150 CT-ICH slices")
    ap.add_argument("--out", default=str(DATA_DIR / "refusal_diagnosis.jsonl"))
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE),
                    help="system prompt variant to test (defaults to the real system_prompt.txt)")
    ap.add_argument("--skip-manifest", action="store_true",
                    help="skip the CT-ICH manifest (defensive-only run)")
    ap.add_argument("--defensive-dir", action="append", default=[],
                    metavar="LABEL=PATH",
                    help="extra directory of images that SHOULD be refused, e.g. "
                         "non_medical=data/test_images/defensive/non_medical "
                         "(repeatable)")
    ap.add_argument("--fewshot", action="store_true",
                    help="variant 3: no defense rule in the system prompt at all; instead "
                         "show 'should-answer' example (image, ideal report) pairs as "
                         "few-shot turns (see fewshot_examples.json). --prompt-file is "
                         "ignored in this mode (uses test_prompt_A_no_defense_rule.txt).")
    args = ap.parse_args()
    system_prompt = (FEWSHOT_SYSTEM_PROMPT_FILE if args.fewshot else Path(args.prompt_file)).read_text(encoding="utf-8")
    fewshot_examples = (
        json.loads(FEWSHOT_EXAMPLES_FILE.read_text(encoding="utf-8")).values()
        if args.fewshot else None
    )

    items = [] if args.skip_manifest else _load_manifest_items(args.limit)
    for spec in args.defensive_dir:
        label, _, path = spec.partition("=")
        items.extend(_load_defensive_items(path, label))

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    dtype = resolve_dtype(model_id, "auto")
    print(f"[info] loading {model_id} ...")
    processor, model, load_s = load_model(model_id, "none", dtype)
    print(f"[ok] loaded in {load_s:.1f}s; running {len(items)} images")

    counts = {"answered": 0, "refused_modality": 0, "refused_quality": 0, "refused_other": 0}
    mismatches = []
    t0 = time.perf_counter()
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, item in enumerate(items, 1):
            if args.fewshot:
                text, gen_s, n_tok = run_inference_fewshot(
                    processor, model, item["image_path"], USER_PROMPT, system_prompt,
                    fewshot_examples, max_new_tokens=600, max_image_size=896)
            else:
                text, gen_s, n_tok = run_inference(
                    processor, model, item["image_path"], USER_PROMPT, system_prompt,
                    max_new_tokens=600, max_image_size=896)
            label = classify(text)
            counts[label] += 1
            got = "answered" if label == "answered" else "refused"
            ok = got == item["expect"]
            if not ok:
                mismatches.append(item["image_file"])
            rec = {
                "image_file": item["image_file"],
                "any_hemorrhage": item["any_hemorrhage"],
                "expect": item["expect"],
                "got": got,
                "ok": ok,
                "label": label,
                "gen_seconds": round(gen_s, 2),
                "n_tokens": n_tok,
                "text_preview": text[:300],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 10 == 0 or i == len(items):
                rate = i / (time.perf_counter() - t0)
                print(f"[{i:>3}/{len(items)}] {item['image_file']:<30} {label:<18} "
                      f"{'OK' if ok else 'MISMATCH':<8} ({rate:.2f} img/s)")

    total = sum(counts.values())
    print("\n[done] summary:")
    for k, v in counts.items():
        print(f"  {k:<18} {v:>4}/{total} ({100*v/total:.1f}%)")
    print(f"  mismatches (expected vs got): {len(mismatches)}/{total}")
    for m in mismatches:
        print(f"    - {m}")
    print(f"  wrote per-image detail -> {args.out}")


if __name__ == "__main__":
    main()
