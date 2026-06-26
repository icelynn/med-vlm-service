# -*- coding: utf-8 -*-
"""
Comparison table — {method} x {dataset}, built from score_baseline.py summaries.
==================================================================================

Reads the summary_*.json files score_baseline.py already produces (no
re-scoring, no re-inference) and renders a side-by-side markdown table:
Any-hem F1 + 95% bootstrap CI + dangerous-FN rate, one row per (dataset,
method) run.

Add a new row by adding one entry to RUNS below once its summary JSON exists.
No other code changes.

Usage
-----
    python compare_runs.py
    python compare_runs.py --out comparison.md
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# (dataset, method, summary_path) -- summary_path is relative to DATA_DIR.
RUNS = [
    ("CT-ICH", "No-RAG (Qwen3-VL-4B)",                  "ct_ich/baseline_summary.json"),
    ("CT-ICH", "Text-RAG (R1)",                         "ct_ich/summary_rag.json"),
    ("CT-ICH", "Text-RAG two-stage (findings-conditioned)", "ct_ich/summary_rag_twostage.json"),
    ("CT-ICH", "MedGemma-4B (B)",                       "ct_ich/summary_medgemma.json"),
    ("CT-ICH", "Image-Retrieval (R2-B)",                "ct_ich/summary_imageretrieval.json"),
    ("CT-ICH", "Image-Retrieval (R2-B, random control)", "ct_ich/summary_imageretrieval_random.json"),
    ("RSNA",   "No-RAG (Qwen3-VL-4B)",                  "rsna/summary.json"),
    ("RSNA",   "Text-RAG (R1)",                         "rsna/summary_rag.json"),
    ("RSNA",   "Text-RAG two-stage (findings-conditioned)", "rsna/summary_rag_twostage.json"),
    ("RSNA",   "MedGemma-4B (B)",                       "rsna/summary_medgemma.json"),
    ("RSNA",   "Image-Retrieval (R2-C, res-matched)",   "rsna/summary_imageretrieval_128matched.json"),
    ("RSNA",   "Image-Retrieval (R2-C, res-mismatched)", "rsna/summary_imageretrieval.json"),
    ("RSNA",   "Image-Retrieval (R2-C, random control)", "rsna/summary_imageretrieval_128matched_random.json"),
    ("RSNA",   "Image-Retrieval (R2-C, res+intensity-matched)", "rsna/summary_imageretrieval_128harmonized.json"),
    ("RSNA",   "Image-Retrieval (R2-C, res+intensity, random control)", "rsna/summary_imageretrieval_128harmonized_random.json"),
    ("RSNA",   "Image-Retrieval (R2-C, pool@96px)",         "rsna/summary_imageretrieval_96.json"),
    ("RSNA",   "Image-Retrieval (R2-C, pool@192px)",        "rsna/summary_imageretrieval_192.json"),
    ("CT-ICH", "MedGemma + Random",                         "ct_ich/summary_medgemma_random.json"),
    ("RSNA",   "MedGemma + Random",                         "rsna/summary_medgemma_random.json"),
    ("CT-ICH", "Qwen3-VL + Contrastive ICL",                 "ct_ich/summary_qwen_contrastive.json"),
    ("RSNA",   "Qwen3-VL + Contrastive ICL",                 "rsna/summary_qwen_contrastive.json"),
    ("CT-ICH", "MedGemma + Contrastive ICL",                 "ct_ich/summary_medgemma_contrastive.json"),
    ("RSNA",   "MedGemma + Contrastive ICL",                 "rsna/summary_medgemma_contrastive.json"),
]


def load_runs():
    rows = []
    for dataset, method, rel_path in RUNS:
        path = DATA_DIR / rel_path
        if not path.is_file():
            print(f"[skip] {dataset} / {method}: {rel_path} not found yet")
            continue
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        rows.append((dataset, method, summary))
    return rows


def render_markdown(rows):
    lines = [
        "# Method x Dataset comparison — Any-hemorrhage F1",
        "",
        "| Dataset | Method | n | F1 | 95% CI | P | R | Dangerous FN rate | Parse fail | Refusal |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, method, s in rows:
        a = s["any_hemorrhage"]
        d = s["dangerous_fn"]
        ci_lo, ci_hi = a["f1_ci95"]
        lines.append(
            f"| {dataset} | {method} | {s['n']} | {a['f1']:.3f} | "
            f"[{ci_lo:.3f}, {ci_hi:.3f}] | {a['precision']:.3f} | {a['recall']:.3f} | "
            f"{d['rate_of_hemorrhage_slices']:.1%} ({d['n']}/{d['n'] + a['tp']}) | "
            f"{s['parse_failure_rate']:.1%} | {s['refusal_rate']:.1%} |"
        )
    lines.append("")
    lines.append(
        "CI = 95% bootstrap percentile interval (2000 resamples, image-level, "
        "see metrics.py::bootstrap_f1_ci). A CI that does not cross 0 means "
        "the F1 is distinguishable from the floor effect (F1=0); a CI that "
        "hugs 0 means the point estimate is statistically indistinguishable "
        "from no effect, regardless of how it looks at face value."
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Build the {method}x{dataset} comparison table")
    ap.add_argument("--out", default=str(DATA_DIR / "comparison.md"))
    args = ap.parse_args()

    rows = load_runs()
    if not rows:
        print("[FAIL] no summaries found.")
        return

    md = render_markdown(rows)
    print(md)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
