# -*- coding: utf-8 -*-
"""
Score the No-RAG baseline — Any-hemorrhage F1 + per-subtype + parse rates.
==========================================================================

Joins results.jsonl (model predictions) with manifest.csv (ground truth) on
image_file, then reports:

  * HEADLINE:  Any-hemorrhage binary precision / recall / F1
  * MUST:      parse-failure rate, refusal rate (first-class)
  * STRETCH:   per-subtype P/R/F1 + macro-F1 (with support counts)

CPU-only. Re-run with a different --uncertain policy to see how hedged model
answers swing the numbers — no re-inference needed.

Usage
-----
    python score_baseline.py
    python score_baseline.py --uncertain positive
    python score_baseline.py --results results_v2.jsonl
"""

import argparse
import csv
import json
from pathlib import Path

from metrics import evaluate, prf1, per_class_counts
from parse_answer import SUBTYPES

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ct_ich"


def load_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image_file"]: r for r in csv.DictReader(f)}


def load_results(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser(description="Score CT-ICH No-RAG baseline")
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.csv"))
    ap.add_argument("--results", default=str(DATA_DIR / "results.jsonl"))
    ap.add_argument("--uncertain", default="negative",
                    choices=["negative", "positive", "ignore"],
                    help="how to score model hedges (-1): negative/positive/ignore")
    ap.add_argument("--out", default=str(DATA_DIR / "baseline_summary.json"))
    args = ap.parse_args()

    gt = load_manifest(Path(args.manifest))
    results = load_results(Path(args.results))
    # Align on image_file (score only slices that were actually inferred).
    results = [r for r in results if r["image_file"] in gt]
    n = len(results)
    if n == 0:
        print("[FAIL] no results joined with manifest — check paths.")
        return

    # ---- parse-failure / refusal rates (first-class) --------------------
    n_failed = sum(not r["parsed"] for r in results)
    n_refused = sum(r["refused"] for r in results)
    method_counts = {}
    for r in results:
        method_counts[r["method"]] = method_counts.get(r["method"], 0) + 1

    # ---- HEADLINE: Any-hemorrhage binary --------------------------------
    any_gt = [int(gt[r["image_file"]]["any_hemorrhage"]) for r in results]
    any_pred = [r["pred_any_hem"] for r in results]
    a_tp, a_fp, a_fn = per_class_counts(any_gt, any_pred, args.uncertain)
    any_f1 = prf1(a_tp, a_fp, a_fn)

    # ---- STRETCH: per-subtype P/R/F1 + macro-F1 -------------------------
    gt_by_class = {k: [int(gt[r["image_file"]][k]) for r in results] for k in SUBTYPES}
    pred_by_class = {k: [r[f"pred_{k}"] for r in results] for k in SUBTYPES}
    per_class, macro_f1 = evaluate(gt_by_class, pred_by_class, args.uncertain)
    support = {k: sum(v) for k, v in gt_by_class.items()}  # GT positives per class

    # ---- report ---------------------------------------------------------
    print(f"\n{'='*64}")
    print(f" CT-ICH No-RAG baseline   n={n}   uncertain_policy={args.uncertain}")
    print(f"{'='*64}")

    print(f"\n[parse health]  (first-class metric)")
    print(f"  parse-failure rate : {n_failed}/{n} = {n_failed/n:.1%}")
    print(f"  refusal rate       : {n_refused}/{n} = {n_refused/n:.1%}")
    print(f"  methods            : {method_counts}")

    print(f"\n[HEADLINE] Any-hemorrhage (binary triage)")
    print(f"  P={any_f1.precision:.3f}  R={any_f1.recall:.3f}  F1={any_f1.f1:.3f}"
          f"  (tp={a_tp} fp={a_fp} fn={a_fn})")

    print(f"\n[stretch] per-subtype")
    print(f"  {'subtype':<6} {'P':>6} {'R':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'sup':>4}")
    for k in SUBTYPES:
        r = per_class[k]
        print(f"  {k:<6} {r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
              f"{r.tp:>4} {r.fp:>4} {r.fn:>4} {support[k]:>4}")
    print(f"  macro-F1 = {macro_f1:.4f}")

    summary = {
        "n": n,
        "uncertain_policy": args.uncertain,
        "parse_failure_rate": n_failed / n,
        "refusal_rate": n_refused / n,
        "method_counts": method_counts,
        "any_hemorrhage": {"precision": any_f1.precision, "recall": any_f1.recall,
                           "f1": any_f1.f1, "tp": a_tp, "fp": a_fp, "fn": a_fn},
        "per_subtype": {k: {"precision": per_class[k].precision,
                            "recall": per_class[k].recall, "f1": per_class[k].f1,
                            "tp": per_class[k].tp, "fp": per_class[k].fp,
                            "fn": per_class[k].fn, "support": support[k]}
                        for k in SUBTYPES},
        "macro_f1": macro_f1,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[ok] summary -> {args.out}")


if __name__ == "__main__":
    main()
