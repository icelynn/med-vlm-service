# -*- coding: utf-8 -*-
"""
Paired bootstrap significance report — Any-hemorrhage F1, A (baseline) vs B (contender).
==========================================================================================

Every paired comparison in this project's writeups (R1 vs No-RAG, R2 vs its
random-exemplar control, two-stage vs R1, ...) was run as an ad-hoc inline
Python snippet over ssh -- not reproducible from a single command, and easy
to typo a path/column name. This wraps metrics.paired_bootstrap_f1_diff in a
CLI so every p-value quoted in README/docs has a command that regenerates it.

CPU-only, reads results.jsonl files that already exist -- no re-inference.

Usage
-----
    python significance_report.py --manifest data/rsna/manifest.csv \
        --a data/rsna/results_rag.jsonl          --label-a "Text-RAG (R1)" \
        --b data/rsna/results_rag_twostage.jsonl --label-b "Text-RAG two-stage"
"""

import argparse
import csv
import json
from pathlib import Path

from metrics import paired_bootstrap_f1_diff

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent


def load_gt(manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {r["image_file"]: int(r["any_hemorrhage"]) for r in csv.DictReader(f)}


def load_pred(results_path):
    preds = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                preds[r["image_file"]] = r["pred_any_hem"]
    return preds


def main():
    ap = argparse.ArgumentParser(
        description="Paired bootstrap F1 significance: A (baseline) vs B (contender)")
    ap.add_argument("--manifest", required=True, help="ground-truth manifest.csv")
    ap.add_argument("--a", required=True, help="baseline results.jsonl")
    ap.add_argument("--b", required=True, help="contender results.jsonl")
    ap.add_argument("--label-a", default=None, help="display name for --a (default: filename)")
    ap.add_argument("--label-b", default=None, help="display name for --b (default: filename)")
    ap.add_argument("--uncertain", default="negative",
                    choices=["negative", "positive", "ignore"],
                    help="how to score model hedges (-1), see metrics.binarize")
    ap.add_argument("--out", default=None, help="optional path to write the result as JSON")
    args = ap.parse_args()

    label_a = args.label_a or Path(args.a).name
    label_b = args.label_b or Path(args.b).name

    gt_map = load_gt(args.manifest)
    pred_a_map = load_pred(args.a)
    pred_b_map = load_pred(args.b)
    common = sorted(set(gt_map) & set(pred_a_map) & set(pred_b_map))
    if not common:
        print("[FAIL] no image_file in common across manifest/--a/--b — check paths.")
        return

    gt = [gt_map[k] for k in common]
    pred_a = [pred_a_map[k] for k in common]
    pred_b = [pred_b_map[k] for k in common]

    result = paired_bootstrap_f1_diff(gt, pred_a, pred_b, uncertain_policy=args.uncertain)
    ci_lo, ci_hi = result["ci95"]

    print(f"n={len(common)}  uncertain_policy={args.uncertain}")
    print(f"  {label_a} (baseline)  : F1={result['f1_a']:.4f}")
    print(f"  {label_b} (contender) : F1={result['f1_b']:.4f}")
    print(f"  diff (b - a) = {result['diff']:+.4f}  "
          f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  p={result['p_value']:.4f}")

    if args.out:
        out_record = {
            "n": len(common), "uncertain_policy": args.uncertain,
            "label_a": label_a, "label_b": label_b,
            "f1_a": result["f1_a"], "f1_b": result["f1_b"],
            "diff": result["diff"], "ci95": list(result["ci95"]),
            "p_value": result["p_value"],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_record, f, ensure_ascii=False, indent=2)
        print(f"\n[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
