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

    python significance_report.py --self-test   # offline, no GPU, no external files
"""

import argparse
import csv
import json
import tempfile
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


def _self_test():
    """Offline, zero-GPU check of the file-loading + join glue (the part
    metrics.py's own self-test doesn't cover). Writes synthetic manifest/
    results files to a temp dir, runs the same load_gt/load_pred/join path
    main() uses, and checks the join excludes files not common to all three
    sources, then sanity-checks two known-shape comparisons."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest_path = tmp / "manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_file", "any_hemorrhage"])
            # img6 only appears in the manifest -- must be excluded from the join.
            for name, label in [("img1", 1), ("img2", 0), ("img3", 1),
                                ("img4", 0), ("img5", 1), ("img6", 0)]:
                w.writerow([name, label])

        def write_results(path, preds):
            with open(path, "w", encoding="utf-8") as f:
                for name, pred in preds.items():
                    f.write(json.dumps({"image_file": name, "pred_any_hem": pred}) + "\n")

        # Baseline floors every positive; contender catches all 3 positives, no FP.
        # Neither mentions img6, so the join must drop it -> n=5, not 6.
        path_a = tmp / "results_a.jsonl"
        path_b = tmp / "results_b.jsonl"
        write_results(path_a, {"img1": 0, "img2": 0, "img3": 0, "img4": 0, "img5": 0})
        write_results(path_b, {"img1": 1, "img2": 0, "img3": 1, "img4": 0, "img5": 1})

        gt_map = load_gt(manifest_path)
        pred_a_map = load_pred(path_a)
        pred_b_map = load_pred(path_b)
        common = sorted(set(gt_map) & set(pred_a_map) & set(pred_b_map))

        ok_join = common == ["img1", "img2", "img3", "img4", "img5"]
        print(f"  {'ok' if ok_join else 'FAIL'}   join excludes img6 (manifest-only): "
              f"common={common}")

        gt = [gt_map[k] for k in common]
        pred_a = [pred_a_map[k] for k in common]
        pred_b = [pred_b_map[k] for k in common]

        result = paired_bootstrap_f1_diff(gt, pred_a, pred_b)
        ok_diff = (result["f1_a"] == 0.0 and result["f1_b"] == 1.0
                  and result["diff"] == 1.0 and result["p_value"] < 0.05)
        print(f"  {'ok' if ok_diff else 'FAIL'}   floor-vs-perfect: f1_a={result['f1_a']:.3f} "
              f"f1_b={result['f1_b']:.3f} diff={result['diff']:.3f} p={result['p_value']:.4f}")

        result_identical = paired_bootstrap_f1_diff(gt, pred_b, pred_b)
        ok_identical = result_identical["diff"] == 0.0 and result_identical["p_value"] >= 0.05
        print(f"  {'ok' if ok_identical else 'FAIL'}   identical preds: "
              f"diff={result_identical['diff']:.3f} p={result_identical['p_value']:.4f}")

        passed = ok_join and ok_diff and ok_identical
        print("\n" + ("ALL TESTS PASSED" if passed else "TESTS FAILED — keep going"))
        return passed


def main():
    ap = argparse.ArgumentParser(
        description="Paired bootstrap F1 significance: A (baseline) vs B (contender)")
    ap.add_argument("--manifest", default=None, help="ground-truth manifest.csv")
    ap.add_argument("--a", default=None, help="baseline results.jsonl")
    ap.add_argument("--b", default=None, help="contender results.jsonl")
    ap.add_argument("--label-a", default=None, help="display name for --a (default: filename)")
    ap.add_argument("--label-b", default=None, help="display name for --b (default: filename)")
    ap.add_argument("--uncertain", default="negative",
                    choices=["negative", "positive", "ignore"],
                    help="how to score model hedges (-1), see metrics.binarize")
    ap.add_argument("--out", default=None, help="optional path to write the result as JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline self-test (no GPU, no external files) and exit")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not (args.manifest and args.a and args.b):
        ap.error("--manifest, --a, and --b are required (or pass --self-test)")

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
