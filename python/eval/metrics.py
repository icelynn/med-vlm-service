# -*- coding: utf-8 -*-
"""
Evaluation metrics — per-pathology precision / recall / F1 + macro-F1.
======================================================================
Run:
    python python/eval/metrics.py

The self-test at the bottom uses a hand-calc example. When your implementation
is correct you will see "ALL TESTS PASSED".

Classes are the intracranial-hemorrhage subtypes (Intraparenchymal,
Intraventricular, Subarachnoid, Epidural, Subdural). The metric math is
class-agnostic, so this is identical to the chest-X-ray version — only the
class list changed.

Label encoding (per class, per slice):
    1   = present (hemorrhage of this subtype)
    0   = absent
   -1   = uncertain  (model hedged, e.g. "possible ...")
   None = not mentioned / not parsed  (treat as absent unless you decide otherwise)

Keep it plain Python + stdlib. No numpy needed for the core logic.
"""

import random
from collections import namedtuple

PRF1 = namedtuple("PRF1", ["precision", "recall", "f1", "tp", "fp", "fn"])


def binarize(label, uncertain_policy="negative"):
    """Map a raw label {1, 0, -1, None} to a binary {1, 0} for scoring.

    uncertain_policy:
        "negative" -> -1 becomes 0
        "positive" -> -1 becomes 1
        "ignore"   -> return None (caller must skip this entry)
    None (not mentioned) always maps to 0.
    """
    if label is None:
        return 0
    if label == 1:
        return 1
    if label == 0:
        return 0
    # label == -1 (uncertain)
    if uncertain_policy == "positive":
        return 1
    if uncertain_policy == "ignore":
        return None
    return 0  # "negative"


def per_class_counts(gt, pred, uncertain_policy="negative"):
    """Given two equal-length lists of raw labels for ONE class
    (gt[i], pred[i] for study i), return (tp, fp, fn).

    Use binarize() on each entry. If uncertain_policy == "ignore" and either
    gt[i] or pred[i] binarizes to None, skip that study.
    """
    tp = fp = fn = 0
    for g, p in zip(gt, pred):
        g_bin = binarize(g, uncertain_policy)
        p_bin = binarize(p, uncertain_policy)
        if g_bin is None or p_bin is None:
            continue
        if g_bin == 1 and p_bin == 1:
            tp += 1
        elif g_bin == 0 and p_bin == 1:
            fp += 1
        elif g_bin == 1 and p_bin == 0:
            fn += 1
    return tp, fp, fn


def prf1(tp, fp, fn):
    """Return PRF1(precision, recall, f1, tp, fp, fn).

    Division-by-zero rule: if a denominator is 0, that metric is 0.0
    (so a class with tp=fp=fn=0 yields precision=recall=f1=0.0).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return PRF1(precision, recall, f1, tp, fp, fn)


def bootstrap_f1_ci(gt, pred, uncertain_policy="negative", n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for F1 on ONE class, resampling studies (the
    list index) with replacement.

    Small eval sets (CT-ICH/RSNA ~150 slices) give noisy point estimates --
    a wide CI here is the honest signal that the point F1 isn't well
    determined, not a bug in this function.

    Returns (lo, hi) at the (1-alpha) level, e.g. alpha=0.05 -> 95% CI.
    """
    rng = random.Random(seed)
    n = len(gt)
    boot_f1s = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gt[i] for i in idx]
        p = [pred[i] for i in idx]
        tp, fp, fn = per_class_counts(g, p, uncertain_policy)
        boot_f1s.append(prf1(tp, fp, fn).f1)
    boot_f1s.sort()
    lo = boot_f1s[int((alpha / 2) * n_boot)]
    hi = boot_f1s[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return lo, hi


def paired_bootstrap_f1_diff(gt, pred_a, pred_b, uncertain_policy="negative",
                             n_boot=2000, seed=42):
    """Paired bootstrap for the F1 difference between two methods (B - A) on the
    SAME eval set, scored against the same ground truth.

    Both methods are evaluated on the *same* resampled indices each iteration,
    so the resampling noise that's common to both cancels -- this has more
    power to detect a real difference than asking whether two independent
    per-method CIs (bootstrap_f1_ci) happen to overlap. Works regardless of
    whether the baseline F1 is 0 (e.g. CT-ICH) or non-zero (e.g. RSNA).

    pred_a is the baseline (e.g. R1 text-RAG), pred_b the contender (e.g. R2
    image retrieval); a positive diff means B scored higher.

    Returns dict:
        f1_a, f1_b   : point F1 of each method on the full set
        diff         : f1_b - f1_a (point estimate)
        ci95         : (lo, hi) 95% percentile interval of the bootstrap diff
        p_value      : two-sided, fraction of bootstrap diffs on the "no
                       improvement" side x2 (clamped to <= 1.0) -- small means
                       B is significantly different from A.
    """
    rng = random.Random(seed)
    n = len(gt)

    def _f1(g, p):
        tp, fp, fn = per_class_counts(g, p, uncertain_policy)
        return prf1(tp, fp, fn).f1

    f1_a = _f1(gt, pred_a)
    f1_b = _f1(gt, pred_b)
    point_diff = f1_b - f1_a

    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gt[i] for i in idx]
        da = _f1(g, [pred_a[i] for i in idx])
        db = _f1(g, [pred_b[i] for i in idx])
        diffs.append(db - da)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(int(0.975 * n_boot), n_boot - 1)]

    # Two-sided p-value: how often the bootstrap diff lands on the opposite
    # side of 0 from the observed effect (the side that would mean "no
    # improvement"), doubled for two-sidedness.
    n_wrong_side = sum(1 for d in diffs if (d <= 0 if point_diff > 0 else d >= 0))
    p_value = min(1.0, 2 * n_wrong_side / n_boot)

    return {"f1_a": f1_a, "f1_b": f1_b, "diff": point_diff,
            "ci95": (lo, hi), "p_value": p_value}


def evaluate(gt_by_class, pred_by_class, uncertain_policy="negative"):
    """Top-level metric.

    gt_by_class / pred_by_class: dict {class_name: [labels per study]}.
    Both dicts have the same keys and equal-length value lists.

    Return (per_class, macro_f1) where:
        per_class : dict {class_name: PRF1}
        macro_f1  : float = mean of the per-class F1 over all classes
    """
    per_class = {}
    for cls in gt_by_class:
        tp, fp, fn = per_class_counts(gt_by_class[cls], pred_by_class[cls], uncertain_policy)
        per_class[cls] = prf1(tp, fp, fn)
    macro_f1 = sum(r.f1 for r in per_class.values()) / len(per_class) if per_class else 0.0
    return per_class, macro_f1


# --------------------------------------------------------------------------
# Self-test — do not edit.
# --------------------------------------------------------------------------
def _approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


def _run_self_test():
    gt = {
        "Intraparenchymal": [1, 1, 0, 1, 0],
        "Intraventricular": [0, 1, 0, 1, 1],
        "Subarachnoid":     [0, 0, 1, 0, 0],
    }
    pred = {
        "Intraparenchymal": [1, 0, 1, 1, 0],
        "Intraventricular": [0, 1, 0, 0, 1],
        "Subarachnoid":     [1, 0, 1, 0, 0],
    }
    expected_f1 = {"Intraparenchymal": 0.6667, "Intraventricular": 0.8, "Subarachnoid": 0.6667}
    expected_counts = {  # (tp, fp, fn)
        "Intraparenchymal": (2, 1, 1),
        "Intraventricular": (2, 0, 1),
        "Subarachnoid":     (1, 1, 0),
    }
    expected_macro = 0.7111

    per_class, macro = evaluate(gt, pred, uncertain_policy="negative")

    ok = True
    for cls in gt:
        r = per_class[cls]
        etp, efp, efn = expected_counts[cls]
        if (r.tp, r.fp, r.fn) != (etp, efp, efn):
            print(f"  FAIL {cls}: counts {(r.tp, r.fp, r.fn)} != expected {(etp, efp, efn)}")
            ok = False
        if not _approx(r.f1, expected_f1[cls]):
            print(f"  FAIL {cls}: F1 {r.f1:.4f} != expected {expected_f1[cls]}")
            ok = False
        else:
            print(f"  ok   {cls}: P={r.precision:.3f} R={r.recall:.3f} F1={r.f1:.3f}")
    if not _approx(macro, expected_macro):
        print(f"  FAIL macro-F1 {macro:.4f} != expected {expected_macro}")
        ok = False
    else:
        print(f"  ok   macro-F1 = {macro:.4f}")

    print("\n" + ("ALL TESTS PASSED" if ok else "TESTS FAILED — keep going"))
    return ok


def _test_bootstrap_ci():
    gt =   [1, 1, 0, 1, 0, 1, 0, 0, 1, 0]
    pred = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
    point = prf1(*per_class_counts(gt, pred)).f1
    lo, hi = bootstrap_f1_ci(gt, pred, n_boot=2000, seed=42)
    ok = lo <= point <= hi and 0.0 <= lo <= hi <= 1.0
    print(f"  {'ok' if ok else 'FAIL'}   point F1={point:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")
    print("\n" + ("ALL TESTS PASSED" if ok else "TESTS FAILED — keep going"))
    return ok


def _test_paired_bootstrap():
    # 20 studies. pred_a (baseline) gets none of the positives; pred_b (contender)
    # catches several of them with no false positives -> B should be clearly,
    # significantly better, so diff > 0 and p_value small.
    gt     = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    pred_a = [0] * 20
    pred_b = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    r = paired_bootstrap_f1_diff(gt, pred_a, pred_b, n_boot=2000, seed=42)
    ok = (
        r["f1_a"] == 0.0 and r["f1_b"] > 0.0          # baseline floors, contender doesn't
        and r["diff"] > 0.0                            # improvement is positive
        and r["ci95"][0] <= r["diff"] <= r["ci95"][1]  # point diff inside its own CI
        and r["p_value"] < 0.05                        # and it's significant
    )
    print(f"  {'ok' if ok else 'FAIL'}   diff={r['diff']:.3f} "
          f"CI=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] p={r['p_value']:.4f}")

    # Identical predictions -> zero diff, definitely not significant.
    r2 = paired_bootstrap_f1_diff(gt, pred_b, pred_b, n_boot=2000, seed=42)
    ok2 = r2["diff"] == 0.0 and r2["p_value"] >= 0.05
    print(f"  {'ok' if ok2 else 'FAIL'}   identical preds: diff={r2['diff']:.3f} p={r2['p_value']:.4f}")

    passed = ok and ok2
    print("\n" + ("ALL TESTS PASSED" if passed else "TESTS FAILED — keep going"))
    return passed


if __name__ == "__main__":
    try:
        _run_self_test()
        _test_bootstrap_ci()
        _test_paired_bootstrap()
    except NotImplementedError:
        print("metrics.py not implemented yet — fill in the TODO functions and re-run.")
