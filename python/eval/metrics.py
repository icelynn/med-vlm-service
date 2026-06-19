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

from collections import namedtuple

# present is the "positive" class for every pathology.
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


if __name__ == "__main__":
    try:
        _run_self_test()
    except NotImplementedError:
        print("metrics.py not implemented yet — fill in the TODO functions and re-run.")
