# -*- coding: utf-8 -*-
"""
Retrieval sentinel check -- decides B vs C for Week4 R2's image-retrieval pool.
================================================================================

Background (核心難題 12 / Week4 計畵 §3b, §0g): the original R2 reference-pool
plan (CT-ICH patient split) turned out to be mathematically infeasible (only
28 patients / 5 hemorrhage slices left after carving out the eval set,
IVH/EDH/SDH at zero). The two remaining options are:

  B = RSNA pool -> CT-ICH eval   cross-dataset, structurally zero-leakage,
                                  but retrieval has to cross a different
                                  resolution/windowing visual gap -- it might
                                  match "dataset style" instead of "lesion"
  C = RSNA pool -> RSNA eval     visually consistent pool/eval, but needs
                                  RSNA patient-level metadata to avoid
                                  leakage and moves R2's eval target off the
                                  CT-ICH narrative

This script is the pre-registered test that turns "which one" into a
measured question instead of a preference: embed every CT-ICH eval
hemorrhage-positive slice (query) and every RSNA hemorrhage-positive slice
(pool) with BiomedCLIP, and ask "do CT-ICH's top-k visual nearest neighbours
in the RSNA pool actually share a hemorrhage subtype label more often than
chance?" Chance is a permutation null: n_trials times, for each query draw k
random pool slices and recompute the same hit rate -- this implicitly bakes
in the pool's true per-subtype prevalence, so no separate prevalence
correction is needed.

Decision rule (fixed BEFORE running on real data, see Week4 plan §0g):
    real hit_rate@k  >=  95th percentile of the permutation null  -> go B
    else: try histogram-matching the pool images once, recompute   -> go B
    still short                                                     -> go C

Run modes
---------
    python retrieval_sentinel.py --self-test
        Pure-logic check on synthetic embeddings/labels. No torch, no
        open_clip, no images -- safe to run anywhere (incl. a plain dev box)
        to validate the hit-rate / permutation / decision math before
        spending GPU time on real embeddings.

    python retrieval_sentinel.py
        Real run: embeds CT-ICH eval hemorrhage-positive slices (query) and
        RSNA hemorrhage-positive slices (pool -- defaults to the
        already-fetched RSNA eval set, REUSED here as the retrieval pool:
        the sentinel's query is a different dataset, so pool/query patient
        overlap is structurally impossible no matter which RSNA rows are
        used, and reusing them needs zero new Kaggle fetches) with
        BiomedCLIP, runs the protocol above, and prints/writes the verdict.
        Needs `pip install open_clip_torch scikit-image` and a GPU or
        patience (CPU works, just slower).
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBTYPE_KEYS = ["IPH", "IVH", "SAH", "EDH", "SDH"]

DEFAULT_QUERY_MANIFEST = PROJECT_ROOT / "data" / "ct_ich" / "manifest.csv"
DEFAULT_QUERY_IMAGES = PROJECT_ROOT / "data" / "ct_ich" / "images"
DEFAULT_POOL_MANIFEST = PROJECT_ROOT / "data" / "rsna" / "manifest.csv"
DEFAULT_POOL_IMAGES = PROJECT_ROOT / "data" / "rsna" / "images"


# --------------------------------------------------------------------------
# Core, dependency-free logic (embeddings/labels in, verdict out). Covered by
# --self-test below with synthetic data -- no torch/open_clip needed here.
# --------------------------------------------------------------------------

def load_label_sets(manifest_path, positive_only=True):
    """Read a manifest.csv (CT-ICH or RSNA schema) and return a list of
    (image_file, frozenset(subtype keys present)) for hemorrhage-positive
    rows (any_hemorrhage == '1')."""
    rows_out = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if positive_only and row["any_hemorrhage"] != "1":
                continue
            labels = frozenset(k for k in SUBTYPE_KEYS if row[k] == "1")
            rows_out.append((row["image_file"], labels))
    return rows_out


def shares_label(a, b):
    return len(a & b) > 0


def cosine_topk_indices(query_vec, pool_matrix, k):
    """Indices (descending similarity) of the top-k rows of pool_matrix
    closest to query_vec, by cosine similarity. Assumes rows are already
    L2-normalized (so cosine == dot product); doesn't re-normalize, so the
    same function works on synthetic self-test vectors as long as callers
    are consistent about normalization."""
    sims = pool_matrix @ query_vec
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    return top[np.argsort(-sims[top])]


def hit_rate_at_k(query_embeddings, query_labels, pool_embeddings, pool_labels, k=5):
    """Mean, over queries, of the fraction of the top-k pool nearest
    neighbours (by cosine similarity) that share >=1 subtype label with the
    query."""
    hits = []
    for q_emb, q_labels in zip(query_embeddings, query_labels):
        idx = cosine_topk_indices(q_emb, pool_embeddings, k)
        matched = sum(1 for i in idx if shares_label(q_labels, pool_labels[i]))
        hits.append(matched / k)
    return float(np.mean(hits))


def permutation_null(query_labels, pool_labels, k=5, n_trials=1000, seed=42):
    """n_trials times: for each query, draw k random pool indices (no
    embeddings involved) and compute the same hit-rate metric. This is the
    "chance" distribution -- it already bakes in the pool's true
    per-subtype prevalence (a query with a common label has a higher chance
    hit rate by construction), so no separate prevalence correction is
    needed."""
    rng = random.Random(seed)
    n_pool = len(pool_labels)
    null_samples = []
    for _ in range(n_trials):
        hits = []
        for q_labels in query_labels:
            idx = rng.sample(range(n_pool), min(k, n_pool))
            matched = sum(1 for i in idx if shares_label(q_labels, pool_labels[i]))
            hits.append(matched / k)
        null_samples.append(float(np.mean(hits)))
    return null_samples


def sentinel_verdict(real_hit_rate, null_samples, percentile=95):
    """Pre-registered pass/fail: real_hit_rate >= the given percentile of the
    permutation null."""
    threshold = float(np.percentile(null_samples, percentile))
    return {
        "real_hit_rate": real_hit_rate,
        "null_mean": float(np.mean(null_samples)),
        "null_threshold_p95": threshold,
        "passed": real_hit_rate >= threshold,
    }


# --------------------------------------------------------------------------
# Real embedding backend (BiomedCLIP via open_clip, shared with
# build_image_index.py / providers.py -- see biomedclip_embed.py). Imported
# lazily inside run_real() so --self-test never needs torch/open_clip
# installed.
# --------------------------------------------------------------------------

def histogram_match_images(image_paths, reference_path, out_dir):
    """Contrast/windowing harmonization fallback (§0g): match each pool
    image's histogram to one reference query image so a B verdict isn't
    blocked by a contrast/windowing mismatch rather than a genuine lack of
    visual signal. Uses skimage's exact histogram matching (not the full
    Nyul piecewise-linear landmark method, which needs a training set of
    references) -- close enough for this one-shot recheck."""
    from PIL import Image
    from skimage.exposure import match_histograms

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = np.array(Image.open(reference_path).convert("L"))
    matched_paths = []
    for p in image_paths:
        img = np.array(Image.open(p).convert("L"))
        matched = match_histograms(img, reference, channel_axis=None)
        out_path = out_dir / Path(p).name
        Image.fromarray(matched.astype(np.uint8)).save(out_path)
        matched_paths.append(out_path)
    return matched_paths


# --------------------------------------------------------------------------
# Real-data run.
# --------------------------------------------------------------------------

def _load_set(manifest_path, images_dir):
    rows = load_label_sets(manifest_path)
    paths = [Path(images_dir) / fname for fname, _ in rows]
    labels = [labels for _, labels in rows]
    return paths, labels


def run_real(args):
    print(f"[load  ] query manifest: {args.query_manifest}")
    query_paths, query_labels = _load_set(args.query_manifest, args.query_images)
    print(f"[load  ] pool manifest:  {args.pool_manifest}")
    pool_paths, pool_labels = _load_set(args.pool_manifest, args.pool_images)
    print(f"[info  ] {len(query_paths)} query slices, {len(pool_paths)} pool slices")

    from biomedclip_embed import embed_images

    print("[embed ] query slices...")
    query_emb = embed_images(query_paths, device=args.device)
    print("[embed ] pool slices...")
    pool_emb = embed_images(pool_paths, device=args.device)

    real_hit_rate = hit_rate_at_k(query_emb, query_labels, pool_emb, pool_labels, k=args.k)
    null_samples = permutation_null(query_labels, pool_labels, k=args.k,
                                     n_trials=args.n_trials, seed=args.seed)
    verdict = sentinel_verdict(real_hit_rate, null_samples)
    print(f"[result] hit_rate@{args.k} = {verdict['real_hit_rate']:.3f}  "
          f"null mean = {verdict['null_mean']:.3f}  "
          f"null p95 = {verdict['null_threshold_p95']:.3f}  "
          f"-> {'PASS (real >= null p95)' if verdict['passed'] else 'FAIL'}")

    decision = "B"
    histogram_matched = False
    if not verdict["passed"]:
        print("[retry ] below threshold -- trying histogram-matching the pool "
              "images once before falling back to C")
        matched_dir = Path(args.pool_images).parent / "images_histmatch"
        matched_paths = histogram_match_images(pool_paths, query_paths[0], matched_dir)
        print("[embed ] histogram-matched pool slices...")
        pool_emb_matched = embed_images(matched_paths, device=args.device)
        real_hit_rate_2 = hit_rate_at_k(query_emb, query_labels, pool_emb_matched, pool_labels, k=args.k)
        verdict2 = sentinel_verdict(real_hit_rate_2, null_samples)
        print(f"[result] (post hist-match) hit_rate@{args.k} = {verdict2['real_hit_rate']:.3f}  "
              f"-> {'PASS' if verdict2['passed'] else 'FAIL'}")
        histogram_matched = True
        decision = "B" if verdict2["passed"] else "C"
        verdict["after_histogram_match"] = verdict2

    verdict["decision"] = decision
    verdict["histogram_matched"] = histogram_matched
    verdict["k"] = args.k
    verdict["n_trials"] = args.n_trials
    verdict["n_query"] = len(query_paths)
    verdict["n_pool"] = len(pool_paths)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(verdict, indent=2))
        print(f"[done  ] wrote {out_path}")
    print(f"\n[DECISION] -> {decision}")
    return verdict


# --------------------------------------------------------------------------
# Self-test -- do not edit lightly; this is what proves the decision math is
# right before it ever touches real embeddings.
# --------------------------------------------------------------------------

def _make_cluster_data(n_per_label=20, dim=16, signal=True, seed=0):
    """Synthetic query/pool embeddings + labels. With signal=True, each
    subtype gets its own random center in embedding space so same-label
    items cluster together (a stand-in for "BiomedCLIP finds real visual
    signal"). With signal=False, every label shares the same (zero) center,
    so embeddings carry no label information (a stand-in for "no usable
    visual signal" / pure noise)."""
    rng = np.random.default_rng(seed)
    centers = {lab: rng.normal(size=dim) for lab in SUBTYPE_KEYS}
    query_emb, query_labels, pool_emb, pool_labels = [], [], [], []
    for lab in SUBTYPE_KEYS:
        center = centers[lab] if signal else np.zeros(dim)
        for _ in range(n_per_label):
            v = center + rng.normal(scale=0.3, size=dim)
            v = v / np.linalg.norm(v)
            query_emb.append(v)
            query_labels.append(frozenset([lab]))
        for _ in range(n_per_label):
            v = center + rng.normal(scale=0.3, size=dim)
            v = v / np.linalg.norm(v)
            pool_emb.append(v)
            pool_labels.append(frozenset([lab]))
    return np.array(query_emb), query_labels, np.array(pool_emb), pool_labels


def _self_test():
    print("[self-test] hit_rate_at_k / permutation_null / sentinel_verdict on "
          "synthetic data\n")

    # Scenario 1: real visual signal (same-label embeddings cluster) -> real
    # hit rate should clear the permutation-null p95 threshold -> B.
    q_emb, q_lab, p_emb, p_lab = _make_cluster_data(signal=True, seed=1)
    real = hit_rate_at_k(q_emb, q_lab, p_emb, p_lab, k=5)
    null = permutation_null(q_lab, p_lab, k=5, n_trials=300, seed=1)
    v = sentinel_verdict(real, null)
    ok1 = v["passed"] and real > v["null_mean"]
    print(f"  {'ok' if ok1 else 'FAIL'}   signal=True : real={real:.3f} "
          f"null_mean={v['null_mean']:.3f} p95={v['null_threshold_p95']:.3f} "
          f"passed={v['passed']}")

    # Scenario 2: no visual signal (embeddings carry no label information)
    # -> real hit rate should land at chance, i.e. NOT clear p95 -> C (after
    # the hist-match retry, which this self-test doesn't cover since it
    # needs skimage + real images).
    q_emb2, q_lab2, p_emb2, p_lab2 = _make_cluster_data(signal=False, seed=2)
    real2 = hit_rate_at_k(q_emb2, q_lab2, p_emb2, p_lab2, k=5)
    null2 = permutation_null(q_lab2, p_lab2, k=5, n_trials=300, seed=2)
    v2 = sentinel_verdict(real2, null2)
    ok2 = not v2["passed"]
    print(f"  {'ok' if ok2 else 'FAIL'}   signal=False: real={real2:.3f} "
          f"null_mean={v2['null_mean']:.3f} p95={v2['null_threshold_p95']:.3f} "
          f"passed={v2['passed']}")

    ok3 = all(0.0 <= x <= 1.0 for x in null) and all(0.0 <= x <= 1.0 for x in null2)
    print(f"  {'ok' if ok3 else 'FAIL'}   null samples within [0, 1]")

    passed = ok1 and ok2 and ok3
    print("\n" + ("ALL TESTS PASSED" if passed else "TESTS FAILED -- keep going"))
    return passed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                     help="run the dependency-free logic self-test and exit")
    ap.add_argument("--query-manifest", default=str(DEFAULT_QUERY_MANIFEST))
    ap.add_argument("--query-images", default=str(DEFAULT_QUERY_IMAGES))
    ap.add_argument("--pool-manifest", default=str(DEFAULT_POOL_MANIFEST))
    ap.add_argument("--pool-images", default=str(DEFAULT_POOL_IMAGES))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(
        PROJECT_ROOT / "data" / "rag" / "retrieval_sentinel_result.json"))
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(0 if _self_test() else 1)

    run_real(args)


if __name__ == "__main__":
    main()
