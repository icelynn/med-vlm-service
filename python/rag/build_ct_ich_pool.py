# -*- coding: utf-8 -*-
"""
Build a CT-ICH-native R2 reference pool -- same dataset as the eval set,
instead of the cross-dataset RSNA pool every CT-ICH R2 run has used so far.

Background: every existing CT-ICH Image-Retrieval result (R2-B) retrieves
exemplars from a histogram-matched RSNA pool -- a structural cross-domain
confound that's never been isolated. This script builds the alternative:
a pool drawn from CT-ICH's OWN source data (same 75-patient PhysioNet
cohort, same window/render pipeline as prep_ct_ich.py).

Patient-disjoint pooling (zero overlap with the 47 patients used by the
150-image eval manifest) is NOT viable here: of the 75 patients, only 47
were sampled into the eval manifest specifically because the stratified
sampler prioritized hemorrhage-positive patients -- the 28 untouched
patients have only 5 hemorrhage slices total across all 5 subtypes
combined (3x IPH, 2x SAH; zero IVH/EDH/SDH), which can't support a usable
retrieval pool and specifically excludes EDH/SDH, the two subtypes that
dominate this project's hardest failure cases.

This script instead pools from slices NOT in the 150-image manifest,
drawn from the SAME 47 patients already used by eval (each patient's CT
volume has ~30 slices; the manifest only sampled ~3/patient). This is a
real, disclosed methodological compromise: pool and eval can share a
patient (different slice, different anatomical level), which is a softer
leakage risk than literal same-image overlap but is NOT the strict
study-disjoint guarantee build_rsna_pool.py achieves for RSNA. Report
results from this pool with that caveat explicit, not as equivalent to R2-C.

Usage
-----
    python build_ct_ich_pool.py --root data/ct_ich/_raw --n 128
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))
from prep_ct_ich import window_slice, SUBTYPES, find_root  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ct_ich"
RAW_DIR = DATA_DIR / "_raw"
FULL_INDEX = DATA_DIR / "full_index.csv"
EVAL_MANIFEST = DATA_DIR / "manifest.csv"
POOL_MANIFEST = DATA_DIR / "pool_manifest_native.csv"
POOL_IMAGES_DIR = DATA_DIR / "pool_images_native"

FIELDS = ["image_file", "patient", "slice", "any_hemorrhage", "is_normal",
          "present_subtypes", *SUBTYPES.values()]


def load_full_index(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["patient"] = int(r["patient"])
        r["slice"] = int(r["slice"])
        for k in SUBTYPES.values():
            r[k] = int(r[k])
        r["is_normal"] = r["is_normal"] in ("True", "1", "true")
        r["any_hemorrhage"] = int(r["any_hemorrhage"])
    return rows


def load_manifest_pairs(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {(int(r["patient"]), int(r["slice"])) for r in csv.DictReader(f)}


def stratified_sample(candidates, n, normal_frac, seed):
    """Same approach as prep_ct_ich.py / build_rsna_pool.py."""
    rng = random.Random(seed)
    normals = [c for c in candidates if c["is_normal"]]
    hemo = [c for c in candidates if not c["is_normal"]]
    rng.shuffle(normals)

    n_normal_target = min(len(normals), round(n * normal_frac))
    n_hemo_target = min(len(hemo), n - n_normal_target)
    deficit = n - (n_normal_target + n_hemo_target)
    if deficit > 0:
        n_normal_target = min(len(normals), n_normal_target + deficit)

    chosen_normals = normals[:n_normal_target]

    label_matrix = np.array([[c[k] for k in SUBTYPES.values()] for c in hemo])
    X_dummy = np.zeros((len(hemo), 1))
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=n_hemo_target / len(hemo), random_state=seed)
    _, idx = next(splitter.split(X_dummy, label_matrix))
    chosen_hemo = [hemo[i] for i in idx]

    chosen = chosen_normals + chosen_hemo
    rng.shuffle(chosen)
    return chosen


def render(chosen, root, wl, ww, size, images_dir):
    images_dir.mkdir(parents=True, exist_ok=True)
    by_patient = defaultdict(list)
    for r in chosen:
        by_patient[r["patient"]].append(r)

    written, missing = [], []
    for pid, recs in sorted(by_patient.items()):
        nii = root / "ct_scans" / f"{pid:03d}.nii"
        if not nii.is_file():
            missing.extend(recs)
            continue
        vol = nib.load(str(nii)).get_fdata()
        n_sl = vol.shape[2]
        for r in recs:
            idx = r["slice"] - 1
            if not (0 <= idx < n_sl):
                missing.append(r)
                continue
            img = window_slice(vol[:, :, idx], wl, ww)
            im = Image.fromarray(img)
            if size and im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            fname = f"pool_{pid:03d}_{r['slice']:03d}.png"
            im.save(images_dir / fname, "PNG")
            r["image_file"] = fname
            present = [k for k in SUBTYPES.values() if r[k]]
            r["present_subtypes"] = "; ".join(present)
            written.append(r)
    return written, missing


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ok  ] wrote {len(rows)} rows -> {path.relative_to(PROJECT_ROOT)}")


def summarize(rows):
    n = len(rows)
    n_norm = sum(r["is_normal"] for r in rows)
    print(f"\n[pool] n={n}  normal={n_norm} ({n_norm/n:.0%})  hemorrhage={n - n_norm}")
    for k in SUBTYPES.values():
        print(f"    {k}: {sum(r[k] for r in rows)}")
    print(f"    distinct patients: {len({r['patient'] for r in rows})}  "
          f"(eval manifest patients: same pool as eval -- disclosed caveat, see module docstring)")


def main():
    ap = argparse.ArgumentParser(description="Build CT-ICH-native R2 reference pool")
    ap.add_argument("--root", default=str(RAW_DIR))
    ap.add_argument("--full-index", default=str(FULL_INDEX))
    ap.add_argument("--eval-manifest", default=str(EVAL_MANIFEST))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--normal-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wl", type=float, default=40.0)
    ap.add_argument("--ww", type=float, default=120.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=str(POOL_MANIFEST))
    ap.add_argument("--images", default=str(POOL_IMAGES_DIR))
    args = ap.parse_args()

    root = find_root(args.root)
    full_index = load_full_index(Path(args.full_index))
    manifest_pairs = load_manifest_pairs(Path(args.eval_manifest))

    candidates = [r for r in full_index if (r["patient"], r["slice"]) not in manifest_pairs]
    print(f"[ok  ] {len(candidates)} candidate slices (same patients as eval, "
          f"excluding the {len(manifest_pairs)} slices already in the eval manifest)")

    chosen = stratified_sample(candidates, args.n, args.normal_frac, args.seed)
    print(f"[ok  ] sampled {len(chosen)}/{args.n} requested")

    written, missing = render(chosen, root, args.wl, args.ww, args.size, Path(args.images))
    if missing:
        print(f"[warn] {len(missing)} sampled slices had no NIfTI/slice and were dropped")

    write_csv(Path(args.out), written)
    summarize(written)


if __name__ == "__main__":
    main()
