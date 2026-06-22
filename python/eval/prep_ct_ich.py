# -*- coding: utf-8 -*-
"""
CT-ICH (PhysioNet v1.3.1) data preparation
===========================================

Turns the "Computed Tomography Images for Intracranial Hemorrhage Detection and
Segmentation" dataset (Hssayeni et al., 2020) into a clean, reproducible
evaluation manifest for the No-RAG baseline (head-CT hemorrhage detection).

Run this on the machine where you develop (local). It only needs the dataset
you downloaded after signing the PhysioNet DUA; it does NOT download anything.

What it does
------------
1. Read hemorrhage_diagnosis_raw_ct.csv -> per-slice labels for the 5 ICH
   subtypes (+ fracture, + No_Hemorrhage). Build a full index (one row / slice).
2. Stratified-sample N slices (default 150): a fixed "normal" (no-hemorrhage)
   fraction, the rest hemorrhage slices drawn via multi-label stratified
   sampling (MultilabelStratifiedShuffleSplit) across the 5 subtypes, so the
   sample's per-subtype prevalence and co-occurrence both track the true
   population instead of an artificial balance.
3. Render ONLY the sampled slices: load each patient's NIfTI volume, take the
   slice, apply the brain window, save a PNG. (We never materialise all ~2500
   slices -- only the ones we sampled.)
4. Write:
     data/ct_ich/full_index.csv   -- every labeled slice (labels only)
     data/ct_ich/manifest.csv     -- the sampled N slices (the eval set)
     data/ct_ich/images/<pid>_<slice>.png  -- the sampled images

Verified dataset facts (from the dataset authors' prepare_data.py and the
PhysioNet v1.3.1 layout):
  * NIfTI volumes:  ct_scans/{PatientNumber:03d}.nii , array shape (H, W, slices),
    raw Hounsfield Units; slice axis = last.
  * CSV SliceNumber is 1-indexed -> NIfTI slice index = SliceNumber - 1.
  * CSV columns: PatientNumber, SliceNumber, Intraventricular, Intraparenchymal,
    Subarachnoid, Epidural, Subdural, No_Hemorrhage, Fracture_Yes_No.
  * Brain window WL=40, WW=120 (the authors' parameters; textbook brain window
    is WW=80 -- pass --ww 80 if you prefer that. Documented either way).
  * Authors do NOT rotate slices (their np.rot90 is commented out); we follow.

Usage
-----
    python prep_ct_ich.py --root <path-to-extracted-ct-ich-1.3.1>
    python prep_ct_ich.py --root ... --n 300 --seed 7
    python prep_ct_ich.py --root ... --ww 80          # textbook brain window
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ct_ich"
IMAGES_DIR = DATA_DIR / "images"
RAW_DIR = DATA_DIR / "_raw"

CSV_NAME = "hemorrhage_diagnosis_raw_ct.csv"
# CSV column -> short subtype key. Order = display / coverage order.
SUBTYPES = {
    "Intraparenchymal": "IPH",
    "Intraventricular": "IVH",
    "Subarachnoid":     "SAH",
    "Epidural":         "EDH",
    "Subdural":         "SDH",
}


def find_root(root):
    """Locate the dir that contains the CSV + ct_scans/ (search root + subdirs)."""
    root = Path(root)
    for c in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        if (c / CSV_NAME).is_file() and (c / "ct_scans").is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find {CSV_NAME} + ct_scans/ under {root}. "
        f"Point --root at the extracted CT-ICH v1.3.1 folder."
    )


def _i(x):
    """Parse a CSV cell that may be '0', '1', '0.0', '1.0' -> int."""
    return int(float(x))


def read_index(root):
    """Parse the per-slice CSV into a list of record dicts."""
    records = []
    with open(root / CSV_NAME, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            subs = {key: _i(row[col]) for col, key in SUBTYPES.items()}
            no_hem = _i(row.get("No_Hemorrhage", 0))
            present = [k for k, v in subs.items() if v == 1]
            rec = {
                "patient": int(float(row["PatientNumber"])),
                "slice": int(float(row["SliceNumber"])),
                **subs,
                "Fracture": _i(row.get("Fracture_Yes_No", 0)),
                "No_Hemorrhage": no_hem,
                # A slice is positive if any subtype fired (more robust than
                # trusting No_Hemorrhage alone, though they should agree).
                "any_hemorrhage": int(bool(present)),
                "is_normal": no_hem == 1 and not present,
                "present_subtypes": "; ".join(present),
            }
            records.append(rec)
    return records


def stratified_sample(records, n, normal_frac, seed):
    """Sample n slices: a fixed normal fraction (deliberately enriched vs the
    true ~89% normal rate, so Any-hem F1 has enough hemorrhage slices to be
    measurable -- this enrichment is intentional, unlike everything below it)
    + hemorrhage slices drawn via multi-label stratified sampling across the
    5 subtypes. Deterministic given the seed.

    Why multi-label stratification, not hand-rolled greedy coverage: an
    earlier version picked hemorrhage slices via greedy set-cover ("which
    candidate covers the most still-under-represented subtypes"). That
    structurally favours multi-subtype slices -- on RSNA's much larger pool
    it concentrated >90% of the sample on simultaneous 4-5-subtype slices
    that make up <1% of the true population (see core-challenges doc
    difficulty (10)). A hand-patched "prefer fewer subtypes" tiebreak
    overcorrected to an artificially perfect 0%-multi, equal-support sample
    -- just as fake a fingerprint as the original skew, since real subtype
    prevalence is never balanced (e.g. epidural vs subdural differ ~15x).
    MultilabelStratifiedShuffleSplit (Sechidis et al. 2011, the standard tool
    for this exact problem -- used in published RSNA-competition solutions)
    preserves per-subtype prevalence AND co-occurrence proportions in one
    step, so there's no balance-vs-representativeness tradeoff to hand-tune.
    """
    rng = random.Random(seed)
    normals = [r for r in records if r["is_normal"]]
    hemo = [r for r in records if not r["is_normal"]]

    n_normal = min(len(normals), round(n * normal_frac))
    n_hemo = min(len(hemo), n - n_normal)
    deficit = n - (n_normal + n_hemo)
    if deficit > 0:
        n_normal = min(len(normals), n_normal + deficit)

    sampled = rng.sample(normals, n_normal)

    label_matrix = np.array([[r[k] for k in SUBTYPES.values()] for r in hemo])
    X_dummy = np.zeros((len(hemo), 1))
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=n_hemo / len(hemo), random_state=seed)
    _, sample_idx = next(splitter.split(X_dummy, label_matrix))
    chosen_idx = list(sample_idx[:n_hemo])
    if len(chosen_idx) < n_hemo:
        # Internal rounding in the per-label allocation can land 1-2 short
        # when some labels have very few candidates; top up from the
        # remaining pool (not picked) rather than silently returning n-1.
        leftover = [i for i in range(len(hemo)) if i not in set(chosen_idx)]
        rng.shuffle(leftover)
        chosen_idx += leftover[:n_hemo - len(chosen_idx)]
    sampled += [hemo[i] for i in chosen_idx]

    rng.shuffle(sampled)
    return sampled


def window_slice(slice_hu, wl, ww):
    """Apply a CT window to a 2D HU slice -> uint8 [0,255].

    Linear map from the authors' prepare_data.py:
        out = (hu - (wl - ww/2)) * 255 / ww, clipped to [0,255].
    """
    w_min = wl - ww / 2.0
    w_max = wl + ww / 2.0
    out = (slice_hu.astype(np.float64) - w_min) * (255.0 / (w_max - w_min))
    return np.clip(out, 0, 255).astype(np.uint8)


def render(sampled, root, wl, ww, size):
    """Render the sampled slices to PNG. Loads each NIfTI volume once.

    Returns the subset of records whose image was written (drops any slice whose
    NIfTI / slice index is missing, reporting how many).
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    by_patient = defaultdict(list)
    for r in sampled:
        by_patient[r["patient"]].append(r)

    written, missing = [], []
    for pid, recs in sorted(by_patient.items()):
        nii = root / "ct_scans" / f"{pid:03d}.nii"
        if not nii.is_file():
            missing.extend(recs)
            continue
        vol = nib.load(str(nii)).get_fdata()  # (H, W, slices), raw HU
        n_sl = vol.shape[2]
        for r in recs:
            idx = r["slice"] - 1  # CSV is 1-indexed
            if not (0 <= idx < n_sl):
                missing.append(r)
                continue
            img = window_slice(vol[:, :, idx], wl, ww)
            im = Image.fromarray(img)
            if size and im.size != (size, size):
                im = im.resize((size, size), Image.LANCZOS)
            fname = f"{pid:03d}_{r['slice']:03d}.png"
            im.save(IMAGES_DIR / fname, "PNG")
            r["image_file"] = fname
            written.append(r)
    return written, missing


FIELDS = ["image_file", "patient", "slice", "any_hemorrhage", "is_normal",
          "present_subtypes", *SUBTYPES.values(), "Fracture", "No_Hemorrhage"]


def write_csv(path, rows, with_image=True):
    fields = FIELDS if with_image else [f for f in FIELDS if f != "image_file"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ok  ] wrote {len(rows)} rows -> {path.relative_to(PROJECT_ROOT)}")


def summarize(rows, title):
    n = len(rows)
    if n == 0:
        print(f"[{title}] empty"); return
    n_norm = sum(r["is_normal"] for r in rows)
    counts = {k: sum(r[k] for r in rows) for k in SUBTYPES.values()}
    print(f"\n[{title}] n={n}  normal={n_norm} ({n_norm/n:.0%})  "
          f"hemorrhage={n-n_norm}")
    for k, c in counts.items():
        print(f"    {k}: {c}")


def main():
    ap = argparse.ArgumentParser(description="Prepare CT-ICH eval manifest")
    ap.add_argument("--root", default=str(RAW_DIR),
                    help="path to the extracted CT-ICH v1.3.1 folder")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal-frac", type=float, default=0.3)
    ap.add_argument("--wl", type=float, default=40.0, help="window level (HU)")
    ap.add_argument("--ww", type=float, default=120.0,
                    help="window width (HU); authors=120, textbook brain=80")
    ap.add_argument("--size", type=int, default=512, help="output PNG size (px)")
    args = ap.parse_args()

    root = find_root(args.root)
    print(f"[info] dataset root: {root}")

    records = read_index(root)
    print(f"[ok  ] labeled slices: {len(records)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(DATA_DIR / "full_index.csv", records, with_image=False)
    summarize(records, "full_index")

    sample = stratified_sample(records, args.n, args.normal_frac, args.seed)
    print(f"\n[info] rendering {len(sample)} sampled slices "
          f"(window WL={args.wl} WW={args.ww}) ...")
    written, missing = render(sample, root, args.wl, args.ww, args.size)

    write_csv(DATA_DIR / "manifest.csv", written, with_image=True)
    summarize(written, "manifest")
    if missing:
        print(f"\n[warn] {len(missing)} sampled slices had no NIfTI/slice and were dropped.")
    print(f"\n[done] images in {IMAGES_DIR.relative_to(PROJECT_ROOT)}  "
          f"(seed={args.seed}, normal_frac={args.normal_frac}).")


if __name__ == "__main__":
    main()
