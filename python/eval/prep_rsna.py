# -*- coding: utf-8 -*-
"""
RSNA Intracranial Hemorrhage Detection (Kaggle, 2019) data preparation
========================================================================

Turns the official `stage_2_train.csv` labels + a Kaggle-mirrored PNG subset
into an evaluation manifest with the SAME column schema as prep_ct_ich.py's
manifest.csv, so run_baseline.py / score_baseline.py work unmodified on
either dataset.

Why this differs from prep_ct_ich.py
-------------------------------------
RSNA's public labels release one anonymized image ID per slice -- no
patient/study grouping, no volumes (unlike CT-ICH's per-patient NIfTI). So:
  * "patient" is a synthetic per-row index, NOT a real patient ID -- do not
    use it for patient-level analysis.
  * "slice" is always 1 (each RSNA image is its own unit here).
  * "Fracture" is left blank: RSNA does not label fractures.
  * Images come from the Kaggle mirror `guiferviz/rsna_stage1_png_128`
    (128x128 PNGs, already brain-windowed) -- NOT the 512x512 we used for
    CT-ICH. This is a known cross-dataset resolution confound; record it in
    the honest-limitations write-up. (Tried `vaillant/rsna-ich-png`, original
    size, but its files are nested under unpredictable per-study/series
    folders with no ID->path lookup short of listing all ~750k files, so
    individual images can't be fetched by ID. `guiferviz` uses flat
    `stage_1_train_images/<id>.png` naming, fetchable by ID directly.)
  * That mirror only covers "stage 1" images (~90% of stage_2's full label
    set), so the stratified candidate list is drawn oversized and tried in
    order via the Kaggle API, skipping ones the mirror doesn't have (see
    CANDIDATE_OVERSAMPLE).

Preconditions
-------------
  * `~/.kaggle/access_token` or `~/.kaggle/kaggle.json` set up (Kaggle auth).
  * `stage_2_train.csv` downloaded manually from the competition's Data tab
    (the competition's file-list/download API returns empty for this 2019
    competition regardless of auth method) and placed at
    data/rsna/_raw/stage_2_train.csv.
  * `pip install kagglehub` in the venv.
  * Run this on the machine with the above (EC2), not local dev.

Usage
-----
    python prep_rsna.py
    python prep_rsna.py --n 150 --seed 42
"""

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import kagglehub
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from kagglehub.exceptions import KaggleApiHTTPError
from PIL import Image

# How many extra hemorrhage candidates to draw beyond the target, since some
# stratified-sample picks won't exist in the stage_1 mirror subset (fetch
# misses). Candidates are tried in the stratified split's own (shuffled)
# order until the target is met, so a roughly-random subset of misses
# shouldn't meaningfully disturb the label-distribution fidelity the
# stratification provides. 2x comfortably covers the ~12% miss rate observed
# in practice; if it's not enough, sample_and_fetch just returns fewer than
# requested (graceful, not silent) rather than grinding indefinitely.
CANDIDATE_OVERSAMPLE = 2

# Throttle between single-file Kaggle fetches. The API rate-limits a whole
# account/token across endpoints; kagglehub's own error reporting collapses
# a 429 (too many requests) into the same exception/status as a genuine 404
# (file not found) -- see core-challenges write-up -- so we cannot tell "miss"
# from "rate-limited" from the exception alone. A long run of consecutive
# misses is the only observable signal something is wrong (either the mirror
# truly doesn't have that ID, or we are rate-limited and every ID looks like
# a miss); MAX_CONSECUTIVE_MISSES turns that into a hard stop instead of a
# silent multi-hour hang.
FETCH_DELAY_SECONDS = 1.5
MAX_CONSECUTIVE_MISSES = 40

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rsna"
IMAGES_DIR = DATA_DIR / "images"
RAW_DIR = DATA_DIR / "_raw"

LABELS_CSV = RAW_DIR / "stage_2_train.csv"
MIRROR_DATASET = "guiferviz/rsna_stage1_png_128"
MIRROR_PREFIX = "stage_1_train_images"

# Official RSNA subtype name -> our short key (same keys as CT-ICH).
SUBTYPES = {
    "intraparenchymal": "IPH",
    "intraventricular": "IVH",
    "subarachnoid":     "SAH",
    "epidural":         "EDH",
    "subdural":         "SDH",
}

FIELDS = ["image_file", "patient", "slice", "any_hemorrhage", "is_normal",
          "present_subtypes", *SUBTYPES.values(), "Fracture", "No_Hemorrhage"]


def read_labels(path):
    """Parse the long-format stage_2_train.csv (ID_<img>_<subtype>,Label)
    into one record per image."""
    by_image = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: ID,Label
        for row_id, label in reader:
            img_id, suffix = row_id.rsplit("_", 1)
            by_image[img_id][suffix] = int(label)

    records = []
    for img_id, labels in by_image.items():
        subs = {key: labels.get(official, 0) for official, key in SUBTYPES.items()}
        present = [k for k, v in subs.items() if v == 1]
        records.append({
            "image_id": img_id,
            **subs,
            "any_hemorrhage": int(bool(present)),
            "is_normal": not bool(present),
            "present_subtypes": "; ".join(present),
        })
    return records


def fetch_image(img_id):
    """Try to pull this image's PNG from the Kaggle mirror (single-file
    fetch, not the whole dataset). Returns a local cache path, or None if the
    mirror doesn't have this ID (it only covers a "stage 1" subset) -- or if
    we're rate-limited, which looks identical from here (see module-level
    comment on MAX_CONSECUTIVE_MISSES)."""
    time.sleep(FETCH_DELAY_SECONDS)
    try:
        return kagglehub.dataset_download(
            MIRROR_DATASET, path=f"{MIRROR_PREFIX}/{img_id}.png")
    except KaggleApiHTTPError:
        return None


class TooManyConsecutiveMisses(RuntimeError):
    pass


def sample_and_fetch(records, n, normal_frac, seed):
    """Stratified sample + fetch in one pass: a fixed normal fraction, plus
    hemorrhage images drawn via multi-label stratified sampling
    (MultilabelStratifiedShuffleSplit, Sechidis et al. 2011 -- the standard
    tool for this exact problem, used in published RSNA-competition
    solutions). See prep_ct_ich.py's stratified_sample docstring for why a
    hand-rolled greedy
    coverage loop was replaced: it structurally favours multi-subtype
    images (whichever covers the most still-rare labels at once), which blew
    up to >90% multi-subtype on this dataset's much larger pool, and a
    hand-patched "prefer fewer subtypes" tiebreak overcorrected to an
    artificially perfect 0%-multi sample -- just as fake a fingerprint as
    the original skew.

    The stratified candidate set is drawn oversized (CANDIDATE_OVERSAMPLE x
    the target, see module-level comment) since some candidates won't exist
    in the stage_1 mirror subset; candidates are tried in the split's own
    shuffled order until the target is met or the candidate list runs out.

    Prints live progress (one line per fetch) since this can run for a
    while -- stdout is block-buffered when redirected to a log file, so
    flush=True is needed or nothing appears until the process exits.

    Raises TooManyConsecutiveMisses if MAX_CONSECUTIVE_MISSES candidates in a
    row all fail: that streak means either this category is essentially
    absent from the mirror, or we're rate-limited and every request is
    failing regardless of whether the file exists. Either way, grinding
    through the rest of the pool one-by-one would just silently burn hours;
    stop and let a human look instead.
    """
    rng = random.Random(seed)
    normals = [r for r in records if r["is_normal"]]
    hemo = [r for r in records if not r["is_normal"]]
    rng.shuffle(normals)

    n_normal_target = round(n * normal_frac)
    n_hemo_target = n - n_normal_target

    label_matrix = np.array([[r[k] for k in SUBTYPES.values()] for r in hemo])
    X_dummy = np.zeros((len(hemo), 1))
    oversample_n = min(len(hemo), n_hemo_target * CANDIDATE_OVERSAMPLE)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=oversample_n / len(hemo), random_state=seed)
    _, cand_idx = next(splitter.split(X_dummy, label_matrix))
    hemo_candidates = [hemo[i] for i in cand_idx]
    rng.shuffle(hemo_candidates)

    chosen, missing = [], []
    consecutive_misses = 0

    def attempt(r):
        nonlocal consecutive_misses
        path = fetch_image(r["image_id"])
        if path is None:
            missing.append(r)
            consecutive_misses += 1
            print(f"  [{len(chosen)}/{n}] miss  {r['image_id']}  "
                  f"(consecutive misses: {consecutive_misses})", flush=True)
            if consecutive_misses >= MAX_CONSECUTIVE_MISSES:
                raise TooManyConsecutiveMisses(
                    f"{consecutive_misses} consecutive misses -- stopping "
                    f"instead of grinding through the rest of the pool. "
                    f"Got {len(chosen)}/{n} so far.")
            return False
        consecutive_misses = 0
        r["_local_path"] = path
        chosen.append(r)
        print(f"  [{len(chosen)}/{n}] hit   {r['image_id']}", flush=True)
        return True

    try:
        for r in normals:
            if len(chosen) >= n_normal_target:
                break
            attempt(r)

        n_hemo = 0
        for r in hemo_candidates:
            if n_hemo >= n_hemo_target:
                break
            if attempt(r):
                n_hemo += 1
        if n_hemo < n_hemo_target:
            print(f"\n[warn] candidate list exhausted -- got {n_hemo}/{n_hemo_target} "
                  f"hemorrhage images. Raise CANDIDATE_OVERSAMPLE and re-run if you "
                  f"need the full target.")
    except TooManyConsecutiveMisses as e:
        print(f"\n[warn] {e} Returning the {len(chosen)} fetched so far instead "
              f"of hanging -- investigate before re-running (see module "
              f"docstring on MAX_CONSECUTIVE_MISSES).")

    rng.shuffle(chosen)
    return chosen, missing


def save_images(chosen):
    """Copy fetched PNGs into data/rsna/images/ and fill in the
    schema fields that have no real RSNA equivalent (see module docstring)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(chosen):
        fname = f"{r['image_id']}.png"
        dest = IMAGES_DIR / fname
        if not dest.is_file():
            Image.open(r["_local_path"]).convert("L").save(dest, "PNG")
        r["image_file"] = fname
        r["patient"] = i  # synthetic index, NOT a real patient ID
        r["slice"] = 1
        r["Fracture"] = ""  # not labeled in RSNA
        r["No_Hemorrhage"] = int(not r["any_hemorrhage"])


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[ok  ] wrote {len(rows)} rows -> {path.relative_to(PROJECT_ROOT)}")


def summarize(rows):
    n = len(rows)
    if n == 0:
        print("[manifest] empty"); return
    n_norm = sum(r["is_normal"] for r in rows)
    counts = {k: sum(r[k] for r in rows) for k in SUBTYPES.values()}
    print(f"\n[manifest] n={n}  normal={n_norm} ({n_norm/n:.0%})  "
          f"hemorrhage={n-n_norm}")
    for k, c in counts.items():
        print(f"    {k}: {c}")


def main():
    ap = argparse.ArgumentParser(description="Prepare RSNA eval manifest")
    ap.add_argument("--labels", default=str(LABELS_CSV))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal-frac", type=float, default=0.3)
    args = ap.parse_args()

    records = read_labels(Path(args.labels))
    print(f"[ok  ] labeled images in {Path(args.labels).name}: {len(records)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    chosen, missing = sample_and_fetch(records, args.n, args.normal_frac, args.seed)
    print(f"\n[info] fetched {len(chosen)}/{args.n} requested "
          f"({len(missing)} candidates not in the stage_1 mirror, skipped)")

    save_images(chosen)
    write_csv(DATA_DIR / "manifest.csv", chosen)
    summarize(chosen)
    print(f"\n[done] images in {IMAGES_DIR.relative_to(PROJECT_ROOT)}  (seed={args.seed}, "
          f"normal_frac={args.normal_frac}).")


if __name__ == "__main__":
    main()
