# -*- coding: utf-8 -*-
"""
Build the R2 (image-retrieval) reference pool -- RSNA images, study-disjoint
from the existing RSNA eval set.
=============================================================================

Background: R2's chosen path is C (RSNA pool +
RSNA eval, decided by the retrieval sentinel check). C needs a pool of
labeled hemorrhage slices with zero PATIENT/STUDY overlap with the 150-image
RSNA eval set. RSNA's official labels carry no patient/study identifier, and
the third-party metadata datasets that claim to reconstruct one
(carlolepelaars/rsna-stage-2-metadata-ihd-2019 and its kio9999 fork) turned
out to have a real bug: their PatientID column was attached to label rows by
positional `df[col] = other_df[col]` assignment after a merge, not a keyed
join, so the same image's 6 subtype rows end up with 6 different (meaningless)
PatientID values -- verified by reading their own published notebook.

The fix used here needs no DICOM parsing and no third-party metadata at all:
`vaillant/rsna-ich-png` (original-resolution PNG mirror, CC0) stores files as
    pngs/pngs/<study_id>/<series_id>/IM_<n>-<image_id>.png
The STUDY_ID path segment *is* a real grouping key (one CT study = one
patient encounter) straight from the file layout. This mirror was previously
rejected for R1/eval fetching because there's no ID->path lookup short of
listing the whole ~750k-file dataset -- but this script doesn't need to look
up specific pre-chosen IDs. It scans forward through the listing (a free,
metadata-only API -- no per-file download, no rate limit) and POOL CANDIDATES
ARE WHATEVER IT FINDS, so the "no ID->path lookup" limitation doesn't apply.

Leakage guarantee: while scanning, any study that contains so much as one of
the 150 existing eval image IDs is dropped *in its entirety* (every slice in
that study, not just the matching one) before it's ever considered a pool
candidate. So the final pool can structurally never share a study with the
eval set -- this is checked at selection time, not argued probabilistically.

Usage
-----
    python build_rsna_pool.py --max-pages 400 --n 150
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from kaggle.api.kaggle_api_extended import KaggleApi

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))
from prep_rsna import read_labels, SUBTYPES  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rsna"
EVAL_MANIFEST = DATA_DIR / "manifest.csv"
LABELS_CSV = DATA_DIR / "_raw" / "stage_2_train.csv"
POOL_MANIFEST = DATA_DIR / "pool_manifest.csv"
POOL_IMAGES_DIR = DATA_DIR / "pool_images"

POOL_MIRROR_DATASET = "vaillant/rsna-ich-png"
PAGE_SIZE = 200
CANDIDATE_OVERSAMPLE = 2  # same rationale as prep_rsna.py

FIELDS = ["image_file", "study_id", "any_hemorrhage", "is_normal",
          "present_subtypes", *SUBTYPES.values()]


def load_excluded_image_ids(eval_manifest_path):
    """The 150 RSNA eval image IDs (e.g. 'ID_7e9af4496') -- any study
    containing one of these gets dropped wholesale from the pool."""
    with open(eval_manifest_path, newline="", encoding="utf-8") as f:
        return {Path(row["image_file"]).stem for row in csv.DictReader(f)}


def _study_id_and_image_id(path):
    # path looks like 'pngs/pngs/<study_id>/<series_id>/IM_0000-ID_xxxx.png'
    parts = path.split("/")
    study_id = parts[2]
    filename = parts[-1]
    image_id = filename.split("-", 1)[1].rsplit(".", 1)[0]
    return study_id, image_id


def _list_files_with_retry(api, token, page_size, max_retries=5, base_delay=10):
    """The listing API rate-limits after a few hundred sequential calls.
    Retry with exponential backoff; if it still won't recover, return None
    so the caller can stop scanning and use whatever was already gathered
    instead of crashing the whole pool build."""
    import requests

    for attempt in range(max_retries):
        try:
            return api.dataset_list_files(POOL_MIRROR_DATASET, page_token=token, page_size=page_size)
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[scan] 429 rate-limited, retry {attempt + 1}/{max_retries} "
                      f"after {delay}s...", flush=True)
                time.sleep(delay)
                continue
            return None
    return None


def scan_safe_studies(excluded_image_ids, max_pages, page_size=PAGE_SIZE):
    """Page through the mirror's file listing (free, no download). Groups
    contiguous rows by study_id -- a study is only finalized once the NEXT
    study's rows start, so a study split across a page boundary is never
    judged on a partial file list. Returns {image_id: path} for every image
    in a study that does NOT contain any excluded_image_ids."""
    api = KaggleApi()
    api.authenticate()

    safe_images = {}
    n_safe_studies = n_excluded_studies = 0
    current_study, current_files = None, []

    def finalize(study_id, files):
        nonlocal n_safe_studies, n_excluded_studies
        image_ids = [f[0] for f in files]
        if excluded_image_ids.intersection(image_ids):
            n_excluded_studies += 1
            return
        n_safe_studies += 1
        for image_id, path in files:
            safe_images[image_id] = path

    token = None
    t_start = time.perf_counter()
    for page_i in range(max_pages):
        res = _list_files_with_retry(api, token, page_size)
        if res is None:
            print(f"[scan] giving up after repeated rate-limit errors at page {page_i + 1} -- "
                  f"using the {len(safe_images)} images already scanned instead of failing the "
                  f"whole run")
            break
        for f in res.dataset_files:
            study_id, image_id = _study_id_and_image_id(f.name)
            if study_id != current_study:
                if current_study is not None:
                    finalize(current_study, current_files)
                current_study, current_files = study_id, []
            current_files.append((image_id, f.name))
        token = res.next_page_token
        if (page_i + 1) % 20 == 0 or not token:
            rate = (page_i + 1) / (time.perf_counter() - t_start)
            print(f"[scan] page {page_i + 1}/{max_pages}  "
                  f"safe_studies={n_safe_studies}  excluded_studies={n_excluded_studies}  "
                  f"images_so_far={len(safe_images)}  ({rate:.1f} pages/s)", flush=True)
        if not token:
            print("[scan] reached end of dataset listing")
            break
    if current_study is not None:
        finalize(current_study, current_files)  # last study, never followed by a new one
    return safe_images


def build_candidates(safe_images, labels_by_image_id):
    """Join scanned (image_id -> path) against stage_2_train.csv labels.
    Images the mirror has but the labels file doesn't (or vice versa) are
    dropped -- can't score what we can't label."""
    candidates = []
    for image_id, path in safe_images.items():
        rec = labels_by_image_id.get(image_id)
        if rec is None:
            continue
        candidates.append({**rec, "image_id": image_id, "path": path})
    return candidates


def stratified_sample(candidates, n, normal_frac, seed):
    """Same MultilabelStratifiedShuffleSplit approach as prep_rsna.py's
    sample_and_fetch -- but no fetch-miss handling needed here, since every
    candidate was already confirmed present in the mirror by construction of
    scan_safe_studies()."""
    rng = random.Random(seed)
    normals = [c for c in candidates if c["is_normal"]]
    hemo = [c for c in candidates if not c["is_normal"]]
    rng.shuffle(normals)

    n_normal_target = round(n * normal_frac)
    n_hemo_target = n - n_normal_target

    chosen_normals = normals[:n_normal_target]

    if not hemo:
        return chosen_normals
    label_matrix = np.array([[c[k] for k in SUBTYPES.values()] for c in hemo])
    X_dummy = np.zeros((len(hemo), 1))
    take_n = min(len(hemo), n_hemo_target)
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=take_n / len(hemo), random_state=seed)
    _, idx = next(splitter.split(X_dummy, label_matrix))
    chosen_hemo = [hemo[i] for i in idx]

    chosen = chosen_normals + chosen_hemo
    rng.shuffle(chosen)
    return chosen


def _download_file_with_retry(api, file_name, images_dir, max_retries=5, base_delay=10):
    import requests

    for attempt in range(max_retries):
        try:
            api.dataset_download_file(POOL_MIRROR_DATASET, file_name=file_name,
                                       path=str(images_dir), quiet=True)
            return True
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[fetch] 429 rate-limited, retry {attempt + 1}/{max_retries} "
                      f"after {delay}s...", flush=True)
                time.sleep(delay)
                continue
            return False
    return False


def download_chosen(chosen, images_dir):
    """Resumable: skips any image already on disk, so a rate-limit abort
    can be re-run with the same --out to pick up where it left off."""
    api = KaggleApi()
    api.authenticate()
    images_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, c in enumerate(chosen, 1):
        dest = images_dir / f"{c['image_id']}.png"
        if not dest.is_file():
            if not _download_file_with_retry(api, c["path"], images_dir):
                failed.append(c)
                continue
            downloaded = images_dir / Path(c["path"]).name
            downloaded.rename(dest)
        c["image_file"] = dest.name
        if i % 20 == 0 or i == len(chosen):
            print(f"[fetch] {i}/{len(chosen)}  (failed so far: {len(failed)})", flush=True)
    if failed:
        print(f"[warn] {len(failed)} downloads failed after retries -- dropped from the pool")
        for c in failed:
            chosen.remove(c)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["study_id"], _ = _study_id_and_image_id(r["path"])
            w.writerow(r)
    try:
        shown = path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = path
    print(f"[ok  ] wrote {len(rows)} rows -> {shown}")


def summarize(rows):
    n = len(rows)
    n_norm = sum(r["is_normal"] for r in rows)
    print(f"\n[pool] n={n}  normal={n_norm} ({n_norm/n:.0%})  hemorrhage={n - n_norm}")
    for k in SUBTYPES.values():
        print(f"    {k}: {sum(r[k] for r in rows)}")
    print(f"    distinct studies: {len({r['study_id'] for r in rows})}")


def main():
    ap = argparse.ArgumentParser(description="Build the R2 RSNA reference pool")
    ap.add_argument("--eval-manifest", default=str(EVAL_MANIFEST))
    ap.add_argument("--labels", default=str(LABELS_CSV))
    ap.add_argument("--max-pages", type=int, default=400,
                     help=f"each page = {PAGE_SIZE} files; 400 pages ~ 80k files scanned")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--normal-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(POOL_MANIFEST))
    ap.add_argument("--images", default=str(POOL_IMAGES_DIR))
    args = ap.parse_args()

    excluded = load_excluded_image_ids(Path(args.eval_manifest))
    print(f"[ok  ] excluded {len(excluded)} eval image IDs (any study containing "
          f"one of these is dropped wholesale)")

    labels = read_labels(Path(args.labels))
    labels_by_image_id = {r["image_id"]: r for r in labels}
    print(f"[ok  ] labeled images in stage_2_train.csv: {len(labels_by_image_id)}")

    safe_images = scan_safe_studies(excluded, args.max_pages)
    print(f"[ok  ] {len(safe_images)} images in eval-disjoint studies")

    candidates = build_candidates(safe_images, labels_by_image_id)
    print(f"[ok  ] {len(candidates)} candidates joined with labels")

    chosen = stratified_sample(candidates, args.n, args.normal_frac, args.seed)
    print(f"[ok  ] sampled {len(chosen)}/{args.n} requested")

    download_chosen(chosen, Path(args.images))
    write_csv(Path(args.out), chosen)
    summarize(chosen)


if __name__ == "__main__":
    main()
