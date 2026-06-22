# -*- coding: utf-8 -*-
"""
One-off controlled resolution experiment: fetch the SAME 150 RSNA image IDs
already used in data/rsna/manifest.csv, but from the 512x512 mirror
(`backaggle/rsna_512`) instead of the 128x128 one (`guiferviz/rsna_stage1_png_128`).

Everything else stays fixed (same images, same GT labels, same model/prompt),
so any F1 difference vs the existing 128px result is attributable to
resolution (plus whatever windowing difference the two community mirrors
happen to use -- not a perfectly pure ablation, but far cleaner than
comparing across two different datasets). See core-challenges doc difficulty
(11) for why the cross-dataset comparison alone couldn't isolate resolution.

This is a one-off diagnostic script, not part of the regular prep pipeline --
intentionally not folded into prep_rsna.py's general machinery.

Usage:
    python fetch_rsna_512_control.py
"""
import csv
import time
from pathlib import Path

import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError
from PIL import Image

FETCH_DELAY_SECONDS = 1.5
MIRROR_DATASET = "backaggle/rsna_512"
MIRROR_PREFIX = "stage_1_train_images_jpg"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rsna"
MANIFEST = DATA_DIR / "manifest.csv"
OUT_DIR = DATA_DIR / "images_512px_control"


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hits, misses = [], []
    for i, r in enumerate(rows, 1):
        img_id = r["image_file"].removesuffix(".png")
        time.sleep(FETCH_DELAY_SECONDS)
        try:
            path = kagglehub.dataset_download(
                MIRROR_DATASET, path=f"{MIRROR_PREFIX}/{img_id}.jpg")
        except KaggleApiHTTPError:
            misses.append(img_id)
            print(f"  [{i}/{len(rows)}] miss  {img_id}", flush=True)
            continue
        Image.open(path).convert("L").save(OUT_DIR / r["image_file"], "PNG")
        hits.append(img_id)
        print(f"  [{i}/{len(rows)}] hit   {img_id}", flush=True)

    print(f"\n[done] {len(hits)}/{len(rows)} fetched at 512px -> "
          f"{OUT_DIR.relative_to(PROJECT_ROOT)}")
    if misses:
        print(f"[warn] {len(misses)} IDs not found in the 512px mirror: {misses}")


if __name__ == "__main__":
    main()
