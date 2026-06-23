# -*- coding: utf-8 -*-
"""
Build the R2 BiomedCLIP image-retrieval index over the RSNA pool.
====================================================================

Reads data/rsna/pool_manifest.csv + data/rsna/pool_images/ (built by
build_rsna_pool.py), embeds every pool slice with BiomedCLIP, and saves the
embeddings + labels under data/rag/image_index/ for providers.py's
ImageExemplarProvider to load at eval time.

Also runs a quick sanity check (Week4 plan Day1 step 3): a few random
hemorrhage-positive pool images query the REST of the pool, printing their
top-3 nearest neighbours' labels -- a human can eyeball whether "visually
similar" tracks "same subtype" before trusting this for real retrieval.

Usage
-----
    python build_image_index.py
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

RAG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RAG_DIR))
from biomedclip_embed import embed_images  # noqa: E402

PROJECT_ROOT = RAG_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rsna"
POOL_MANIFEST = DATA_DIR / "pool_manifest.csv"
POOL_IMAGES_DIR = DATA_DIR / "pool_images"
INDEX_DIR = PROJECT_ROOT / "data" / "rag" / "image_index"

SUBTYPE_KEYS = ["IPH", "IVH", "SAH", "EDH", "SDH"]


def load_pool(manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sanity_check(rows, embeddings, n_queries=3, k=3, seed=0):
    rng = random.Random(seed)
    positives = [i for i, r in enumerate(rows) if r["any_hemorrhage"] == "1"]
    queries = rng.sample(positives, min(n_queries, len(positives)))

    print("\n[sanity check] nearest neighbours for a few hemorrhage-positive queries:")
    for qi in queries:
        sims = embeddings @ embeddings[qi]
        sims[qi] = -1  # exclude self
        top = np.argsort(-sims)[:k]
        q_subtypes = rows[qi]["present_subtypes"] or "none"
        print(f"\n  query {rows[qi]['image_file']}  subtypes=[{q_subtypes}]")
        for ni in top:
            n_subtypes = rows[ni]["present_subtypes"] or "none"
            print(f"    -> {rows[ni]['image_file']}  sim={sims[ni]:.3f}  "
                  f"subtypes=[{n_subtypes}]")


def main():
    ap = argparse.ArgumentParser(description="Build the R2 BiomedCLIP image index")
    ap.add_argument("--manifest", default=str(POOL_MANIFEST))
    ap.add_argument("--images", default=str(POOL_IMAGES_DIR))
    ap.add_argument("--out-dir", default=str(INDEX_DIR))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = load_pool(Path(args.manifest))
    print(f"[ok  ] {len(rows)} pool slices in {args.manifest}")

    images_dir = Path(args.images)
    paths = [images_dir / r["image_file"] for r in rows]
    print("[embed] pool slices with BiomedCLIP...")
    embeddings = embed_images(paths, device=args.device)
    print(f"[ok  ] embeddings shape={embeddings.shape}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    labels = [{"image_file": r["image_file"],
               "any_hemorrhage": int(r["any_hemorrhage"]),
               "present_subtypes": r["present_subtypes"],
               **{k: int(r[k]) for k in SUBTYPE_KEYS}}
              for r in rows]
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"[ok  ] wrote {out_dir / 'embeddings.npy'} and {out_dir / 'labels.json'}")

    sanity_check(rows, embeddings)


if __name__ == "__main__":
    main()
