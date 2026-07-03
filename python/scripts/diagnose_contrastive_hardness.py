# -*- coding: utf-8 -*-
"""Zero-cost diagnostic of how "hard" the hard negative actually is
(positive-sim vs negative-sim gap), no LLM inference needed."""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python" / "rag"))

from biomedclip_embed import embed_image  # noqa: E402
from providers import _get_image_index, _select_contrastive_from_sims  # noqa: E402


def diagnose(manifest_path, images_dir, index_dir, label):
    embeddings, labels = _get_image_index(index_dir)
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    pos_sims, neg_sims, gaps = [], [], []
    for r in rows:
        img_path = Path(images_dir) / r["image_file"]
        q_emb = embed_image(str(img_path))
        sims = embeddings @ q_emb
        idxs = _select_contrastive_from_sims(sims, labels, k=3)
        pos_idx, neg_idx = idxs[:2], idxs[2]
        pos_sims.extend(sims[i] for i in pos_idx)
        neg_sims.append(sims[neg_idx])
        gaps.append(sims[pos_idx[0]] - sims[neg_idx])
    print(f"[{label}] n={len(rows)}  positive sim: mean={np.mean(pos_sims):.3f} "
          f"negative sim: mean={np.mean(neg_sims):.3f}  gap: mean={np.mean(gaps):.3f} "
          f"std={np.std(gaps):.3f} min={np.min(gaps):.3f} max={np.max(gaps):.3f}")


if __name__ == "__main__":
    diagnose("data/ct_ich/manifest.csv", "data/ct_ich/images",
              "data/rag/image_index_b_harmonized", "CT-ICH")
    diagnose("data/rsna/manifest.csv", "data/rsna/images",
              "data/rag/image_index_c_128matched", "RSNA")
