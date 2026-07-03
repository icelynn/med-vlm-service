# -*- coding: utf-8 -*-
"""
Aggregate-reference histogram harmonization for the R2 pool.
========================================================================

The sentinel check's histogram-matching retry (retrieval_sentinel.py) matched
the RSNA pool to a SINGLE arbitrary reference
image's histogram -- a known simplification that worked for 4/5 reference
choices tested but failed on the 5th, right at the decision threshold. This
script replaces that with a steadier reference: the AVERAGE histogram across
ALL images in --reference-images, so one unusually bright/dark/lesion-heavy
slice can't single-handedly determine the target distribution.

The reference set is whichever dataset the pool is being matched TO --
CT-ICH eval images for R2-B (RSNA pool -> CT-ICH eval), or RSNA eval images
for the resolution+intensity-matched R2-C variant (RSNA pool -> RSNA eval,
on top of the separate resolution-matching step already done by downscaling
the pool). The function itself doesn't care which.

Algorithm is standard histogram specification (same idea as
skimage.exposure.match_histograms, generalized to match against a
precomputed aggregate histogram instead of one derived from a single
reference image): build each image's CDF, map each gray level to the gray
level in the reference CDF with the closest cumulative probability.

Usage
-----
    python harmonize_pool.py --reference-images data/ct_ich/images                              # R2-B
    python harmonize_pool.py --reference-images data/rsna/images --pool-images data/rsna/pool_images_128 --out-dir data/rsna/pool_images_128_harmonized  # R2-C variant
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REFERENCE_IMAGES = PROJECT_ROOT / "data" / "ct_ich" / "images"
DEFAULT_POOL_IMAGES = PROJECT_ROOT / "data" / "rsna" / "pool_images"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "rsna" / "pool_images_harmonized"

N_BINS = 256


def compute_aggregate_histogram(image_paths, bins=N_BINS):
    """Average grayscale histogram across all reference images -- the
    "typical slice" contrast profile for whichever dataset image_paths come
    from, not swayed by any single unusually bright/dark/lesion-heavy image."""
    hist_sum = np.zeros(bins, dtype=np.float64)
    for p in image_paths:
        img = np.array(Image.open(p).convert("L"))
        hist, _ = np.histogram(img.ravel(), bins=bins, range=(0, bins))
        hist_sum += hist
    return hist_sum / len(image_paths)


def match_to_histogram(image, ref_hist):
    """Remap `image` (2D uint8 array) so its histogram follows ref_hist,
    via standard CDF-to-CDF gray-level mapping."""
    src_hist, _ = np.histogram(image.ravel(), bins=N_BINS, range=(0, N_BINS))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]

    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    gray_levels = np.arange(N_BINS)
    mapping = np.interp(src_cdf, ref_cdf, gray_levels)

    matched = mapping[image.ravel()].reshape(image.shape)
    return np.clip(matched, 0, 255).astype(np.uint8)


def harmonize_pool(pool_images_dir, out_dir, ref_hist):
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_paths = sorted(Path(pool_images_dir).glob("*.png"))
    for p in pool_paths:
        img = np.array(Image.open(p).convert("L"))
        matched = match_to_histogram(img, ref_hist)
        Image.fromarray(matched).save(out_dir / p.name)
    return pool_paths


def main():
    ap = argparse.ArgumentParser(description="Aggregate-reference histogram harmonization")
    ap.add_argument("--reference-images", default=str(DEFAULT_REFERENCE_IMAGES),
                    help="directory of images whose AVERAGE histogram becomes the target "
                         "distribution -- pass the eval dataset's images (CT-ICH for R2-B, "
                         "RSNA for the resolution+intensity-matched R2-C variant)")
    ap.add_argument("--pool-images", default=str(DEFAULT_POOL_IMAGES))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    reference_paths = sorted(Path(args.reference_images).glob("*.png"))
    print(f"[ok  ] aggregating histogram over {len(reference_paths)} reference images "
          f"({args.reference_images})")
    ref_hist = compute_aggregate_histogram(reference_paths)

    pool_paths = harmonize_pool(args.pool_images, Path(args.out_dir), ref_hist)
    print(f"[ok  ] harmonized {len(pool_paths)} pool images -> {args.out_dir}")


if __name__ == "__main__":
    main()
