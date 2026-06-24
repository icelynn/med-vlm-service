# -*- coding: utf-8 -*-
"""Full-manifest validation of the R2 modality gate (_passes_head_ct_gate),
mirroring the rigor of diagnose_refusal.py for core-challenges #9: a 2-image
smoke test isn't enough to claim a refusal-behavior fix works (that mistake
is exactly what core-challenges #9 warns against) -- this runs the gate alone
(not the full few-shot path) against the entire CT-ICH manifest plus the full
defensive set, and reports an actual false-refusal / false-pass rate.

Usage (on EC2): HF_HOME=/mnt/hf ENV=demo MODEL_ROLE=main python3 python/scripts/validate_r2_gate.py
"""
import asyncio
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

from PIL import Image  # noqa: E402

import inference  # noqa: E402

CT_ICH_IMAGES = PROJECT_ROOT / "data" / "ct_ich" / "images"
CT_ICH_MANIFEST = PROJECT_ROOT / "data" / "ct_ich" / "manifest.csv"
DEFENSIVE_DIR = PROJECT_ROOT / "data" / "test_images" / "defensive"


async def _check(processor, model, path: Path) -> bool:
    image = Image.open(path).convert("RGB")
    image.thumbnail((inference._HF_MAX_IMAGE_SIZE, inference._HF_MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
    return await inference._passes_head_ct_gate(processor, model, image)


async def main():
    with open(CT_ICH_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ct_ich_files = [CT_ICH_IMAGES / r["image_file"] for r in rows]

    defensive_files = []
    for sub in ("non_medical", "non_brain_medical"):
        d = DEFENSIVE_DIR / sub
        defensive_files += sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))

    print(f"CT-ICH manifest: {len(ct_ich_files)} images; defensive set: {len(defensive_files)} images")

    async with inference._hf_lock:
        processor, model = await inference._load_hf_model()

        false_refusals = []
        for i, path in enumerate(ct_ich_files):
            passed = await _check(processor, model, path)
            if not passed:
                false_refusals.append(path.name)
            if (i + 1) % 25 == 0:
                print(f"  CT-ICH {i + 1}/{len(ct_ich_files)}...")

        false_passes = []
        for i, path in enumerate(defensive_files):
            passed = await _check(processor, model, path)
            if passed:
                false_passes.append(str(path.relative_to(DEFENSIVE_DIR)))
            if (i + 1) % 10 == 0:
                print(f"  defensive {i + 1}/{len(defensive_files)}...")

    n_ct_ich = len(ct_ich_files)
    n_defensive = len(defensive_files)
    print(f"\nCT-ICH false-refusal rate: {len(false_refusals)}/{n_ct_ich} "
          f"({100 * len(false_refusals) / n_ct_ich:.1f}%)")
    if false_refusals:
        print(f"  false refusals: {false_refusals}")
    print(f"Defensive false-pass rate: {len(false_passes)}/{n_defensive} "
          f"({100 * len(false_passes) / n_defensive:.1f}%)")
    if false_passes:
        print(f"  false passes: {false_passes}")

    ok = (len(false_refusals) == 0) and (len(false_passes) == 0)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
