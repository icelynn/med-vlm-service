# -*- coding: utf-8 -*-
"""Ad-hoc EC2 smoke test for the R2 modality gate (core-challenges R-2 fix,
2026-06-24): does image_retrieval=true now correctly refuse non-head-CT input
while still answering normally on real CT-ICH slices, both positive and
negative? Mirrors the validation style of diagnose_refusal.py (core-challenges
#9) but drives inference.generate_medical_report directly with image_retrieval=True.

Not part of the committed test suite -- this is a one-off GPU validation run,
kept under python/scripts/ for the same reason feasibility_check.py lives there.

Usage (on EC2): HF_HOME=/mnt/hf ENV=demo MODEL_ROLE=main python3 python/scripts/smoke_test_r2_gate.py
"""
import asyncio
import base64
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python" / "src"))

import inference  # noqa: E402

CT_ICH_IMAGES = PROJECT_ROOT / "data" / "ct_ich" / "images"
CT_ICH_MANIFEST = PROJECT_ROOT / "data" / "ct_ich" / "manifest.csv"
DEFENSIVE_DIR = PROJECT_ROOT / "data" / "test_images" / "defensive"


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _pick_ct_ich_cases():
    with open(CT_ICH_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    positive = next(r for r in rows if r["any_hemorrhage"] == "1")
    negative = next(r for r in rows if r["any_hemorrhage"] == "0")
    return [
        ("CT-ICH positive", CT_ICH_IMAGES / positive["image_file"]),
        ("CT-ICH negative", CT_ICH_IMAGES / negative["image_file"]),
    ]


def _pick_defensive_cases(n=2):
    cases = []
    for sub in ("non_medical", "non_brain_medical"):
        d = DEFENSIVE_DIR / sub
        files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
        for p in files[:n]:
            cases.append((f"defensive/{sub}", p))
    return cases


async def main():
    cases = _pick_ct_ich_cases() + _pick_defensive_cases()
    results = []
    for label, path in cases:
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        text = await inference.generate_medical_report(
            _b64(path), prompt="(unused for image_retrieval)", system_prompt="(unused)",
            image_retrieval=True,
        )
        is_refusal = text.strip() == inference._GATE_REFUSAL_MESSAGE
        results.append((label, path.name, is_refusal, text[:120]))
        tag = "REFUSED" if is_refusal else "ANSWERED"
        print(f"[{tag:8}] {label:24} {path.name:30} {text[:100]!r}")

    n_ct_ich = sum(1 for r in results if r[0].startswith("CT-ICH"))
    n_ct_ich_refused = sum(1 for r in results if r[0].startswith("CT-ICH") and r[2])
    n_defensive = sum(1 for r in results if r[0].startswith("defensive"))
    n_defensive_refused = sum(1 for r in results if r[0].startswith("defensive") and r[2])

    print(f"\nCT-ICH (should ANSWER, not refuse): {n_ct_ich - n_ct_ich_refused}/{n_ct_ich} answered")
    print(f"Defensive (should REFUSE):          {n_defensive_refused}/{n_defensive} refused")

    ok = (n_ct_ich_refused == 0) and (n_defensive_refused == n_defensive)
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
