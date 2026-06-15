#!/usr/bin/env bash
# Feasibility matrix. Run this ON the T4 EC2 instance (inside your venv).
#
#   bash run_feasibility.sh
#
# Each model runs in its OWN python process so VRAM is measured cleanly and an
# OOM on one model does not abort the rest. Results accumulate in results.jsonl.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use python3 if available, else python (some hosts only ship one of them).
PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then echo "ERROR: no python3/python on PATH"; exit 1; fi
echo "[info] using interpreter: $PY"

IMAGE="${1:-../../data/test_images/normal_xray.jpg}"
OUT="results.jsonl"
: > "$OUT"   # truncate previous run

run () {
  echo "============================================================"
  echo ">>> $1   (quant=$2)"
  echo "============================================================"
  "$PY" feasibility_check.py --model "$1" --quant "$2" --image "$IMAGE" --out "$OUT" \
    || echo "[continue] $1 ($2) failed — moving on"
  echo
}

# --- Main subject candidates (general, non-medical VLM) ---------------------
run "Qwen/Qwen3-VL-4B-Instruct" "none"    # expect to fit comfortably (~9 GB)
run "Qwen/Qwen3-VL-8B-Instruct" "4bit"    # the squeeze test on a T4

# --- Medical baseline (plan condition C) ------------------------------------
run "google/medgemma-4b-it"     "none"    # expect ~9 GB fp16

# --- (optional) original plan model, for the record ------------------------
# run "meta-llama/Llama-3.2-11B-Vision-Instruct" "4bit"

echo "============================================================"
echo "Done. Summary table:"
echo "============================================================"
"$PY" - <<'PY'
import json
rows = [json.loads(l) for l in open("results.jsonl") if l.strip()]
hdr = f"{'model':40} {'quant':6} {'status':6} {'VRAM(GB)':9} {'tok/s':7} {'fits16':6}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r.get('model',''):40} {r.get('quant',''):6} {r.get('status',''):6} "
          f"{str(r.get('peak_vram_reserved_gb','-')):9} "
          f"{str(r.get('tokens_per_second','-')):7} {str(r.get('fits_16gb','-')):6}")
PY
echo
echo "Paste results.jsonl (or the table above) back into the chat to finalize the report."
