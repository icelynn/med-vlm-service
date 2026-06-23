# -*- coding: utf-8 -*-
"""
Model & hardware feasibility harness
====================================

Goal
----
Answer the key feasibility question for this project:
    "Can a (quantized) general VLM actually load and run a single chest X-ray
     inference on a T4 (16 GB)? If not, which smaller model does?"

It loads ONE Hugging Face vision-language model, runs ONE image+prompt
inference, and reports the two numbers a go/no-go decision needs:
    1. Peak VRAM  (does it fit in 16 GB with headroom for KV-cache?)
    2. Latency    (load time + single-inference generation time, tokens/sec)

Run one model per process (so VRAM is measured cleanly and an OOM on one model
does not poison the next). A wrapper loop over models lives in run_feasibility.sh.

Designed for an NVIDIA T4 (Turing, sm_75):
    * dtype = float16   (T4 has NO fast bf16 -- do not use bfloat16)
    * attention = "sdpa"/"eager"  (NO flash-attention-2 on Turing)
    * optional 4-bit / 8-bit via bitsandbytes for the larger 8B models

Usage
-----
    python feasibility_check.py \
        --model Qwen/Qwen3-VL-4B-Instruct \
        --quant none \
        --image ../../data/test_images/normal_xray.jpg \
        --max-new-tokens 256 \
        --out results.jsonl
"""

import argparse
import gc
import json
import time
import platform
from datetime import datetime, timezone

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


def bytes_to_gb(n):
    return round(n / (1024 ** 3), 2)


def resolve_dtype(model_id, dtype_arg):
    """Pick the compute dtype.

    Most VLMs (e.g. Qwen3-VL) are numerically fine in fp16 on a T4. The Gemma
    family (incl. MedGemma) is trained in bf16 and OVERFLOWS in fp16 -> it emits
    only <pad> tokens. So Gemma-family models must use bfloat16 even though the
    T4 has no native bf16 acceleration (it still runs, just slower).
    """
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    # auto
    mid = model_id.lower()
    if "gemma" in mid:
        return torch.bfloat16
    return torch.float16


def build_load_kwargs(quant, dtype):
    """Construct from_pretrained kwargs tuned for a T4.

    A non-flash attention backend is mandatory on Turing; dtype is resolved by
    resolve_dtype() (fp16 for most models, bf16 for Gemma-family).
    """
    kwargs = dict(
        dtype=dtype,                      # fp16 for most; bf16 for Gemma-family
        device_map="auto",                # accelerate places weights on the GPU
        attn_implementation="sdpa",       # Turing has no flash-attn-2
        trust_remote_code=True,
    )
    if quant == "4bit":
        assert _HAS_BNB, "bitsandbytes not installed -- pip install bitsandbytes"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,           # match resolved dtype
        )
        kwargs.pop("dtype", None)
    elif quant == "8bit":
        assert _HAS_BNB, "bitsandbytes not installed -- pip install bitsandbytes"
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("dtype", None)
    elif quant != "none":
        raise ValueError("Unknown --quant value: " + quant)
    return kwargs


def load_model(model_id, quant, dtype):
    """Load processor + model, returning (processor, model, load_seconds)."""
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs = build_load_kwargs(quant, dtype)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    except (ValueError, ImportError, RuntimeError) as e:
        # Some architectures reject "sdpa"; retry once with eager attention.
        print("[warn] sdpa load failed (" + str(e) + "); retrying with eager...")
        load_kwargs["attn_implementation"] = "eager"
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return processor, model, time.perf_counter() - t0


def run_inference(processor, model, image_path, prompt, system_prompt, max_new_tokens,
                  max_image_size=896):
    """Run a single image+text generation. Returns (text, gen_seconds, n_new_tokens)."""
    image = Image.open(image_path).convert("RGB")
    # Cap resolution: Qwen-VL vision-token count scales with image size, and an
    # un-resized X-ray can blow up activation memory enough to OOM even a 4B model
    # on a 14.5 GB T4. thumbnail() keeps aspect ratio and only ever shrinks.
    image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)

    # Unified chat format works for both Qwen3-VL and MedGemma (Gemma-3 MM).
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    })

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - t0

    new_tokens = out[0][input_len:]
    n_new = int(new_tokens.shape[-1])
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text, gen_seconds, n_new


def _load_capped_image(image_path, max_image_size):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
    return image


def run_inference_fewshot(processor, model, query_image_path, exemplars, prompt,
                          system_prompt, max_new_tokens, max_image_size=896):
    """Multi-image few-shot variant of run_inference, for R2 (image-retrieval
    ablation): shows the model `exemplars` -- a list of (image_path,
    label_text) pairs, each rendered as one user-image-turn + one
    assistant-answer-turn demonstrating the exact answer grammar -- before
    the real query image+question. This is a SEPARATE function (not a
    run_inference parameter) because the message structure genuinely differs
    (N+1 images across multiple turns vs. one image in one turn), not just
    the inputs; existing call sites (No-RAG, R1 text-RAG) are unaffected.

    Returns (text, gen_seconds, n_new_tokens), same shape as run_inference.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    for ex_path, ex_label in exemplars:
        ex_image = _load_capped_image(ex_path, max_image_size)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": ex_image},
                {"type": "text", "text": prompt},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex_label}],
        })
    query_image = _load_capped_image(query_image_path, max_image_size)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": query_image},
            {"type": "text", "text": prompt},
        ],
    })

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - t0

    new_tokens = out[0][input_len:]
    n_new = int(new_tokens.shape[-1])
    text = processor.decode(new_tokens, skip_special_tokens=True)
    return text, gen_seconds, n_new


DEFAULT_PROMPT = ("Please systematically evaluate this chest radiograph and "
                  "describe any findings.")
DEFAULT_SYSTEM = ("You are a radiologist. Describe the chest X-ray findings "
                  "objectively and concisely.")


def main():
    ap = argparse.ArgumentParser(description="T4 VLM feasibility check")
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--quant", default="none", choices=["none", "4bit", "8bit"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"],
                    help="auto = bf16 for Gemma-family, fp16 otherwise")
    ap.add_argument("--image", default="../../data/test_images/normal_xray.jpg")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-image-size", type=int, default=896,
                    help="longest image edge in px; caps vision-token memory")
    ap.add_argument("--out", default="results.jsonl", help="append JSONL results here")
    args = ap.parse_args()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "quant": args.quant,
        "max_new_tokens": args.max_new_tokens,
        "status": "error",
    }

    if not torch.cuda.is_available():
        record["error"] = "CUDA not available -- run this ON the T4 instance."
        print(json.dumps(record, indent=2))
        _append(args.out, record)
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = bytes_to_gb(torch.cuda.get_device_properties(0).total_memory)
    dtype = resolve_dtype(args.model, args.dtype)
    record.update(gpu=gpu_name, total_vram_gb=total_vram, dtype=str(dtype),
                  torch=torch.__version__, python=platform.python_version())
    print("[info] GPU=" + gpu_name + " total_vram=" + str(total_vram) + " GB  "
          + "model=" + args.model + " quant=" + args.quant)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        processor, model, load_s = load_model(args.model, args.quant, dtype)
        text, gen_s, n_new = run_inference(
            processor, model, args.image, args.prompt,
            args.system_prompt, args.max_new_tokens, args.max_image_size)

        peak_alloc = bytes_to_gb(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_gb(torch.cuda.max_memory_reserved())
        tok_per_s = round(n_new / gen_s, 2) if gen_s > 0 else None

        record.update(
            status="ok",
            load_seconds=round(load_s, 1),
            generation_seconds=round(gen_s, 1),
            new_tokens=n_new,
            tokens_per_second=tok_per_s,
            peak_vram_allocated_gb=peak_alloc,
            peak_vram_reserved_gb=peak_reserved,
            fits_16gb=bool(peak_reserved < 15.0),   # ~1 GB headroom margin
            output_preview=text[:500],
        )
        print("\n===== RESULT =====")
        print(json.dumps({k: v for k, v in record.items()
                          if k != "output_preview"}, indent=2))
        print("\n----- model output (first 500 chars) -----")
        print(text[:500])
    except torch.cuda.OutOfMemoryError as e:
        record["status"] = "oom"
        record["error"] = "CUDA OOM: " + str(e)
        print("[FAIL] OOM -- " + args.model + " (" + args.quant + ") does NOT fit.")
    except Exception as e:
        record["error"] = type(e).__name__ + ": " + str(e)
        print("[FAIL] " + type(e).__name__ + ": " + str(e))
    finally:
        _append(args.out, record)
        # Each model runs in its own process (see run_feasibility.sh), so VRAM is
        # reclaimed on exit. We still gc + empty_cache for tidiness.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("[info] appended result to " + path)


if __name__ == "__main__":
    main()
