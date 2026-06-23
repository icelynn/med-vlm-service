# -*- coding: utf-8 -*-
"""
BiomedCLIP image embedding -- shared by retrieval_sentinel.py, build_image_index.py,
and providers.py's image-retrieval backend (R2).
======================================================================================

Lazy-imports torch/open_clip so nothing that only needs the rest of this
package (e.g. running --self-test scripts) is forced to have them installed.
"""

import numpy as np

BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

_model = None
_preprocess = None
_torch = None
_device = None


def load_biomedclip(device="cpu"):
    """Lazy singleton: load BiomedCLIP once per process. Returns
    (model, preprocess, torch) -- torch is returned too so callers don't need
    their own top-level `import torch` just to build tensors/no_grad blocks."""
    global _model, _preprocess, _torch, _device
    if _model is None or _device != device:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL)
        _model = model.to(device).eval()
        _preprocess = preprocess
        _torch = torch
        _device = device
    return _model, _preprocess, _torch


def embed_images(image_paths, device="cpu", batch_size=16):
    """L2-normalized BiomedCLIP image embeddings, one row per path, in the
    same order as image_paths."""
    from PIL import Image

    model, preprocess, torch = load_biomedclip(device=device)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch = torch.stack([
                preprocess(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def embed_image(image_path, device="cpu"):
    """Single-image convenience wrapper (1D embedding vector)."""
    return embed_images([image_path], device=device)[0]
