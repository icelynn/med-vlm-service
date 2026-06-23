# -*- coding: utf-8 -*-
"""
build_context() -- the one retrieval-injection seam for the RAG ablations.
============================================================================

The text-RAG ablation (R1) plugs in text retrieval here. The image-retrieval
ablation (R2) swaps in an image-similarity backend at this exact function --
the call site in run_baseline.py (`--context none|text|image`) never changes.

provider="none"  -> "" (No-RAG baseline behaviour, unchanged)
provider="text"  -> embed `query`, retrieve top-k subtype descriptions from
                    the ChromaDB KB built by build_kb.py, format as a context
                    block appended to the system prompt.
provider="image" -> embed `image_path` (the CURRENT query slice -- unlike
                    "text", this is per-slice, not a fixed injection) with
                    BiomedCLIP, retrieve top-k visually-similar RSNA pool
                    exemplars from the index build_image_index.py built, and
                    return them as (image_path, label_text) pairs for the
                    caller to splice into a multi-image few-shot chat
                    (see feasibility_check.run_inference_fewshot). This is
                    NOT a text string like the other two providers -- R2's
                    whole point is showing the model images, not describing
                    them in words (that's what R1 already tried and it
                    didn't fix the perception gap, see 核心難題 4).
"""

import json
import sys
from pathlib import Path

import chromadb
import numpy as np
from chromadb.utils import embedding_functions

RAG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rag"
CHROMA_DIR = RAG_DIR / "chroma"
COLLECTION_NAME = "ich_subtypes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

IMAGE_INDEX_DIR = RAG_DIR / "image_index"
POOL_IMAGES_DIR = RAG_DIR.parent / "rsna" / "pool_images"
SUBTYPE_KEYS = ["IPH", "IVH", "SAH", "EDH", "SDH"]

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_QUERY = "acute intracranial hemorrhage findings on a head CT slice"

_collection = None  # lazy singleton: load the embedding model once per process
_image_index = None  # lazy singleton: (embeddings, labels) loaded once per process


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL)
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def retrieve_text(query=DEFAULT_QUERY, k=5):
    """Return the top-k KB documents for `query`. k=5 == the whole KB (see
    build_kb.py docstring on why retrieval ~= fixed injection on this closed
    5-class set)."""
    coll = _get_collection()
    result = coll.query(query_texts=[query], n_results=k)
    return result["documents"][0]


def _get_image_index():
    global _image_index
    if _image_index is None:
        embeddings = np.load(IMAGE_INDEX_DIR / "embeddings.npy")
        with open(IMAGE_INDEX_DIR / "labels.json", encoding="utf-8") as f:
            labels = json.load(f)
        _image_index = (embeddings, labels)
    return _image_index


def _label_text(label_row):
    """Format a pool exemplar's GT label as the exact answer grammar
    run_baseline.py's USER_PROMPT asks for, so the few-shot assistant turns
    look like valid completions of the same task the query image is asked."""
    present = [k for k in SUBTYPE_KEYS if label_row[k]]
    if not present:
        return "HEMORRHAGE: no\nSUBTYPES: none"
    return f"HEMORRHAGE: yes\nSUBTYPES: {', '.join(present)}"


def retrieve_image_exemplars(image_path, k=3):
    """Embed the CURRENT query slice with BiomedCLIP, retrieve the top-k
    visually-nearest RSNA pool exemplars, and return them as a list of
    (image_path, label_text) pairs in similarity order."""
    from biomedclip_embed import embed_image

    embeddings, labels = _get_image_index()
    query_emb = embed_image(image_path)
    sims = embeddings @ query_emb
    top = np.argsort(-sims)[:k]
    return [(POOL_IMAGES_DIR / labels[i]["image_file"], _label_text(labels[i])) for i in top]


def build_context(provider, query=DEFAULT_QUERY, image_path=None):
    if provider == "none":
        return ""
    if provider == "text":
        docs = retrieve_text(query)
        bullets = "\n".join(f"- {d}" for d in docs)
        return (
            "Reference imaging findings for hemorrhage subtypes (background "
            "knowledge only -- judge strictly by what is visible in THIS "
            f"image, not by this list):\n{bullets}"
        )
    if provider == "image":
        if image_path is None:
            raise ValueError("provider='image' needs image_path (retrieval is per-slice)")
        return retrieve_image_exemplars(image_path)
    raise ValueError(f"unknown context provider: {provider!r}")
