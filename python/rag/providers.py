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
                    didn't fix the perception gap -- see the README's
                    Honest limitations section).
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
_image_index_cache = {}  # {index_dir: (embeddings, labels)} -- keyed so a process
                         # can serve both the C-path and B-path indexes without
                         # one silently overwriting the other's cache


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


def _get_image_index(index_dir):
    key = str(index_dir)
    if key not in _image_index_cache:
        embeddings = np.load(Path(index_dir) / "embeddings.npy")
        with open(Path(index_dir) / "labels.json", encoding="utf-8") as f:
            labels = json.load(f)
        _image_index_cache[key] = (embeddings, labels)
    return _image_index_cache[key]


def _label_text(label_row):
    """Format a pool exemplar's GT label as the exact answer grammar
    run_baseline.py's USER_PROMPT asks for, so the few-shot assistant turns
    look like valid completions of the same task the query image is asked."""
    present = [k for k in SUBTYPE_KEYS if label_row[k]]
    if not present:
        return "HEMORRHAGE: no\nSUBTYPES: none"
    return f"HEMORRHAGE: yes\nSUBTYPES: {', '.join(present)}"


def _select_contrastive_from_sims(sims, labels, k=3, neg_position="last",
                                  n_pos=2, n_neg=1, random_neg=False, seed=None):
    """Given similarity scores (1D array, one per pool image) and pool labels,
    select contrastive exemplar indices. See retrieve_image_exemplars docstring
    for the selection logic.

    neg_position controls where the hard negative sits in the returned order
    (recency-bias ablation): 'last' (default, current published behaviour) |
    'first' | 'middle'. Only applies to the default n_pos=2/n_neg=1 case --
    with other counts (presence-vs-count ablation) exemplars are just
    returned positives-then-negatives, ordering isn't the variable under
    test there.

    n_pos/n_neg override the default 2 positives + 1 hard negative.

    random_neg=True (soft-negative ablation) draws the negative(s)
    uniformly at random from all opposite-label pool images instead of the
    most visually similar one -- isolates "negative exists" from "negative is
    visually confusing". seed makes the draw reproducible (pass the query
    image_path)."""
    order = np.argsort(-sims)
    # First image sets the reference label
    ref_label = labels[order[0]]["any_hemorrhage"]
    positives = [order[0]]
    negs = []
    for idx in order[1:]:
        if len(positives) >= n_pos and (len(negs) >= n_neg or random_neg):
            break
        if labels[idx]["any_hemorrhage"] == ref_label:
            if len(positives) < n_pos:
                positives.append(idx)
        elif not random_neg:
            if len(negs) < n_neg:
                negs.append(idx)
    if len(positives) < n_pos:
        # Degenerate case: not enough same-label images in the pool.
        for idx in order[1:]:
            if idx not in positives and idx not in negs:
                positives.append(idx)
            if len(positives) >= n_pos:
                break
    if random_neg and n_neg > 0:
        import random as _random
        opposite = [i for i in range(len(labels))
                   if labels[i]["any_hemorrhage"] != ref_label and i not in positives]
        rng = _random.Random(seed)
        negs = rng.sample(opposite, min(n_neg, len(opposite)))
    if len(negs) < n_neg and n_neg > 0:
        # No (more) opposite-label images at all -- fall back to next nearest.
        for idx in order:
            if idx not in positives and idx not in negs:
                negs.append(idx)
            if len(negs) >= n_neg:
                break
    pos, negs = positives[:n_pos], negs[:n_neg]
    if n_pos == 2 and n_neg == 1:
        hard_neg = negs[0]
        if neg_position == "first":
            return [hard_neg] + pos
        if neg_position == "middle":
            return [pos[0], hard_neg, pos[1]]
        return pos + [hard_neg]  # "last", current published behaviour
    return pos + negs


def retrieve_image_exemplars(image_path, k=3, index_dir=IMAGE_INDEX_DIR,
                             pool_images_dir=POOL_IMAGES_DIR, random_baseline=False,
                             contrastive=False, contrastive_neg_position="last",
                             contrastive_n_pos=2, contrastive_n_neg=1,
                             contrastive_random_neg=False):
    """Embed the CURRENT query slice with BiomedCLIP, retrieve the top-k
    visually-nearest pool exemplars, and return them as a list of
    (image_path, label_text) pairs in similarity order.

    index_dir/pool_images_dir default to the R2-C index (RSNA pool -> RSNA
    eval). Pass the R2-B (harmonized RSNA pool -> CT-ICH eval) paths to reuse
    this exact same retrieval code for the other path.

    random_baseline=True skips similarity search entirely and draws k random
    pool exemplars instead (deterministic per query image, via a seed derived
    from image_path, so reruns/resumes are reproducible). This isolates
    "does the model benefit from any few-shot exemplar at all" (format
    demonstration) from "does it benefit from a VISUALLY RELEVANT one".

    contrastive=True selects 2 high-similarity same-label exemplars + 1 hard
    negative (visually similar but opposite any_hemorrhage label), instead of
    plain top-k. The hard negative teaches the model to discriminate rather
    than rely on superficial visual similarity. k is ignored (always 2+1=3).
    No test-time label leakage: selection uses pool GT labels only, not the
    query's GT."""
    embeddings, labels = _get_image_index(index_dir)
    if random_baseline:
        import random as _random
        rng = _random.Random(str(image_path))
        top = rng.sample(range(len(labels)), min(k, len(labels)))
    elif contrastive:
        from biomedclip_embed import embed_image
        query_emb = embed_image(image_path)
        sims = embeddings @ query_emb
        top = _select_contrastive_from_sims(sims, labels, k=k,
                                            neg_position=contrastive_neg_position,
                                            n_pos=contrastive_n_pos, n_neg=contrastive_n_neg,
                                            random_neg=contrastive_random_neg,
                                            seed=str(image_path))
    else:
        from biomedclip_embed import embed_image
        query_emb = embed_image(image_path)
        sims = embeddings @ query_emb
        top = np.argsort(-sims)[:k]
    return [(Path(pool_images_dir) / labels[i]["image_file"], _label_text(labels[i])) for i in top]


def build_context(provider, query=DEFAULT_QUERY, image_path=None,
                  image_index_dir=IMAGE_INDEX_DIR, image_pool_dir=POOL_IMAGES_DIR,
                  image_random_baseline=False, image_contrastive=False,
                  image_contrastive_neg_position="last",
                  image_contrastive_n_pos=2, image_contrastive_n_neg=1,
                  image_contrastive_random_neg=False,
                  image_k=3, text_k=5):
    if provider == "none":
        return ""
    if provider == "text":
        docs = retrieve_text(query, k=text_k)
        bullets = "\n".join(f"- {d}" for d in docs)
        return (
            "Reference imaging findings for hemorrhage subtypes (background "
            "knowledge only -- judge strictly by what is visible in THIS "
            f"image, not by this list):\n{bullets}"
        )
    if provider == "image":
        if image_path is None:
            raise ValueError("provider='image' needs image_path (retrieval is per-slice)")
        return retrieve_image_exemplars(image_path, k=image_k, index_dir=image_index_dir,
                                        pool_images_dir=image_pool_dir,
                                        random_baseline=image_random_baseline,
                                        contrastive=image_contrastive,
                                        contrastive_neg_position=image_contrastive_neg_position,
                                        contrastive_n_pos=image_contrastive_n_pos,
                                        contrastive_n_neg=image_contrastive_n_neg,
                                        contrastive_random_neg=image_contrastive_random_neg)
    raise ValueError(f"unknown context provider: {provider!r}")
