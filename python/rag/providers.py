# -*- coding: utf-8 -*-
"""
build_context() -- the one retrieval-injection seam for the RAG ablations.
============================================================================

The text-RAG ablation (R1) plugs in text retrieval here. The planned image-
retrieval ablation (R2) swaps in an image-similarity backend at this exact
function -- the call site in run_baseline.py (`--context none|text|image`)
never changes.

provider="none"  -> "" (No-RAG baseline behaviour, unchanged)
provider="text"  -> embed `query`, retrieve top-k subtype descriptions from
                    the ChromaDB KB built by build_kb.py, format as a context
                    block appended to the system prompt.
provider="image" -> not implemented yet; raises so a typo doesn't silently
                    fall through to no-op.
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

RAG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rag"
CHROMA_DIR = RAG_DIR / "chroma"
COLLECTION_NAME = "ich_subtypes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_QUERY = "acute intracranial hemorrhage findings on a head CT slice"

_collection = None  # lazy singleton: load the embedding model once per process


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


def build_context(provider, query=DEFAULT_QUERY):
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
        raise NotImplementedError(
            "image retrieval (R2) is not implemented yet -- swap in the "
            "image-similarity backend here, same call site")
    raise ValueError(f"unknown context provider: {provider!r}")
