# -*- coding: utf-8 -*-
"""
Build the text knowledge base for R1 (text RAG ablation).
===========================================================

5 fixed CT imaging-finding descriptions, one per ICH subtype, embedded into a
local ChromaDB collection.

This is a CLOSED set of 5 documents -- with only 5 candidates, "retrieve the
most relevant K of N" is barely different from "return all N" (retrieval's
filtering value only shows up when the knowledge base is large and queries
are discriminative). We still implement real embed-and-query retrieval rather
than a hardcoded constant string, so the mechanism is genuine and
`providers.py::build_context()` stays the one seam a planned image-retrieval
ablation (a real, large, non-trivial pool) can plug into.

Run once (or whenever the descriptions change):
    python build_kb.py
"""

from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

RAG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rag"
CHROMA_DIR = RAG_DIR / "chroma"
COLLECTION_NAME = "ich_subtypes"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # light, CPU-fine

# Standard neuroradiology teaching-point descriptions (location/shape/density),
# the kind of textbook knowledge a text-RAG system would inject -- not derived
# from any specific patient image.
DESCRIPTIONS = {
    "EDH": (
        "Epidural hematoma: biconvex (lens-shaped) hyperdense collection "
        "between the skull and the dura. Does not cross suture lines. Often "
        "associated with an overlying skull fracture and arterial (middle "
        "meningeal artery) bleeding."
    ),
    "SDH": (
        "Subdural hematoma: crescent-shaped (concave), hyperdense in the "
        "acute phase, collection along the brain's convexity. Crosses "
        "suture lines and can extend along the falx; typically from torn "
        "bridging veins, often after trauma in elderly or anticoagulated "
        "patients."
    ),
    "SAH": (
        "Subarachnoid hemorrhage: hyperdense blood filling the cortical "
        "sulci, basal cisterns, and fissures, following the CSF spaces "
        "(classic 'star/spider' pattern in the basal cisterns). Commonly "
        "from a ruptured aneurysm or trauma."
    ),
    "IVH": (
        "Intraventricular hemorrhage: hyperdense blood within the "
        "ventricular system, often layering dependently and forming a fluid "
        "level. Can cause acute hydrocephalus. Frequently a result of "
        "extension from an adjacent intraparenchymal or subarachnoid bleed."
    ),
    "IPH": (
        "Intraparenchymal (intracerebral) hemorrhage: a focal hyperdense "
        "lesion within the brain tissue itself, often with surrounding "
        "hypodense vasogenic edema. Hypertensive bleeds favor the basal "
        "ganglia, thalamus, pons, and cerebellum; lobar bleeds in older "
        "patients raise suspicion for amyloid angiopathy."
    ),
}


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL)

    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    coll = client.create_collection(COLLECTION_NAME, embedding_function=embed_fn)

    coll.add(
        ids=list(DESCRIPTIONS.keys()),
        documents=list(DESCRIPTIONS.values()),
        metadatas=[{"subtype": k} for k in DESCRIPTIONS],
    )
    print(f"[ok] wrote {coll.count()} entries -> {CHROMA_DIR.relative_to(RAG_DIR.parent.parent)}")


if __name__ == "__main__":
    main()
