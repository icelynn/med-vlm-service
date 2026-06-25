# -*- coding: utf-8 -*-
"""Offline unit tests for the contrastive ICL exemplar selection logic.

Tests `_select_contrastive_from_sims` directly (no GPU, no BiomedCLIP, no
file I/O) with synthetic similarity vectors and pool labels.

Run:  python -m pytest python/tests/test_contrastive_retrieval.py -v
  or:  python python/tests/test_contrastive_retrieval.py
"""
import sys
from pathlib import Path

import numpy as np

# Add rag/ to path so we can import providers._select_contrastive_from_sims
RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))

from providers import _select_contrastive_from_sims  # noqa: E402


def _make_labels(hem_flags):
    """Build a minimal labels list (just any_hemorrhage) from a list of 0/1."""
    return [{"any_hemorrhage": h} for h in hem_flags]


def test_basic_contrastive_selection():
    """Top-1 is hemorrhage=1; should pick 2 hemorrhage + 1 non-hemorrhage."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = _make_labels([1, 1, 0, 1, 0, 0])
    result = _select_contrastive_from_sims(sims, labels)
    assert len(result) == 3, f"expected 3 indices, got {len(result)}"
    # First two should be hemorrhage (matching top-1's label)
    assert labels[result[0]]["any_hemorrhage"] == 1
    assert labels[result[1]]["any_hemorrhage"] == 1
    # Third (hard negative) should be non-hemorrhage
    assert labels[result[2]]["any_hemorrhage"] == 0
    # Hard negative should be the most similar non-hemorrhage
    assert result[2] == 2  # index 2 has sim 0.7, highest among 0-labels


def test_top1_is_negative():
    """Top-1 is hemorrhage=0; should pick 2 non-hem + 1 hemorrhage hard neg."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = _make_labels([0, 0, 1, 0, 1, 1])
    result = _select_contrastive_from_sims(sims, labels)
    assert len(result) == 3
    assert labels[result[0]]["any_hemorrhage"] == 0
    assert labels[result[1]]["any_hemorrhage"] == 0
    assert labels[result[2]]["any_hemorrhage"] == 1
    # Hard negative = most similar hemorrhage image
    assert result[2] == 2  # sim 0.7


def test_hard_negative_is_most_similar_opposite():
    """The hard negative must be the highest-similarity opposite-label image,
    not just any opposite-label image."""
    sims = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.10])
    labels = _make_labels([1, 1, 1, 0, 0, 0])
    result = _select_contrastive_from_sims(sims, labels)
    # Positives: index 0 (0.95) and 1 (0.90), both label=1
    assert result[0] == 0
    assert result[1] == 1
    # Hard negative: index 3 (0.80) — most similar among label=0
    assert result[2] == 3


def test_ordering_preserves_similarity_for_positives():
    """Positives should be in descending similarity order."""
    sims = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 0, 1, 0, 0])
    result = _select_contrastive_from_sims(sims, labels)
    # First positive has highest sim
    assert sims[result[0]] >= sims[result[1]]
    # Hard negative (label=0) is index 2 (sim 0.8), the most similar 0-label
    assert result[2] == 2


def test_all_same_label_fallback():
    """If all pool images share the same label (no opposite available),
    the hard negative falls back to the next nearest image."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 1, 1, 1])
    result = _select_contrastive_from_sims(sims, labels)
    assert len(result) == 3
    # All are label 1, so hard neg is just the 3rd nearest
    assert result[2] == 2


def test_minimal_pool():
    """Pool with exactly 3 images: 2 same + 1 different."""
    sims = np.array([0.9, 0.8, 0.7])
    labels = _make_labels([1, 1, 0])
    result = _select_contrastive_from_sims(sims, labels)
    assert len(result) == 3
    assert labels[result[0]]["any_hemorrhage"] == 1
    assert labels[result[1]]["any_hemorrhage"] == 1
    assert labels[result[2]]["any_hemorrhage"] == 0


def test_no_gt_leakage_verification():
    """Verify that the selection only uses pool labels, not the query's.
    This test passes by construction (the function signature only takes
    sims + pool labels), but we make it explicit."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 0, 1, 0, 1])
    # The function has no access to the query's GT — it only sees pool labels
    result = _select_contrastive_from_sims(sims, labels)
    assert len(result) == 3
    # Result is deterministic given sims + pool labels
    result2 = _select_contrastive_from_sims(sims, labels)
    assert result == result2


if __name__ == "__main__":
    tests = [
        test_basic_contrastive_selection,
        test_top1_is_negative,
        test_hard_negative_is_most_similar_opposite,
        test_ordering_preserves_similarity_for_positives,
        test_all_same_label_fallback,
        test_minimal_pool,
        test_no_gt_leakage_verification,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
