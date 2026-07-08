# -*- coding: utf-8 -*-
"""Tests for the ablation-flag parameters added to _select_contrastive_from_sims.

The existing test_contrastive_retrieval.py covers the default configuration
(n_pos=2, n_neg=1, neg_position="last"). This file covers the four mechanism-
ablation variants that were added in the contrastive ICL follow-up experiments:

  - neg_position="first" / "middle"  (recency-bias ablation)
  - n_pos=3, n_neg=0                 (presence-vs-count: remove the negative)
  - n_pos=2, n_neg=2                 (presence-vs-count: extra negative)
  - random_neg=True + seed           (soft-negative ablation)

No GPU, no BiomedCLIP, no file I/O required.

Run:  python -m pytest python/tests/test_contrastive_ablation_params.py -v
  or:  python python/tests/test_contrastive_ablation_params.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rag"))

from providers import _select_contrastive_from_sims  # noqa: E402


def _make_labels(hem_flags):
    return [{"any_hemorrhage": h} for h in hem_flags]


# ---------------------------------------------------------------------------
# neg_position ablation (recency-bias)
# ---------------------------------------------------------------------------

def test_neg_position_first():
    """Hard negative must be the first element when neg_position='first'."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 0, 0, 0])
    result = _select_contrastive_from_sims(sims, labels, neg_position="first")
    assert len(result) == 3
    assert labels[result[0]]["any_hemorrhage"] == 0, "hard neg must be first"
    assert labels[result[1]]["any_hemorrhage"] == 1
    assert labels[result[2]]["any_hemorrhage"] == 1
    # The hard neg is the most-similar opposite-label image (index 2, sim=0.7)
    assert result[0] == 2


def test_neg_position_middle():
    """Hard negative must be the middle element when neg_position='middle'."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 0, 0, 0])
    result = _select_contrastive_from_sims(sims, labels, neg_position="middle")
    assert len(result) == 3
    assert labels[result[0]]["any_hemorrhage"] == 1
    assert labels[result[1]]["any_hemorrhage"] == 0, "hard neg must be middle"
    assert labels[result[2]]["any_hemorrhage"] == 1
    assert result[1] == 2


def test_neg_position_last_is_default():
    """'last' behaviour must be identical whether specified explicitly or omitted."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 0, 0, 0])
    default = _select_contrastive_from_sims(sims, labels)
    explicit = _select_contrastive_from_sims(sims, labels, neg_position="last")
    assert default == explicit


# ---------------------------------------------------------------------------
# presence-vs-count ablation (n_pos / n_neg)
# ---------------------------------------------------------------------------

def test_3pos_0neg_returns_three_positives():
    """n_pos=3, n_neg=0 must return three same-label images with no negative."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 0, 1, 0])
    result = _select_contrastive_from_sims(sims, labels, n_pos=3, n_neg=0)
    assert len(result) == 3
    assert all(labels[i]["any_hemorrhage"] == 1 for i in result)
    # Must pick the three highest-sim same-label images: indices 0, 1, 3
    assert set(result) == {0, 1, 3}


def test_2pos_2neg_returns_correct_split():
    """n_pos=2, n_neg=2 must return 2 positives followed by 2 negatives."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = _make_labels([1, 1, 0, 0, 1, 0])
    result = _select_contrastive_from_sims(sims, labels, n_pos=2, n_neg=2)
    assert len(result) == 4
    assert labels[result[0]]["any_hemorrhage"] == 1
    assert labels[result[1]]["any_hemorrhage"] == 1
    assert labels[result[2]]["any_hemorrhage"] == 0
    assert labels[result[3]]["any_hemorrhage"] == 0
    # Negatives: indices 2 (sim=0.7) and 3 (sim=0.6), in similarity order
    assert result[2] == 2
    assert result[3] == 3


def test_1pos_1neg():
    """n_pos=1, n_neg=1 must return exactly 2 images."""
    sims = np.array([0.9, 0.8, 0.7])
    labels = _make_labels([1, 0, 1])
    result = _select_contrastive_from_sims(sims, labels, n_pos=1, n_neg=1)
    assert len(result) == 2
    assert labels[result[0]]["any_hemorrhage"] == 1  # index 0
    assert labels[result[1]]["any_hemorrhage"] == 0  # index 1


# ---------------------------------------------------------------------------
# soft-negative ablation (random_neg)
# ---------------------------------------------------------------------------

def test_random_neg_draws_from_opposite_label():
    """random_neg=True must still draw the negative from the opposite-label pool."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 1, 0, 0])  # opposite-label pool: indices 3, 4
    result = _select_contrastive_from_sims(sims, labels, random_neg=True, seed="test.png")
    assert len(result) == 3
    assert labels[result[0]]["any_hemorrhage"] == 1
    assert labels[result[1]]["any_hemorrhage"] == 1
    assert labels[result[2]]["any_hemorrhage"] == 0


def test_random_neg_is_reproducible_with_same_seed():
    """Same seed must produce the same draw every time."""
    sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = _make_labels([1, 1, 1, 0, 0])
    r1 = _select_contrastive_from_sims(sims, labels, random_neg=True, seed="img_001.png")
    r2 = _select_contrastive_from_sims(sims, labels, random_neg=True, seed="img_001.png")
    assert r1 == r2


def test_random_neg_varies_across_seeds():
    """Different seeds must not always pick the same negative (requires a large
    enough opposite-label pool to have real choices)."""
    sims = np.array([0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])
    labels = _make_labels([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 8 opposite-label options
    chosen = {
        _select_contrastive_from_sims(sims, labels, random_neg=True, seed=f"img_{i}.png")[2]
        for i in range(20)
    }
    assert len(chosen) > 1, "random_neg should select different indices across seeds"


def test_random_neg_not_same_as_hard_neg():
    """With a clear hardest negative (highest-sim opposite), random_neg must not
    always pick it — confirming it's uniform, not sim-ranked."""
    sims = np.array([0.95, 0.90, 0.85, 0.80, 0.10, 0.05])
    # index 2 (sim=0.85) is the obvious hard neg; opposite-label pool also has 3,4,5
    labels = _make_labels([1, 1, 0, 0, 0, 0])
    choices = {
        _select_contrastive_from_sims(sims, labels, random_neg=True, seed=f"q{i}.png")[2]
        for i in range(30)
    }
    # If it were sim-ranked, we'd always get index 2; uniform should hit others too
    assert 2 in choices  # must be reachable
    assert len(choices) > 1  # must not be the only one reached


if __name__ == "__main__":
    tests = [
        test_neg_position_first,
        test_neg_position_middle,
        test_neg_position_last_is_default,
        test_3pos_0neg_returns_three_positives,
        test_2pos_2neg_returns_correct_split,
        test_1pos_1neg,
        test_random_neg_draws_from_opposite_label,
        test_random_neg_is_reproducible_with_same_seed,
        test_random_neg_varies_across_seeds,
        test_random_neg_not_same_as_hard_neg,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
