# -*- coding: utf-8 -*-
"""pytest tests for parse_answer.py — structured + fallback + edge cases.

parse_answer.py already has an inline self-test (run via `python
parse_answer.py`), but it can't be collected alongside the other tests in
python/tests/. This file re-expresses those cases as parametrized pytest
tests and adds edge cases not covered by the original self-test:
case-insensitive HEMORRHAGE line, full synonym names for SDH/EDH, the
'uncertain' keyword on the HEMORRHAGE line, and None/empty input.

Run:  python -m pytest python/tests/test_parse_answer.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from parse_answer import parse_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Core cases (ported from parse_answer.py's _run_self_test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, exp_any, exp_sub, exp_parsed, exp_refused", [
    (
        "HEMORRHAGE: yes\nSUBTYPES: IPH, IVH",
        1,
        {"IPH": 1, "IVH": 1, "SAH": 0, "EDH": 0, "SDH": 0},
        True, False,
    ),
    (
        "HEMORRHAGE: no\nSUBTYPES: none",
        0,
        {"IPH": 0, "IVH": 0, "SAH": 0, "EDH": 0, "SDH": 0},
        True, False,
    ),
    (
        "HEMORRHAGE: yes\nSUBTYPES: possible SAH",
        1,
        {"SAH": -1, "IPH": 0},
        True, False,
    ),
    (
        "There is a hyperdense intraparenchymal hematoma in the right frontal lobe.",
        1,
        {"IPH": 1},
        True, False,
    ),
    (
        "Possible subarachnoid hemorrhage along the left sylvian fissure.",
        -1,
        {"SAH": -1},
        True, False,
    ),
    (
        "No acute intracranial hemorrhage is identified.",
        0,
        {"IPH": 0, "SAH": 0},
        True, False,
    ),
    (
        "As an AI, I cannot provide a medical diagnosis. Please consult a radiologist.",
        None, {}, False, True,
    ),
    (
        "The image shows the brain.",
        None, {}, False, False,
    ),
])
def test_parse_answer_core_cases(text, exp_any, exp_sub, exp_parsed, exp_refused):
    r = parse_answer(text)
    assert r["any_hem"] == exp_any, f"any_hem mismatch: got {r['any_hem']!r}"
    for k, v in exp_sub.items():
        assert r[k] == v, f"{k}: expected {v!r}, got {r[k]!r}"
    assert r["parsed"] == exp_parsed
    assert r["refused"] == exp_refused


# ---------------------------------------------------------------------------
# Edge cases not in the original self-test
# ---------------------------------------------------------------------------

def test_hemorrhage_line_all_caps():
    r = parse_answer("HEMORRHAGE: YES\nSUBTYPES: none")
    assert r["any_hem"] == 1
    assert r["parsed"] is True


def test_hemorrhage_line_mixed_case():
    r = parse_answer("Hemorrhage: No\nSubtypes: none")
    assert r["any_hem"] == 0
    assert r["parsed"] is True


def test_hemorrhage_uncertain_keyword():
    r = parse_answer("HEMORRHAGE: uncertain\nSUBTYPES: none")
    assert r["any_hem"] == -1
    assert r["parsed"] is True


def test_sdh_full_name_detected():
    r = parse_answer("There is a small subdural hematoma along the left convexity.")
    assert r["SDH"] == 1
    assert r["any_hem"] == 1


def test_edh_full_name_detected():
    r = parse_answer("A hyperdense epidural collection is present in the right temporal region.")
    assert r["EDH"] == 1
    assert r["any_hem"] == 1


def test_edh_extradural_synonym():
    r = parse_answer("Small extra-dural hematoma identified.")
    assert r["EDH"] == 1


def test_negated_subtype_not_flagged():
    r = parse_answer("No subdural hematoma is seen. No epidural collection.")
    assert r["SDH"] != 1
    assert r["EDH"] != 1


def test_none_input_returns_failed():
    r = parse_answer(None)
    assert r["any_hem"] is None
    assert r["parsed"] is False
    assert r["method"] == "failed"


def test_empty_string_returns_failed():
    r = parse_answer("")
    assert r["any_hem"] is None
    assert r["parsed"] is False
