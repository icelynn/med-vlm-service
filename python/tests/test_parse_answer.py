# -*- coding: utf-8 -*-
"""Tests for parse_answer.py — structured + fallback + edge cases.

parse_answer.py already has an inline self-test (run via `python
parse_answer.py`), but it can't be collected alongside the other tests in
python/tests/. This file re-expresses those cases as named functions (so
pytest picks them up if installed) and adds edge cases not covered by the
original self-test: case-insensitive HEMORRHAGE line, full synonym names for
SDH/EDH, the 'uncertain' keyword on the HEMORRHAGE line, and None/empty input.

Run:  python -m pytest python/tests/test_parse_answer.py -v   (if pytest installed)
  or:  python python/tests/test_parse_answer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from parse_answer import parse_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_parse(text, exp_any, exp_sub, exp_parsed, exp_refused):
    r = parse_answer(text)
    assert r["any_hem"] == exp_any, f"any_hem: got {r['any_hem']!r}, want {exp_any!r}  text={text!r}"
    for k, v in exp_sub.items():
        assert r[k] == v, f"{k}: got {r[k]!r}, want {v!r}  text={text!r}"
    assert r["parsed"] == exp_parsed, f"parsed: got {r['parsed']}, want {exp_parsed}  text={text!r}"
    assert r["refused"] == exp_refused, f"refused: got {r['refused']}, want {exp_refused}  text={text!r}"


# ---------------------------------------------------------------------------
# Core cases (ported from parse_answer.py's _run_self_test)
# ---------------------------------------------------------------------------

def test_structured_positive_two_subtypes():
    _assert_parse(
        "HEMORRHAGE: yes\nSUBTYPES: IPH, IVH",
        1, {"IPH": 1, "IVH": 1, "SAH": 0, "EDH": 0, "SDH": 0}, True, False,
    )


def test_structured_explicit_negative():
    _assert_parse(
        "HEMORRHAGE: no\nSUBTYPES: none",
        0, {"IPH": 0, "IVH": 0, "SAH": 0, "EDH": 0, "SDH": 0}, True, False,
    )


def test_structured_uncertain_subtype():
    _assert_parse(
        "HEMORRHAGE: yes\nSUBTYPES: possible SAH",
        1, {"SAH": -1, "IPH": 0}, True, False,
    )


def test_fallback_free_text_iph():
    _assert_parse(
        "There is a hyperdense intraparenchymal hematoma in the right frontal lobe.",
        1, {"IPH": 1}, True, False,
    )


def test_fallback_hedged_sah():
    _assert_parse(
        "Possible subarachnoid hemorrhage along the left sylvian fissure.",
        -1, {"SAH": -1}, True, False,
    )


def test_fallback_prose_negative():
    _assert_parse(
        "No acute intracranial hemorrhage is identified.",
        0, {"IPH": 0, "SAH": 0}, True, False,
    )


def test_refusal_detected():
    _assert_parse(
        "As an AI, I cannot provide a medical diagnosis. Please consult a radiologist.",
        None, {}, False, True,
    )


def test_unparseable_returns_failed():
    _assert_parse("The image shows the brain.", None, {}, False, False)


# ---------------------------------------------------------------------------
# Edge cases not in the original self-test
# ---------------------------------------------------------------------------

def test_hemorrhage_line_all_caps():
    """HEMORRHAGE: YES (all-caps value) must parse as positive."""
    r = parse_answer("HEMORRHAGE: YES\nSUBTYPES: none")
    assert r["any_hem"] == 1
    assert r["parsed"] is True


def test_hemorrhage_line_mixed_case():
    """Hemorrhage: No (mixed-case key and value) must parse as negative."""
    r = parse_answer("Hemorrhage: No\nSubtypes: none")
    assert r["any_hem"] == 0
    assert r["parsed"] is True


def test_hemorrhage_uncertain_keyword():
    """HEMORRHAGE: uncertain must return any_hem=-1."""
    r = parse_answer("HEMORRHAGE: uncertain\nSUBTYPES: none")
    assert r["any_hem"] == -1
    assert r["parsed"] is True


def test_sdh_full_name_detected():
    """'subdural' (full name) must be detected as SDH in free text."""
    r = parse_answer("There is a small subdural hematoma along the left convexity.")
    assert r["SDH"] == 1
    assert r["any_hem"] == 1


def test_edh_full_name_detected():
    """'epidural' (full name) must be detected as EDH in free text."""
    r = parse_answer("A hyperdense epidural collection is present in the right temporal region.")
    assert r["EDH"] == 1
    assert r["any_hem"] == 1


def test_edh_extradural_synonym():
    """'extra-dural' synonym must be recognized as EDH."""
    r = parse_answer("Small extra-dural hematoma identified.")
    assert r["EDH"] == 1


def test_negated_subtype_not_flagged():
    """'no subdural hematoma' must not flag SDH as present."""
    r = parse_answer("No subdural hematoma is seen. No epidural collection.")
    assert r["SDH"] != 1
    assert r["EDH"] != 1


def test_none_input_returns_failed():
    """None input must not raise and must return method='failed'."""
    r = parse_answer(None)
    assert r["any_hem"] is None
    assert r["parsed"] is False
    assert r["method"] == "failed"


def test_empty_string_returns_failed():
    r = parse_answer("")
    assert r["any_hem"] is None
    assert r["parsed"] is False


# ---------------------------------------------------------------------------
# Standalone runner (matches style of other tests in this directory)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_structured_positive_two_subtypes,
        test_structured_explicit_negative,
        test_structured_uncertain_subtype,
        test_fallback_free_text_iph,
        test_fallback_hedged_sah,
        test_fallback_prose_negative,
        test_refusal_detected,
        test_unparseable_returns_failed,
        test_hemorrhage_line_all_caps,
        test_hemorrhage_line_mixed_case,
        test_hemorrhage_uncertain_keyword,
        test_sdh_full_name_detected,
        test_edh_full_name_detected,
        test_edh_extradural_synonym,
        test_negated_subtype_not_flagged,
        test_none_input_returns_failed,
        test_empty_string_returns_failed,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
