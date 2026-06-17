# -*- coding: utf-8 -*-
"""
Answer parser — model free text -> 6-dim hemorrhage label vector.
=================================================================

This replaces CheXbert (the chest-X-ray report labeler). Its job: take whatever
Qwen3-VL writes for one head-CT slice and turn it into structured labels that
metrics.py can score.

Core challenge ② (see v3 blueprint §9): a general VLM may follow the format,
describe findings without naming subtypes, hedge ("possible SAH"), or refuse.
So this parser:
  1. tries the STRUCTURED path first (the constrained prompt asks for two fixed
     lines: HEMORRHAGE: ... / SUBTYPES: ...);
  2. falls back to a keyword scan over free text;
  3. reports parse-failure and refusal as first-class flags, so the harness can
     publish a parse-failure rate / refusal rate as a first-class output metric.

Label encoding (matches metrics.py):
    1    = present
    0    = absent
   -1    = uncertain (model hedged)
   None  = not mentioned / could not parse  (metrics.binarize maps None -> 0)

The uncertain (-1) decision is intentionally LEFT to metrics.py's
uncertain_policy — the parser only detects the hedge, scoring decides how to
count it. That keeps the u-positive / u-negative / ignore sensitivity analysis
a one-line change at scoring time, no re-inference.

Run `python python/eval/parse_answer.py` for the self-test.
"""

import re

# Subtype key -> regex of synonyms (abbteviation OR full name). The constrained
# prompt asks for the abbreviations, so those are the primary hits; full names
# cover the free-text fallback.
SUBTYPE_PATTERNS = {
    "IPH": r"\b(IPH|intra[\s-]?parenchymal|intra[\s-]?cerebral)\b",
    "IVH": r"\b(IVH|intra[\s-]?ventricular)\b",
    "SAH": r"\b(SAH|sub[\s-]?arachnoid)\b",
    "EDH": r"\b(EDH|epidural|extra[\s-]?dural)\b",
    "SDH": r"\b(SDH|sub[\s-]?dural)\b",
}
SUBTYPES = list(SUBTYPE_PATTERNS.keys())

# HEMORRHAGE: value -> binary determination.
_POS_WORDS = r"(yes|present|positive|seen|identified|evident)"
_NEG_WORDS = r"(no|none|absent|negative|nil)"
_UNC_WORDS = r"(uncertain|possible|possibly|equivocal|indeterminate|questionable|cannot exclude|cannot rule out)"

# Hedge cues near a subtype mention -> uncertain (-1).
_HEDGE = re.compile(
    r"\b(possible|possibly|probable|likely|suspicious for|concerning for|"
    r"cannot exclude|cannot rule out|may represent|questionable|equivocal|"
    r"could represent|suggestive of)\b", re.I)

# Refusal / deferral cues. Deliberately conservative so that clinical hedging
# ("cannot exclude SAH") is NOT misread as a refusal.
_REFUSAL = re.compile(
    r"(as an ai|i (?:can ?not|cannot|am unable to|'m unable to|am not able to) "
    r"(?:provide|give|make|offer|assist|help|analyze|interpret|diagnos)|"
    r"consult (?:a|your) (?:doctor|physician|radiologist|healthcare)|"
    r"seek medical|not a (?:substitute|medical)|unable to determine)", re.I)


def _hemorrhage_line(text):
    """Return the HEMORRHAGE: line value as 1/0/-1, or None if absent."""
    m = re.search(r"hemorrhage\s*[:\-]\s*(.+)", text, re.I)
    if not m:
        return None
    val = m.group(1).strip().lower()
    if re.match(_UNC_WORDS, val):
        return -1
    if re.match(_POS_WORDS, val):
        return 1
    if re.match(_NEG_WORDS, val):
        return 0
    return None


def _subtypes_line(text):
    """Return the SUBTYPES: line raw value, or None if the line is absent."""
    m = re.search(r"subtypes?\s*[:\-]\s*(.*)", text, re.I)
    return m.group(1).strip() if m else None


def _scan_subtype(text, pattern):
    """Free-text scan for ONE subtype. Returns 1 / 0 / -1 / None.

    Looks at the clause around each mention for negation or hedge cues.
    """
    found = None
    for m in re.finditer(pattern, text, re.I):
        # window: a few words on each side of the mention
        lo, hi = max(0, m.start() - 40), min(len(text), m.end() + 25)
        ctx = text[lo:hi]
        if re.search(r"\b(no|without|negative for|absence of|no evidence of)\b",
                     ctx[:m.start() - lo + 5], re.I):
            found = 0 if found is None else found
            continue
        if _HEDGE.search(ctx):
            return -1
        return 1
    return found


def parse_answer(text):
    """Parse one model output into a structured result.

    Returns a dict:
        any_hem : 1 / 0 / -1 / None
        IPH/IVH/SAH/EDH/SDH : 1 / 0 / -1 / None
        refused : bool   (model declined / deferred)
        parsed  : bool   (we extracted a hemorrhage determination)
        method  : "structured" | "fallback" | "failed"
    """
    out = {k: None for k in SUBTYPES}
    out.update(any_hem=None, refused=False, parsed=False, method="failed")
    if text is None:
        return out
    text = text.strip()

    refused = bool(_REFUSAL.search(text))

    # ---- 1. structured path: HEMORRHAGE: / SUBTYPES: lines --------------
    hem = _hemorrhage_line(text)
    sub_line = _subtypes_line(text)
    if hem is not None:
        out["any_hem"] = hem
        out["method"] = "structured"
        out["parsed"] = True
        # Subtypes: model was asked to list ALL present -> listed = 1,
        # not listed = explicit 0 (only when we trust the structured answer).
        if sub_line is not None:
            for key, pat in SUBTYPE_PATTERNS.items():
                if re.search(pat, sub_line, re.I):
                    out[key] = -1 if _HEDGE.search(sub_line) else 1
                else:
                    out[key] = 0
            if hem == 0:  # explicit "no hemorrhage" -> all subtypes absent
                for key in SUBTYPES:
                    out[key] = 0
        else:
            # No SUBTYPES line but a HEMORRHAGE verdict: scan body for subtypes,
            # default the rest to absent if hem says no.
            for key, pat in SUBTYPE_PATTERNS.items():
                out[key] = _scan_subtype(text, pat)
            if hem == 0:
                out = {**out, **{k: 0 for k in SUBTYPES}}
        out["refused"] = refused
        return out

    # ---- 2. fallback: free-text keyword scan ----------------------------
    hits = {key: _scan_subtype(text, pat) for key, pat in SUBTYPE_PATTERNS.items()}
    any_pos = any(v == 1 for v in hits.values())
    any_unc = any(v == -1 for v in hits.values())
    if any_pos or any_unc:
        out.update(hits)
        out["any_hem"] = 1 if any_pos else -1
        out["method"] = "fallback"
        out["parsed"] = True
        out["refused"] = refused
        return out

    # explicit global negative in prose ("no acute intracranial hemorrhage")?
    if re.search(r"\bno\b[^.]{0,40}\bhemorrhage\b", text, re.I) and not refused:
        out.update({k: 0 for k in SUBTYPES})
        out["any_hem"] = 0
        out["method"] = "fallback"
        out["parsed"] = True
        return out

    # ---- 3. failed (incl. pure refusal) ---------------------------------
    out["refused"] = refused
    return out


# --------------------------------------------------------------------------
# Self-test — synthetic model outputs covering the paths. Do not edit.
# --------------------------------------------------------------------------
def _run_self_test():
    cases = [
        # (text, expected any_hem, expected subtype dict subset, parsed, refused)
        ("HEMORRHAGE: yes\nSUBTYPES: IPH, IVH",
         1, {"IPH": 1, "IVH": 1, "SAH": 0, "EDH": 0, "SDH": 0}, True, False),
        ("HEMORRHAGE: no\nSUBTYPES: none",
         0, {"IPH": 0, "IVH": 0, "SAH": 0, "EDH": 0, "SDH": 0}, True, False),
        ("HEMORRHAGE: yes\nSUBTYPES: possible SAH",
         1, {"SAH": -1, "IPH": 0}, True, False),
        ("There is a hyperdense intraparenchymal hematoma in the right frontal lobe.",
         1, {"IPH": 1}, True, False),
        ("Possible subarachnoid hemorrhage along the left sylvian fissure.",
         -1, {"SAH": -1}, True, False),
        ("No acute intracranial hemorrhage is identified.",
         0, {"IPH": 0, "SAH": 0}, True, False),
        ("As an AI, I cannot provide a medical diagnosis. Please consult a radiologist.",
         None, {}, False, True),
        ("The image shows the brain.",
         None, {}, False, False),
    ]
    ok = True
    for i, (text, exp_any, exp_sub, exp_parsed, exp_refused) in enumerate(cases, 1):
        r = parse_answer(text)
        fails = []
        if r["any_hem"] != exp_any:
            fails.append(f"any_hem={r['any_hem']}!={exp_any}")
        for k, v in exp_sub.items():
            if r[k] != v:
                fails.append(f"{k}={r[k]}!={v}")
        if r["parsed"] != exp_parsed:
            fails.append(f"parsed={r['parsed']}!={exp_parsed}")
        if r["refused"] != exp_refused:
            fails.append(f"refused={r['refused']}!={exp_refused}")
        if fails:
            ok = False
            print(f"  FAIL case {i}: {'; '.join(fails)}\n        text={text!r}")
        else:
            print(f"  ok   case {i}: any_hem={r['any_hem']} method={r['method']}")
    print("\n" + ("ALL TESTS PASSED" if ok else "TESTS FAILED — keep going"))
    return ok


if __name__ == "__main__":
    _run_self_test()
