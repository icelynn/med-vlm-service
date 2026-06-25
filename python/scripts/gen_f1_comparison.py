# -*- coding: utf-8 -*-
"""Generator for docs/media/f1_comparison.svg -- the headline any-hemorrhage
F1 chart. Reads the same summary JSON files the README's comparison table
cites; no new inference runs. Rewritten as a script (was hand-authored SVG)
once a 6th bar (MedGemma + Image-Retrieval) needed adding -- computing bar
geometry by hand for N categories is error-prone, a generator isn't.

Usage: ./.venv/Scripts/python.exe python/scripts/gen_f1_comparison.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# (label, json path, color, is_winner, p_value_annotation or None)
GREEN, LIGHT_GREEN, PURPLE, GRAY, BLUE, MAGENTA, ORANGE, LIGHT_PURPLE = (
    "#16A34A", "#86EFAC", "#A78BFA", "#9CA3AF", "#60A5FA", "#C026D3",
    "#F97316", "#D8B4FE",
)
CT_ICH_BARS = [
    ("No-RAG", "data/ct_ich/baseline_summary.json", GRAY, False, None),
    ("Text-RAG", "data/ct_ich/summary_rag.json", BLUE, False, None),
    ("MedGemma", "data/ct_ich/summary_medgemma.json", PURPLE, False, None),
    ("Image-\nRetrieval", "data/ct_ich/summary_imageretrieval.json", GREEN, True, "p&lt;0.0001"),
    ("Random\nControl", "data/ct_ich/summary_imageretrieval_random.json", LIGHT_GREEN, False, None),
    ("MedGemma+\nImg-Retr.", "data/ct_ich/summary_medgemma_imageretrieval.json", MAGENTA, True, "p&lt;0.0001"),
    ("MedGemma+\nRandom", "data/ct_ich/summary_medgemma_random.json", LIGHT_PURPLE, False, None),
    ("Qwen+\nContrast.", "data/ct_ich/summary_qwen_contrastive.json", ORANGE, False, None),
    ("MedGemma+\nContrast.", "data/ct_ich/summary_medgemma_contrastive.json", "#E879F9", False, None),
]
RSNA_BARS = [
    ("No-RAG", "data/rsna/summary.json", GRAY, False, None),
    ("Text-RAG", "data/rsna/summary_rag.json", BLUE, False, None),
    ("MedGemma", "data/rsna/summary_medgemma.json", PURPLE, False, None),
    ("Image-\nRetrieval", "data/rsna/summary_imageretrieval_128matched.json", GREEN, True, "p&lt;0.0001"),
    ("Random\nControl", "data/rsna/summary_imageretrieval_128matched_random.json", LIGHT_GREEN, False, None),
    ("MedGemma+\nImg-Retr.", "data/rsna/summary_medgemma_imageretrieval.json", MAGENTA, True, "p&lt;0.0001"),
    ("MedGemma+\nRandom", "data/rsna/summary_medgemma_random.json", LIGHT_PURPLE, False, None),
    ("Qwen+\nContrast.", "data/rsna/summary_qwen_contrastive.json", ORANGE, False, None),
    ("MedGemma+\nContrast.", "data/rsna/summary_medgemma_contrastive.json", "#E879F9", False, None),
]

# Layout constants
CHART_TOP = 100      # y pixel for F1=Y_MAX (top gridline)
CHART_BOTTOM = 380    # y pixel for F1=0.0
Y_MAX = 1.0
BAR_W = 38
BAR_GAP = 12
PANEL_GAP = 40
LEFT_MARGIN = 50


def y_of(f1):
    return CHART_BOTTOM - (f1 / Y_MAX) * (CHART_BOTTOM - CHART_TOP)


def load_bar(rel_path):
    d = json.load(open(PROJECT_ROOT / rel_path, encoding="utf-8"))
    ah = d["any_hemorrhage"]
    return ah["f1"], ah["f1_ci95"][0], ah["f1_ci95"][1]


def render_panel(bars, x0, title_x):
    svg = []
    svg.append(f'<text x="{title_x}" y="70" text-anchor="middle" font-size="15" '
               f'font-weight="700" fill="#111827">{"CT-ICH" if "ct_ich" in bars[0][1] else "RSNA"}</text>')
    x = x0
    centers = []
    for label, rel_path, color, is_winner, pval in bars:
        f1, lo, hi = load_bar(rel_path)
        cx = x + BAR_W / 2
        centers.append(cx)
        y_top = y_of(f1)
        bar_h = CHART_BOTTOM - y_top
        if f1 == 0.0:
            svg.append(f'<text x="{cx}" y="{CHART_BOTTOM - 8}" text-anchor="middle" '
                       f'font-size="12" fill="#374151">0.000</text>')
        else:
            dash = ' stroke-dasharray="4,3"' if "Random" in label else ""
            stroke = f' stroke="{GREEN}" stroke-width="1.5"' if "Random" in label else ""
            svg.append(f'<rect x="{x}" y="{y_top:.1f}" width="{BAR_W}" height="{bar_h:.1f}" '
                       f'fill="{color}"{stroke}{dash} rx="3"/>')
            y_lo, y_hi = y_of(lo), y_of(hi)
            svg.append(f'<line x1="{cx}" y1="{y_lo:.1f}" x2="{cx}" y2="{y_hi:.1f}" '
                       f'stroke="#4B5563" stroke-width="1.5"/>')
            svg.append(f'<line x1="{cx - 6}" y1="{y_lo:.1f}" x2="{cx + 6}" y2="{y_lo:.1f}" '
                       f'stroke="#4B5563" stroke-width="1.5"/>')
            svg.append(f'<line x1="{cx - 6}" y1="{y_hi:.1f}" x2="{cx + 6}" y2="{y_hi:.1f}" '
                       f'stroke="#4B5563" stroke-width="1.5"/>')
            label_y = y_hi - 8
            weight = ' font-weight="700"' if is_winner else ""
            fill = "#111827" if is_winner else "#374151"
            svg.append(f'<text x="{cx}" y="{label_y:.1f}" text-anchor="middle" font-size="12"'
                       f'{weight} fill="{fill}">{f1:.3f}</text>')
            if pval:
                svg.append(f'<text x="{cx}" y="{label_y - 16:.1f}" text-anchor="middle" '
                           f'font-size="11" font-weight="700" fill="{color}">{pval}</text>')
        x += BAR_W + BAR_GAP
    # category labels (2-line support via \n)
    for (label, *_), cx in zip(bars, centers):
        lines = label.split("\n")
        for i, line in enumerate(lines):
            svg.append(f'<text x="{cx}" y="{CHART_BOTTOM + 18 + i * 15}" text-anchor="middle" '
                       f'font-size="12" fill="#374151">{line}</text>')
    panel_width = x - BAR_GAP - x0
    return "\n  ".join(svg), panel_width


def main():
    ct_ich_svg, ct_ich_w = render_panel(CT_ICH_BARS, LEFT_MARGIN, LEFT_MARGIN + 0)
    rsna_x0 = LEFT_MARGIN + ct_ich_w + PANEL_GAP
    # Need title_x centered per panel -- recompute after knowing width
    ct_ich_title_x = LEFT_MARGIN + ct_ich_w / 2
    rsna_svg, rsna_w = render_panel(RSNA_BARS, rsna_x0, rsna_x0 + 0)
    rsna_title_x = rsna_x0 + rsna_w / 2

    # Re-render with correct title x (cheap: just two panels)
    ct_ich_svg, _ = render_panel(CT_ICH_BARS, LEFT_MARGIN, ct_ich_title_x)
    rsna_svg, _ = render_panel(RSNA_BARS, rsna_x0, rsna_title_x)

    width = rsna_x0 + rsna_w + 30
    height = CHART_BOTTOM + 70

    gridlines = []
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_of(frac)
        gridlines.append(f'<line x1="{LEFT_MARGIN}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" '
                         f'stroke="{"#9CA3AF" if frac == 0 else "#E5E7EB"}" stroke-width="1"/>')
        gridlines.append(f'<text x="{LEFT_MARGIN - 5}" y="{y + 4:.1f}" text-anchor="end" '
                         f'font-size="12" fill="#6B7280">{frac:.1f}</text>')

    divider_x = rsna_x0 - PANEL_GAP / 2
    svg = f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Helvetica, Arial, sans-serif">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#FFFFFF" stroke="#E5E7EB" stroke-width="1"/>
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">Any-Hemorrhage F1 by Method &#215; Dataset</text>
  <text x="{width / 2}" y="47" text-anchor="middle" font-size="12" fill="#6B7280">bars = point estimate &#183; whiskers = 95% bootstrap CI (2000 resamples)</text>
  {chr(10).join(gridlines)}
  <line x1="{divider_x:.1f}" y1="{CHART_TOP}" x2="{divider_x:.1f}" y2="{CHART_BOTTOM}" stroke="#E5E7EB" stroke-width="1.5" stroke-dasharray="5,4"/>
  <text x="20" y="{(CHART_TOP + CHART_BOTTOM) / 2:.0f}" text-anchor="middle" font-size="13" fill="#374151" transform="rotate(-90 20 {(CHART_TOP + CHART_BOTTOM) / 2:.0f})">Any-hemorrhage F1</text>
  {ct_ich_svg}
  {rsna_svg}
  <text x="{width / 2}" y="{height - 14}" text-anchor="middle" font-size="11" fill="#9CA3AF">Two-stage findings-conditioned Text-RAG (p=1.0 vs Text-RAG, both datasets) omitted &#8212; identical to Text-RAG. Full numbers &amp; reproduction commands in README.</text>
</svg>
'''
    out_path = PROJECT_ROOT / "docs" / "media" / "f1_comparison.svg"
    out_path.write_text(svg, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
