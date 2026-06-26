# -*- coding: utf-8 -*-
"""Generator for the two headline F1 charts in docs/media/. Reads the same
summary JSON files the README's comparison table cites; no new inference runs.

Two fully separate SVG files (2026-06-26, was one file with a single 9-bar
row per dataset that made every label collide):
  - f1_comparison_main.svg     -- does retrieval fix the perception gap?
  - f1_comparison_ablation.svg -- exemplar-selection ablation (R-8's 12-cell table)

Both charts use the same layout: one bar-group per method category, with the
two models' bars paired side-by-side within each group (color = model, not
method) -- Qwen3-VL = blue, MedGemma = green.

Usage: ./.venv/Scripts/python.exe python/scripts/gen_f1_comparison.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

QWEN_COLOR = "#3B82F6"
MEDGEMMA_COLOR = "#16A34A"

# Each category: (label, [(model_label, json_path, is_winner, pval_or_None), ...])
# The model list has 1 or 2 entries (Qwen3-VL always first / blue, MedGemma
# second / green where it exists).
def main_categories(prefix):
    return [
        ("Zero-shot", [
            ("Qwen3-VL", f"data/{prefix}/" + ("baseline_summary.json" if prefix == "ct_ich" else "summary.json"),
             False, None),
            ("MedGemma", f"data/{prefix}/summary_medgemma.json", False, None),
        ]),
        ("Text-RAG", [
            ("Qwen3-VL", f"data/{prefix}/summary_rag.json", False, None),
            ("MedGemma", f"data/{prefix}/summary_medgemma_rag.json", True, "p&lt;0.0001"),
        ]),
        ("High-sim", [
            ("Qwen3-VL", f"data/{prefix}/summary_imageretrieval.json"
             if prefix == "ct_ich" else f"data/{prefix}/summary_imageretrieval_128matched.json",
             True, "p&lt;0.0001"),
            ("MedGemma", f"data/{prefix}/summary_medgemma_imageretrieval.json", True, "p&lt;0.0001"),
        ]),
    ]


def ablation_categories(prefix):
    ir = ("data/ct_ich/summary_imageretrieval.json" if prefix == "ct_ich"
          else "data/rsna/summary_imageretrieval_128matched.json")
    ir_random = ("data/ct_ich/summary_imageretrieval_random.json" if prefix == "ct_ich"
                 else "data/rsna/summary_imageretrieval_128matched_random.json")
    return [
        ("High-sim", [
            ("Qwen3-VL", ir, True, "p&lt;0.0001"),
            ("MedGemma", f"data/{prefix}/summary_medgemma_imageretrieval.json", True, "p&lt;0.0001"),
        ]),
        ("Random", [
            ("Qwen3-VL", ir_random, False, None),
            ("MedGemma", f"data/{prefix}/summary_medgemma_random.json", False, None),
        ]),
        ("Contrastive", [
            ("Qwen3-VL", f"data/{prefix}/summary_qwen_contrastive.json", False, None),
            ("MedGemma", f"data/{prefix}/summary_medgemma_contrastive.json", False, None),
        ]),
    ]


# Layout constants
TITLE_Y = 26
LEGEND_SWATCH_Y = 38
LEGEND_TEXT_Y = 49
SUBTITLE_Y = 70
DATASET_LABEL_Y = SUBTITLE_Y + 35
CHART_TOP = DATASET_LABEL_Y + 12
CHART_BOTTOM = CHART_TOP + 230
Y_MAX = 1.0
BAR_W = 36
PAIR_GAP = 6
CATEGORY_GAP = 34
PANEL_GAP = 56
LEFT_MARGIN = 50


def y_of(f1):
    return CHART_BOTTOM - (f1 / Y_MAX) * (CHART_BOTTOM - CHART_TOP)


def load_bar(rel_path):
    d = json.load(open(PROJECT_ROOT / rel_path, encoding="utf-8"))
    ah = d["any_hemorrhage"]
    return ah["f1"], ah["f1_ci95"][0], ah["f1_ci95"][1]


def render_bar(x, color, f1, lo, hi, is_winner, pval):
    svg = []
    cx = x + BAR_W / 2
    if f1 == 0.0:
        svg.append(f'<text x="{cx}" y="{CHART_BOTTOM - 8}" text-anchor="middle" '
                   f'font-size="11" fill="#374151">0.000</text>')
        return "\n  ".join(svg)
    y_top = y_of(f1)
    bar_h = CHART_BOTTOM - y_top
    svg.append(f'<rect x="{x}" y="{y_top:.1f}" width="{BAR_W}" height="{bar_h:.1f}" '
               f'fill="{color}" rx="3"/>')
    y_lo, y_hi = y_of(lo), y_of(hi)
    svg.append(f'<line x1="{cx}" y1="{y_lo:.1f}" x2="{cx}" y2="{y_hi:.1f}" '
               f'stroke="#1F2937" stroke-width="1.3"/>')
    svg.append(f'<line x1="{cx - 5}" y1="{y_lo:.1f}" x2="{cx + 5}" y2="{y_lo:.1f}" '
               f'stroke="#1F2937" stroke-width="1.3"/>')
    svg.append(f'<line x1="{cx - 5}" y1="{y_hi:.1f}" x2="{cx + 5}" y2="{y_hi:.1f}" '
               f'stroke="#1F2937" stroke-width="1.3"/>')
    label_y = y_hi - 7
    weight = ' font-weight="700"' if is_winner else ""
    svg.append(f'<text x="{cx}" y="{label_y:.1f}" text-anchor="middle" font-size="11"'
               f'{weight} fill="#111827">{f1:.3f}</text>')
    if pval:
        svg.append(f'<text x="{cx}" y="{label_y - 14:.1f}" text-anchor="middle" '
                   f'font-size="10" font-weight="700" fill="{color}">{pval}</text>')
    return "\n  ".join(svg)


def render_dataset_panel(categories, x0):
    svg = []
    x = x0
    cat_centers = []
    for label, models in categories:
        group_x0 = x
        for model_label, path, is_winner, pval in models:
            f1, lo, hi = load_bar(path)
            color = QWEN_COLOR if model_label == "Qwen3-VL" else MEDGEMMA_COLOR
            svg.append(render_bar(x, color, f1, lo, hi, is_winner, pval))
            x += BAR_W + PAIR_GAP
        group_end = x - PAIR_GAP
        cat_centers.append(((group_x0 + group_end) / 2, label))
        x += CATEGORY_GAP
    for cx, label in cat_centers:
        svg.append(f'<text x="{cx:.1f}" y="{CHART_BOTTOM + 20}" text-anchor="middle" '
                   f'font-size="12" fill="#374151">{label}</text>')
    return "\n  ".join(svg), x - CATEGORY_GAP


def render_gridlines(width):
    out = []
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = y_of(frac)
        out.append(f'<line x1="{LEFT_MARGIN}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}" '
                   f'stroke="{"#9CA3AF" if frac == 0 else "#E5E7EB"}" stroke-width="1"/>')
        out.append(f'<text x="{LEFT_MARGIN - 5}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="12" fill="#6B7280">{frac:.1f}</text>')
    return "\n  ".join(out)


def render_chart(title, subtitle, ct_ich_cats, rsna_cats, out_name, footnote=None):
    ct_svg, ct_end = render_dataset_panel(ct_ich_cats, LEFT_MARGIN)
    rsna_x0 = ct_end + PANEL_GAP
    rsna_svg, rsna_end = render_dataset_panel(rsna_cats, rsna_x0)
    width = rsna_end + 30
    height = CHART_BOTTOM + 130

    ct_title_x = (LEFT_MARGIN + ct_end) / 2
    rsna_title_x = (rsna_x0 + rsna_end) / 2
    divider_x = rsna_x0 - PANEL_GAP / 2
    gridlines = render_gridlines(width)
    ylabel_y = (CHART_TOP + CHART_BOTTOM) / 2
    if footnote:
        lines = [l for l in footnote.split("\n") if l.strip()]
        if len(lines) == 1:
            footnote_svg = (f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" '
                            f'font-size="11" fill="#9CA3AF">{lines[0]}</text>')
        else:
            tspans = []
            for i, line in enumerate(lines):
                dy = "0" if i == 0 else "14"
                tspans.append(f'<tspan x="{width / 2}" dy="{dy}">{line}</tspan>')
            footnote_svg = (f'<text x="{width / 2}" y="{height - 28}" text-anchor="middle" '
                            f'font-size="11" fill="#9CA3AF">{"".join(tspans)}</text>')
    else:
        footnote_svg = ""

    svg = f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Helvetica, Arial, sans-serif">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#FFFFFF" stroke="#E5E7EB" stroke-width="1"/>
  <text x="{width / 2}" y="{TITLE_Y}" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{title}</text>
  <rect x="{width / 2 - 130}" y="{LEGEND_SWATCH_Y}" width="14" height="14" rx="3" fill="{QWEN_COLOR}"/>
  <text x="{width / 2 - 112}" y="{LEGEND_TEXT_Y}" font-size="12" fill="#374151">Qwen3-VL-4B</text>
  <rect x="{width / 2 - 10}" y="{LEGEND_SWATCH_Y}" width="14" height="14" rx="3" fill="{MEDGEMMA_COLOR}"/>
  <text x="{width / 2 + 8}" y="{LEGEND_TEXT_Y}" font-size="12" fill="#374151">MedGemma-4B</text>
  <text x="{width / 2}" y="{SUBTITLE_Y}" text-anchor="middle" font-size="12" fill="#6B7280">{subtitle}</text>
  {gridlines}
  <text x="20" y="{ylabel_y:.0f}" text-anchor="middle" font-size="13" fill="#374151" transform="rotate(-90 20 {ylabel_y:.0f})">Any-hemorrhage F1</text>
  <text x="{ct_title_x:.1f}" y="{DATASET_LABEL_Y}" text-anchor="middle" font-size="14" font-weight="700" fill="#111827">CT-ICH</text>
  <text x="{rsna_title_x:.1f}" y="{DATASET_LABEL_Y}" text-anchor="middle" font-size="14" font-weight="700" fill="#111827">RSNA</text>
  <line x1="{divider_x:.1f}" y1="{CHART_TOP}" x2="{divider_x:.1f}" y2="{CHART_BOTTOM}" stroke="#E5E7EB" stroke-width="1.5" stroke-dasharray="5,4"/>
  {ct_svg}
  {rsna_svg}
  {footnote_svg}
</svg>
'''
    out_path = PROJECT_ROOT / "docs" / "media" / out_name
    out_path.write_text(svg, encoding="utf-8")
    print(f"wrote {out_path}  ({width:.0f}x{height:.0f})")


def main():
    render_chart(
        "Does retrieval fix the perception gap?",
        "bars = point estimate · whiskers = 95% bootstrap CI (2000 resamples)",
        main_categories("ct_ich"), main_categories("rsna"),
        "f1_comparison_main.svg",
        footnote="MedGemma+Text-RAG (p&lt;0.0001 vs. MedGemma zero-shot, both datasets) \ntext knowledge helps a model with some baseline perception, unlike Qwen3-VL's floor. See Honest limitations.",
    )
    render_chart(
        "Exemplar-selection ablation",
        "High-sim vs. Random vs. Contrastive ICL · 95% bootstrap CI (2000 resamples)",
        ablation_categories("ct_ich"), ablation_categories("rsna"),
        "f1_comparison_ablation.svg",
        footnote="High-sim repeated from the chart above as the shared reference point for this ablation.",
    )


if __name__ == "__main__":
    main()
