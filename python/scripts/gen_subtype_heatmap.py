# -*- coding: utf-8 -*-
"""One-off generator for docs/media/per_subtype_heatmap.svg (per-subtype F1
heatmap, method x subtype). Reads the same summary
JSON files the README's comparison table already cites -- no new GPU runs,
no new numbers, just a different rendering of existing per_subtype F1s.

Usage: ./.venv/Scripts/python.exe python/scripts/gen_subtype_heatmap.py
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUBTYPES = ["IPH", "IVH", "SAH", "EDH", "SDH"]

# (label, json path) -- same canonical files the README table/compare_runs.py use.
CT_ICH_ROWS = [
    ("No-RAG", "data/ct_ich/baseline_summary.json"),
    ("Text-RAG", "data/ct_ich/summary_rag.json"),
    ("MedGemma", "data/ct_ich/summary_medgemma.json"),
    ("Image-Retrieval", "data/ct_ich/summary_imageretrieval.json"),
    ("Random Control", "data/ct_ich/summary_imageretrieval_random.json"),
]
RSNA_ROWS = [
    ("No-RAG", "data/rsna/summary.json"),
    ("Text-RAG", "data/rsna/summary_rag.json"),
    ("MedGemma", "data/rsna/summary_medgemma.json"),
    ("Image-Retrieval", "data/rsna/summary_imageretrieval_128matched.json"),
    ("Random Control", "data/rsna/summary_imageretrieval_random.json"),
]


def load_panel(rows):
    panel = []
    for label, rel_path in rows:
        d = json.load(open(PROJECT_ROOT / rel_path, encoding="utf-8"))
        f1s = [d["per_subtype"][k]["f1"] for k in SUBTYPES]
        supports = [d["per_subtype"][k]["support"] for k in SUBTYPES]
        panel.append((label, f1s, supports))
    return panel


def cell_color(f1, vmax=0.6):
    """White (0) -> green (vmax), same green family as f1_comparison.svg's
    Image-Retrieval bars (#16A34A), clamped so a single outlier doesn't wash
    out the rest of the scale."""
    t = max(0.0, min(1.0, f1 / vmax))
    r = round(255 + t * (0x16 - 255))
    g = round(255 + t * (0xA3 - 255))
    b = round(255 + t * (0x4A - 255))
    return f"#{r:02X}{g:02X}{b:02X}", t


def text_color(t):
    return "#FFFFFF" if t > 0.55 else "#111827"


def render_panel(panel, x0, y0, cell_w, cell_h, title):
    svg = []
    n_rows = len(panel)
    n_cols = len(SUBTYPES)
    svg.append(f'<text x="{x0 + n_cols * cell_w / 2}" y="{y0 - 14}" '
               f'text-anchor="middle" font-size="15" font-weight="700" fill="#111827">{title}</text>')
    # column headers
    for j, sub in enumerate(SUBTYPES):
        cx = x0 + j * cell_w + cell_w / 2
        svg.append(f'<text x="{cx}" y="{y0 - 2}" text-anchor="middle" '
                   f'font-size="12" font-weight="700" fill="#374151">{sub}</text>')
    # rows
    for i, (label, f1s, supports) in enumerate(panel):
        ry = y0 + i * cell_h
        svg.append(f'<text x="{x0 - 8}" y="{ry + cell_h / 2 + 4}" text-anchor="end" '
                   f'font-size="12" fill="#374151">{label}</text>')
        for j, (f1, support) in enumerate(zip(f1s, supports)):
            cx = x0 + j * cell_w
            color, t = cell_color(f1)
            tc = text_color(t)
            svg.append(f'<rect x="{cx}" y="{ry}" width="{cell_w - 2}" height="{cell_h - 2}" '
                       f'fill="{color}" stroke="#E5E7EB" stroke-width="1" rx="3"/>')
            svg.append(f'<text x="{cx + (cell_w - 2) / 2}" y="{ry + cell_h / 2 - 1}" '
                       f'text-anchor="middle" font-size="13" font-weight="600" fill="{tc}">{f1:.2f}</text>')
            svg.append(f'<text x="{cx + (cell_w - 2) / 2}" y="{ry + cell_h / 2 + 13}" '
                       f'text-anchor="middle" font-size="9" fill="{tc}" opacity="0.85">n={support}</text>')
    return "\n  ".join(svg)


def main():
    ct_ich = load_panel(CT_ICH_ROWS)
    rsna = load_panel(RSNA_ROWS)

    cell_w, cell_h = 78, 56
    left_margin = 130
    panel_gap = 70
    x0_left = left_margin
    x0_right = x0_left + 5 * cell_w + panel_gap
    y0 = 95

    width = x0_right + 5 * cell_w + 40
    height = y0 + 5 * cell_h + 70

    body = []
    body.append(render_panel(ct_ich, x0_left, y0, cell_w, cell_h, "CT-ICH"))
    body.append(render_panel(rsna, x0_right, y0, cell_w, cell_h, "RSNA"))

    svg = f'''<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Helvetica, Arial, sans-serif">
  <rect x="0" y="0" width="{width}" height="{height}" rx="14" fill="#FFFFFF" stroke="#E5E7EB" stroke-width="1"/>
  <text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">Per-Subtype F1 by Method &#215; Dataset</text>
  <text x="{width / 2}" y="47" text-anchor="middle" font-size="12" fill="#6B7280">darker = higher F1 &#183; n = positive-subtype support out of 150 slices</text>
  {body[0]}
  {body[1]}
  <text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-size="11" fill="#9CA3AF">Same summary JSONs as the README comparison table &#8212; no new inference runs. Two-stage Text-RAG (identical to Text-RAG, omitted).</text>
</svg>
'''
    out_path = PROJECT_ROOT / "docs" / "media" / "per_subtype_heatmap.svg"
    out_path.write_text(svg, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
