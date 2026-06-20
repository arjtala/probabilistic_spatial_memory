#!/usr/bin/env python3
"""F3 paper figure: H3-resolution sensitivity aggregated across all 14
v1 street-scale sessions × 3 encoders.

Inputs: every `h3_res_*_<sequence>_<encoder>_s<seed>.json` under the
session-h3 capture dirs from `eval_hyperparam_sweep.sh AXES=h3_res`.
Each session contributes its own per-(encoder, h3_res) Hit @ 5 mean
across 5 seeds; we aggregate those means across sessions into a
per-(encoder, h3_res) corpus-mean ± across-session std.

Three encoder curves on one plot, x-axis = H3 resolution {8..12},
y-axis = exemplar Hit @ 5 (%). Mean across sessions, shaded band =
±1 std across sessions.

Usage:

    python scripts/plot_f3_multi_corpus.py \\
        --out journal/figures/f3_multi_corpus_h3.svg

The companion PDF (for pandoc embed) is rendered from the SVG by
`rsvg-convert` — same pattern as the existing journal figures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# The 14 v1 sessions and their capture dirs.
_V1_SESSIONS = [
    # (corpus, session_id, captures_dir)
    ("LookOut", "Mainquad_jan10", "captures/aria_Mainquad_jan10_h3"),
    ("LookOut", "Sanmateopark_garage_jan11", "captures/aria_Sanmateopark_garage_jan11_h3"),
    ("LookOut", "Fostersquare1_jan16", "captures/aria_Fostersquare1_jan16_h3"),
    ("LookOut", "BurlingameDT5_feb5", "captures/aria_BurlingameDT5_feb5_h3"),
    ("LookOut", "SanmateoDT2_Jan12", "captures/aria_SanmateoDT2_Jan12_h3"),
    ("LookOut", "Gates_to_mainquad_jan10", "captures/aria_Gates_to_mainquad_jan10_h3"),
    ("LookOut", "Huang_Gates_jan10", "captures/aria_Huang_Gates_jan10_h3"),
    ("LookOut", "BurlingameDT4_feb5", "captures/aria_BurlingameDT4_feb5_h3"),
    ("LookOut", "SSC3_jan17_", "captures/aria_SSC3_jan17__h3"),
    ("LookOut", "Hillsdale6_jan14", "captures/aria_Hillsdale6_jan14_h3"),
    ("SLOPER4D", "seq003_street_002", "captures/sloper4d_seq003_street_002_h3"),
    ("SLOPER4D", "seq008_running_001", "captures/sloper4d_seq008_running_001_h3"),
    ("SLOPER4D", "seq009_running_002", "captures/sloper4d_seq009_h3"),
    ("Nymeria", "20230608_s0_shelby_arroyo_act0_3ciwl8",
        "captures/sloper4d_20230608_s0_shelby_arroyo_act0_3ciwl8_h3"),
]

_ENCODERS = ("clipL", "bigG", "siglip2L")
_H3_RES = (8, 9, 10, 11, 12)

# Plot dimensions matching journal/figures/codec_tradeoff.svg style.
_W = 720
_H = 420
_MARGIN_L = 80
_MARGIN_R = 140  # extra room for legend
_MARGIN_T = 50
_MARGIN_B = 70
_PLOT_W = _W - _MARGIN_L - _MARGIN_R
_PLOT_H = _H - _MARGIN_T - _MARGIN_B

# Encoder colors (matched to a colorblind-aware palette).
_COLORS = {
    "clipL":    "#1f77b4",  # blue
    "bigG":     "#ff7f0e",  # orange
    "siglip2L": "#2ca02c",  # green
}
_LABELS = {
    "clipL":    "CLIP ViT-L",
    "bigG":     "CLIP ViT-bigG",
    "siglip2L": "SigLIP 2 large",
}


def _load_session_means(captures_dir: Path, sequence: str) -> dict:
    """Read all h3_res_*.json under captures_dir and return
    {encoder: {h3_res: per-seed-mean Hit @ 5}}."""
    rx = re.compile(
        rf"^h3_res_(?P<r>\d+)_{re.escape(sequence)}_(?P<enc>[A-Za-z][A-Za-z0-9]*)_s(?P<seed>\d+)\.json$"
    )
    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for p in sorted(captures_dir.glob("h3_res_*.json")):
        m = rx.match(p.name)
        if not m:
            continue
        rate = json.loads(p.read_text()).get("summary", {}).get("exemplar_hit_rate_at_5")
        if rate is None:
            continue
        by_cell[(m["enc"], int(m["r"]))].append(rate)
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for (enc, r), rates in by_cell.items():
        if rates:
            out[enc][r] = sum(rates) / len(rates)
    return dict(out)


def _aggregate(repo_root: Path) -> dict:
    """Aggregate per-session per-encoder Hit@5 means into corpus-level
    mean ± across-session std at each H3 resolution.

    Returns: {encoder: {h3_res: (mean_across_sessions, std_across_sessions, n_sessions)}}
    """
    per_session = []
    for corpus, sequence, caps in _V1_SESSIONS:
        d = repo_root / caps
        if not d.is_dir():
            print(f"[f3] WARN: missing {d}", file=sys.stderr)
            continue
        means = _load_session_means(d, sequence)
        per_session.append((corpus, sequence, means))

    out: dict[str, dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for enc in _ENCODERS:
        for r in _H3_RES:
            vals = [s[2].get(enc, {}).get(r) for s in per_session]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float64)
            out[enc][r] = (float(arr.mean()), float(arr.std(ddof=0)), len(arr))
    return dict(out)


def _x_pos(r: int) -> float:
    """Map H3 resolution to x-pixel."""
    return _MARGIN_L + (r - _H3_RES[0]) / (_H3_RES[-1] - _H3_RES[0]) * _PLOT_W


def _y_pos(rate: float, y_max: float) -> float:
    """Map Hit @ 5 in [0, y_max] to y-pixel (SVG y is top-down)."""
    return _MARGIN_T + _PLOT_H - (rate / y_max) * _PLOT_H


def _render_svg(agg: dict) -> str:
    # Auto-scale y axis to the data + small headroom.
    all_means = [m for enc_d in agg.values() for (m, _, _) in enc_d.values()]
    all_stds = [s for enc_d in agg.values() for (_, s, _) in enc_d.values()]
    y_max_raw = max(m + s for m, s in zip(all_means, all_stds)) if all_means else 0.5
    # Round up to nearest 5%.
    y_max = max(0.10, round(y_max_raw * 100 / 5 + 0.5) * 5 / 100)

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )

    # Title.
    lines.append(
        f'<text x="{_W/2:.0f}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="bold">H3 resolution sensitivity (mean ±1σ across 14 sessions)</text>'
    )

    # Axes.
    lines.append(
        f'<line x1="{_MARGIN_L}" y1="{_MARGIN_T + _PLOT_H}" '
        f'x2="{_MARGIN_L + _PLOT_W}" y2="{_MARGIN_T + _PLOT_H}" '
        f'stroke="black" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{_MARGIN_L}" y1="{_MARGIN_T}" '
        f'x2="{_MARGIN_L}" y2="{_MARGIN_T + _PLOT_H}" '
        f'stroke="black" stroke-width="1"/>'
    )

    # Y-axis ticks every 5%.
    n_ticks = int(y_max * 100 / 5)
    for i in range(n_ticks + 1):
        pct = i * 5 / 100
        y = _y_pos(pct, y_max)
        lines.append(
            f'<line x1="{_MARGIN_L - 5}" y1="{y:.1f}" x2="{_MARGIN_L}" y2="{y:.1f}" '
            f'stroke="black" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{_MARGIN_L - 10}" y="{y + 4:.1f}" text-anchor="end">'
            f'{int(pct*100)}%</text>'
        )
        # Light gridline across plot
        lines.append(
            f'<line x1="{_MARGIN_L}" y1="{y:.1f}" x2="{_MARGIN_L + _PLOT_W}" y2="{y:.1f}" '
            f'stroke="#dddddd" stroke-width="1" stroke-dasharray="2,2"/>'
        )

    # X-axis ticks at each H3 resolution.
    for r in _H3_RES:
        x = _x_pos(r)
        lines.append(
            f'<line x1="{x:.1f}" y1="{_MARGIN_T + _PLOT_H}" '
            f'x2="{x:.1f}" y2="{_MARGIN_T + _PLOT_H + 5}" '
            f'stroke="black" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{_MARGIN_T + _PLOT_H + 20}" text-anchor="middle">'
            f'r{r}</text>'
        )
    lines.append(
        f'<text x="{_MARGIN_L + _PLOT_W/2:.0f}" y="{_MARGIN_T + _PLOT_H + 50}" '
        f'text-anchor="middle" font-size="13">H3 resolution (finer →)</text>'
    )
    lines.append(
        f'<text x="20" y="{_MARGIN_T + _PLOT_H/2:.0f}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 20 {_MARGIN_T + _PLOT_H/2:.0f})">exemplar Hit @ 5 (%)</text>'
    )

    # Per-encoder curve + shaded band.
    for enc in _ENCODERS:
        if enc not in agg:
            continue
        color = _COLORS[enc]
        # Build (x, mean, std) sorted by H3 res
        pts = sorted(
            (r, m, s) for r, (m, s, _) in agg[enc].items()
        )
        if not pts:
            continue
        # Shaded ±1σ band as a single polygon: upper edge L→R then lower edge R→L.
        upper = [(_x_pos(r), _y_pos(min(y_max, m + s), y_max)) for r, m, s in pts]
        lower = [(_x_pos(r), _y_pos(max(0.0, m - s), y_max)) for r, m, s in pts]
        poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + list(reversed(lower)))
        lines.append(
            f'<polygon points="{poly_pts}" fill="{color}" fill-opacity="0.15" stroke="none"/>'
        )
        # Mean line.
        path = "M " + " L ".join(
            f"{_x_pos(r):.1f} {_y_pos(m, y_max):.1f}" for r, m, _ in pts
        )
        lines.append(
            f'<path d="{path}" stroke="{color}" stroke-width="2" fill="none"/>'
        )
        # Markers.
        for r, m, _ in pts:
            lines.append(
                f'<circle cx="{_x_pos(r):.1f}" cy="{_y_pos(m, y_max):.1f}" r="3" '
                f'fill="{color}" stroke="white" stroke-width="1"/>'
            )

    # Legend (right side).
    lgx = _MARGIN_L + _PLOT_W + 20
    lgy = _MARGIN_T + 30
    lines.append(
        f'<text x="{lgx}" y="{lgy - 12}" font-weight="bold">Encoder</text>'
    )
    for i, enc in enumerate(_ENCODERS):
        if enc not in agg:
            continue
        y = lgy + i * 22
        color = _COLORS[enc]
        lines.append(
            f'<line x1="{lgx}" y1="{y}" x2="{lgx + 20}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<circle cx="{lgx + 10}" cy="{y}" r="3" fill="{color}" '
            f'stroke="white" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{lgx + 26}" y="{y + 4}">{_LABELS[enc]}</text>'
        )

    # Caption note (bottom right).
    n_sessions = max(
        (n for enc_d in agg.values() for (_, _, n) in enc_d.values()),
        default=0,
    )
    lines.append(
        f'<text x="{_W - 10}" y="{_H - 12}" text-anchor="end" font-size="10" fill="#666">'
        f'aggregated across {n_sessions} sessions (3 corpora)</text>'
    )

    lines.append('</svg>')
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--repo-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent
            / "journal" / "figures" / "f3_multi_corpus_h3.svg",
    )
    args = ap.parse_args()

    agg = _aggregate(args.repo_root)

    print(f"\nAggregated per-encoder × H3-resolution Hit @ 5:")
    print(f"{'enc':10s}  " + "  ".join(f"r{r:>2d}" for r in _H3_RES))
    print("-" * 60)
    for enc in _ENCODERS:
        if enc not in agg:
            print(f"{enc:10s}  (no data)")
            continue
        cells = []
        for r in _H3_RES:
            if r in agg[enc]:
                m, s, n = agg[enc][r]
                cells.append(f"{m*100:5.1f}±{s*100:4.1f}")
            else:
                cells.append("    —    ")
        print(f"{enc:10s}  " + "  ".join(cells))
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_svg(agg))
    print(f"[f3] wrote {args.out}")
    pdf = args.out.with_suffix(".pdf")
    print(f"[f3] render PDF for paper: rsvg-convert -f pdf -o {pdf} {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
