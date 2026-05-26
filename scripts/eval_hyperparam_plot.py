#!/usr/bin/env python3
"""Generate the F3 hyperparameter-sensitivity figure for the paper.

Reads JSONs written by `scripts/eval_hyperparam_sweep.sh` (filename
pattern: `<axis>_<label>_<sid>_<encoder>_s<seed>.json`) and produces
a 3-panel SVG: one panel per axis (H3 resolution / retention /
exemplars), x-axis is the knob value, y-axis is exemplar Hit @5
pooled across sessions and seeds (mean ± across-seed std). The v2/E11
default value is marked on each panel with a vertical line so the
"operating point" is visually obvious.

This doesn't use matplotlib — hand-authored SVG matches the style of
journal/figures/codec_tradeoff.svg + v1_v2_lift.svg and avoids needing
matplotlib in the conda env.

Usage:

    python scripts/eval_hyperparam_plot.py captures/hyperparam
    python scripts/eval_hyperparam_plot.py captures/hyperparam \\
        --out journal/figures/hyperparam_sensitivity.svg

The companion PDF (for pandoc embed) is rendered from the SVG by
`rsvg-convert` — same pattern as the existing journal figures.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


# Filename:  <axis>_<label>_<sid>_<encoder>_s<seed>.json
# Examples:  h3_res_10_1501677363692556_bigG_s0.json
#            retention_75x12_201703061033_bigG_s3.json
#            exemplars_128_287142033569927_bigG_s4.json
FNAME_RE = re.compile(
    r"^(?P<axis>h3_res|retention|exemplars)_"
    r"(?P<label>[^_]+)_"
    r"(?P<sid>\d+)_"
    r"(?P<encoder>[^_]+)_"
    r"s(?P<seed>\d+)\.json$"
)


def parse_value(axis: str, label: str) -> float:
    """Project a label string onto the x-axis number we plot against."""
    if axis == "h3_res":
        return float(label)
    if axis == "exemplars":
        return float(label)
    if axis == "retention":
        # "75x12" -> retention horizon in seconds.
        tw, cap = label.split("x")
        return float(tw) * float(cap)
    raise ValueError(f"unknown axis {axis!r}")


def axis_label(axis: str) -> str:
    return {
        "h3_res": "H3 resolution",
        "retention": "retention horizon (s = time_window × capacity)",
        "exemplars": "reservoir size (exemplars per tile)",
    }[axis]


def axis_default(axis: str) -> float:
    return {"h3_res": 10.0, "retention": 75.0 * 12, "exemplars": 128.0}[axis]


def load_records(dir_path: Path, encoder: str | None) -> dict:
    """Group by (axis, label, seed) -> Hit @5 from records pooled across sessions.

    Returns nested dict: by_axis[axis][label][seed] = hit_rate (float in 0..1).
    """
    by_axis: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    # Pool scored records across sessions for each (axis, label, seed)
    # cell, then compute Hit @5 once at the end. This matches how the
    # combined-row in eval_aggregate.py works.
    scored_by_cell: dict[tuple[str, str, int], list[dict]] = defaultdict(list)

    for path_str in glob.glob(str(dir_path / "*.json")):
        path = Path(path_str)
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        if encoder is not None and m["encoder"] != encoder:
            continue
        data = json.loads(path.read_text())
        seed = int(m["seed"])
        axis = m["axis"]
        label = m["label"]
        scored = [r for r in data.get("records", []) if r.get("intervals_gt")]
        scored_by_cell[(axis, label, seed)].extend(scored)

    for (axis, label, seed), recs in scored_by_cell.items():
        if not recs:
            continue
        hit = sum(1 for r in recs if r.get("exemplar_hit_at_k")) / len(recs)
        by_axis[axis][label][seed] = hit
    return by_axis


def aggregate_per_label(seed_map: dict[int, float]) -> tuple[float, float, int]:
    """Mean Hit @5 across seeds + sample std + n_seeds."""
    values = list(seed_map.values())
    if not values:
        return (0.0, 0.0, 0)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return (mean, sd, len(values))


# ----------------------------------------------------------------------------
# SVG layout
#
# Three panels stacked vertically in a single SVG file (so the figure
# fits one column in a workshop paper). Each panel is a small line +
# error-bar plot. Hand-authored to match the existing journal/figures
# style (Helvetica, simple grid, no plot library).
# ----------------------------------------------------------------------------

PANEL_HEIGHT = 220
PANEL_WIDTH = 540
PANEL_GAP = 60
LEFT_MARGIN = 70
RIGHT_MARGIN = 40
TOP_PAD = 50  # for title + axis label per panel


def y_to_px(hit_rate: float, panel_top: int) -> float:
    """Map Hit @5 in [0, 1] -> y pixel inside one panel.

    Y axis fixed to [50%, 100%] so codecs that all land near 80% are
    visually distinguishable rather than squished into the top of a
    [0%, 100%] range.
    """
    pct = max(50.0, min(100.0, hit_rate * 100.0))
    plot_h = PANEL_HEIGHT - TOP_PAD - 30  # bottom pad for x-axis labels
    plot_top = panel_top + TOP_PAD
    plot_bottom = panel_top + PANEL_HEIGHT - 30
    return plot_bottom - (pct - 50.0) / 50.0 * (plot_bottom - plot_top)


def x_to_px(value: float, x_lo: float, x_hi: float, log: bool) -> float:
    """Map value -> x pixel. log=True for exemplars + retention axes."""
    plot_left = LEFT_MARGIN
    plot_right = PANEL_WIDTH - RIGHT_MARGIN
    if log:
        import math
        v = math.log10(value)
        v_lo = math.log10(x_lo)
        v_hi = math.log10(x_hi)
    else:
        v, v_lo, v_hi = value, x_lo, x_hi
    if v_hi <= v_lo:
        return plot_left
    return plot_left + (v - v_lo) / (v_hi - v_lo) * (plot_right - plot_left)


def render_panel(
    out: list[str],
    panel_top: int,
    axis: str,
    points: list[tuple[float, float, float, int]],  # (x, mean, std, n_seeds)
    use_log_x: bool,
) -> None:
    """Render one panel into `out` (list of SVG fragment strings)."""
    if not points:
        return
    xs = [p[0] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    # Pad the range so end-point markers don't sit on the axis line.
    if use_log_x:
        import math
        x_lo_p = 10 ** (math.log10(x_lo) - 0.1)
        x_hi_p = 10 ** (math.log10(x_hi) + 0.1)
    else:
        rng = (x_hi - x_lo) or 1.0
        x_lo_p = x_lo - 0.5
        x_hi_p = x_hi + 0.5

    plot_left = LEFT_MARGIN
    plot_right = PANEL_WIDTH - RIGHT_MARGIN
    plot_top = panel_top + TOP_PAD
    plot_bottom = panel_top + PANEL_HEIGHT - 30

    # Panel title
    out.append(
        f'<text x="{(plot_left + plot_right) / 2:.0f}" y="{panel_top + 28}" '
        f'text-anchor="middle" font-size="13" font-weight="600" fill="#1a1a1a">'
        f'{axis_label(axis)}</text>'
    )

    # Y axis
    out.append(
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
        f'y2="{plot_bottom}" stroke="#333" stroke-width="1.2"/>'
    )
    # Y tick labels at 50/60/70/80/90/100
    for pct in (50, 60, 70, 80, 90, 100):
        y = y_to_px(pct / 100.0, panel_top)
        out.append(
            f'<line x1="{plot_left - 4}" y1="{y:.1f}" x2="{plot_left}" '
            f'y2="{y:.1f}" stroke="#666" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{pct}%</text>'
        )
        # Gridline (lighter)
        out.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
            f'y2="{y:.1f}" stroke="#e5e5e5" stroke-width="0.5"/>'
        )
    out.append(
        f'<text x="{plot_left - 50}" y="{(plot_top + plot_bottom) / 2:.0f}" '
        f'text-anchor="middle" font-size="11" fill="#333" '
        f'transform="rotate(-90 {plot_left - 50} '
        f'{(plot_top + plot_bottom) / 2:.0f})">exemplar Hit @5</text>'
    )

    # X axis
    out.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
        f'y2="{plot_bottom}" stroke="#333" stroke-width="1.2"/>'
    )
    # X tick labels at each measured point
    for x_val in xs:
        x = x_to_px(x_val, x_lo_p, x_hi_p, use_log_x)
        out.append(
            f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" '
            f'y2="{plot_bottom + 4}" stroke="#666" stroke-width="1"/>'
        )
        # Format integer values without trailing ".0"
        label_txt = f"{int(x_val)}" if x_val == int(x_val) else f"{x_val:g}"
        out.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 16}" text-anchor="middle" '
            f'font-size="10" fill="#666">{label_txt}</text>'
        )

    # Vertical reference line at the default value
    default_v = axis_default(axis)
    if x_lo_p <= default_v <= x_hi_p:
        x_def = x_to_px(default_v, x_lo_p, x_hi_p, use_log_x)
        out.append(
            f'<line x1="{x_def:.1f}" y1="{plot_top}" x2="{x_def:.1f}" '
            f'y2="{plot_bottom}" stroke="#a3a3a3" stroke-width="1" '
            f'stroke-dasharray="4 3"/>'
        )
        out.append(
            f'<text x="{x_def + 4:.1f}" y="{plot_top + 12}" text-anchor="start" '
            f'font-size="10" fill="#a3a3a3" font-style="italic">v2 default</text>'
        )

    # Connecting line through the means
    line_points = " ".join(
        f"{x_to_px(x, x_lo_p, x_hi_p, use_log_x):.1f},"
        f"{y_to_px(m, panel_top):.1f}"
        for (x, m, _s, _n) in points
    )
    out.append(
        f'<polyline points="{line_points}" fill="none" stroke="#1f4e79" '
        f'stroke-width="1.8" opacity="0.85"/>'
    )

    # Error bars + markers
    for (x_val, mean, sd, _n) in points:
        x = x_to_px(x_val, x_lo_p, x_hi_p, use_log_x)
        y_mid = y_to_px(mean, panel_top)
        y_up = y_to_px(min(1.0, mean + sd), panel_top)
        y_dn = y_to_px(max(0.0, mean - sd), panel_top)
        if sd > 0:
            out.append(
                f'<line x1="{x:.1f}" y1="{y_up:.1f}" x2="{x:.1f}" '
                f'y2="{y_dn:.1f}" stroke="#1f4e79" stroke-width="1.4"/>'
            )
            out.append(
                f'<line x1="{x - 4:.1f}" y1="{y_up:.1f}" x2="{x + 4:.1f}" '
                f'y2="{y_up:.1f}" stroke="#1f4e79" stroke-width="1.4"/>'
            )
            out.append(
                f'<line x1="{x - 4:.1f}" y1="{y_dn:.1f}" x2="{x + 4:.1f}" '
                f'y2="{y_dn:.1f}" stroke="#1f4e79" stroke-width="1.4"/>'
            )
        out.append(
            f'<circle cx="{x:.1f}" cy="{y_mid:.1f}" r="5" fill="#1f4e79"/>'
        )


def render_svg(by_axis: dict, encoder: str) -> str:
    """Render the 3-panel SVG; returns the full XML as a string."""
    axes_order = ["h3_res", "retention", "exemplars"]
    use_log = {"h3_res": False, "retention": True, "exemplars": True}

    total_height = (
        len(axes_order) * PANEL_HEIGHT + (len(axes_order) - 1) * PANEL_GAP + 60
    )
    out: list[str] = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {PANEL_WIDTH} {total_height}" '
        f'font-family="Helvetica, Arial, sans-serif">',
        # Outer title
        f'<text x="{PANEL_WIDTH / 2}" y="28" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="#1a1a1a">'
        f'PSM hyperparameter sensitivity ({encoder})</text>',
        f'<text x="{PANEL_WIDTH / 2}" y="46" text-anchor="middle" '
        f'font-size="11" fill="#555">3 sessions × 5 seeds × 20 questions; '
        f'one axis varied at a time, others held at v2 defaults</text>',
    ]

    panel_top = 60
    for axis in axes_order:
        labels = by_axis.get(axis, {})
        points: list[tuple[float, float, float, int]] = []
        for label, seed_map in labels.items():
            x_val = parse_value(axis, label)
            mean, sd, n = aggregate_per_label(seed_map)
            if n > 0:
                points.append((x_val, mean, sd, n))
        points.sort(key=lambda p: p[0])
        render_panel(out, panel_top, axis, points, use_log[axis])
        panel_top += PANEL_HEIGHT + PANEL_GAP

    out.append("</svg>\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("input_dir", type=Path,
                    help="directory of <axis>_<label>_<sid>_<encoder>_s<seed>.json")
    ap.add_argument("--encoder", default="bigG",
                    help="restrict to one encoder (default bigG)")
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "journal" / "figures" / "hyperparam_sensitivity.svg",
    )
    args = ap.parse_args()

    by_axis = load_records(args.input_dir, args.encoder)
    if not by_axis:
        raise SystemExit(
            f"no matching JSONs found in {args.input_dir} for encoder={args.encoder!r}"
        )

    # Markdown summary to stdout (mirrors the codec_tradeoff style).
    print()
    print(f"## PSM hyperparameter sensitivity ({args.encoder})")
    print("_3 sessions × 5 seeds × 20 questions; one axis varied at a time_")
    print()
    for axis in ("h3_res", "retention", "exemplars"):
        labels = by_axis.get(axis, {})
        if not labels:
            continue
        print(f"### axis: {axis_label(axis)}")
        print()
        print("| value | n seeds | mean Hit @5 | std |")
        print("|---|---|---|---|")
        rows = []
        for label, seed_map in labels.items():
            x_val = parse_value(axis, label)
            mean, sd, n = aggregate_per_label(seed_map)
            rows.append((x_val, label, mean, sd, n))
        rows.sort(key=lambda r: r[0])
        for (_, label, mean, sd, n) in rows:
            print(f"| {label} | {n} | {mean:.1%} | ±{sd:.1%} |")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_svg(by_axis, args.encoder))
    print(f"[plot] wrote {args.out}")
    print(
        f"[plot] render PDF for paper: "
        f"rsvg-convert -f pdf -o {args.out.with_suffix('.pdf')} {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
