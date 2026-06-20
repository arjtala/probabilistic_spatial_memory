#!/usr/bin/env python3
"""F2 paper figure: PSM vs vanilla MLLM retrieval ablation across the
v1 street-scale corpus.

Per session, plots two grouped bars:
  - Vanilla MLLM Hit@5 at K=8 (Gemini 3.1 Pro picks 1 of 8 uniformly-
    sampled frames). Read from captures/mllm_baseline/<sess>_gemini.json
    (LookOut sessions use lookout_<sess>_gemini.json prefix).
  - PSM clipL r12 Hit@5 (mean across 5 seeds). Read from the same
    h3_res_12_*_clipL_s*.json captures the H3 sweep produced.

Both at K=8 / top-5 respectively so the comparison is apples-to-apples
on candidate budget. PSM is shown for clipL only — using the best of
{clipL, bigG, siglip2L} per session would overstate PSM's strength
relative to a fixed-encoder comparison; the cross-encoder result is
the table next door.

Sessions with no MLLM baseline data are simply skipped (the figure
grows as the LookOut MLLM baseline run lands).

Usage:

    python scripts/plot_f2_psm_vs_mllm.py \\
        --out journal/figures/f2_psm_vs_mllm.svg
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Sessions in display order (largest bbox first within each corpus).
# (corpus, sequence_id, h3_captures_dir, mllm_baseline_json)
_V1_SESSIONS = [
    ("LookOut", "Mainquad_jan10",
        "captures/aria_Mainquad_jan10_h3",
        "captures/mllm_baseline/lookout_Mainquad_jan10_gemini.json"),
    ("LookOut", "Sanmateopark_garage_jan11",
        "captures/aria_Sanmateopark_garage_jan11_h3",
        "captures/mllm_baseline/lookout_Sanmateopark_garage_jan11_gemini.json"),
    ("LookOut", "Fostersquare1_jan16",
        "captures/aria_Fostersquare1_jan16_h3",
        "captures/mllm_baseline/lookout_Fostersquare1_jan16_gemini.json"),
    ("LookOut", "BurlingameDT5_feb5",
        "captures/aria_BurlingameDT5_feb5_h3",
        "captures/mllm_baseline/lookout_BurlingameDT5_feb5_gemini.json"),
    ("LookOut", "SanmateoDT2_Jan12",
        "captures/aria_SanmateoDT2_Jan12_h3",
        "captures/mllm_baseline/lookout_SanmateoDT2_Jan12_gemini.json"),
    ("LookOut", "Gates_to_mainquad_jan10",
        "captures/aria_Gates_to_mainquad_jan10_h3",
        "captures/mllm_baseline/lookout_Gates_to_mainquad_jan10_gemini.json"),
    ("LookOut", "Huang_Gates_jan10",
        "captures/aria_Huang_Gates_jan10_h3",
        "captures/mllm_baseline/lookout_Huang_Gates_jan10_gemini.json"),
    ("LookOut", "BurlingameDT4_feb5",
        "captures/aria_BurlingameDT4_feb5_h3",
        "captures/mllm_baseline/lookout_BurlingameDT4_feb5_gemini.json"),
    ("LookOut", "SSC3_jan17_",
        "captures/aria_SSC3_jan17__h3",
        "captures/mllm_baseline/lookout_SSC3_jan17__gemini.json"),
    ("LookOut", "Hillsdale6_jan14",
        "captures/aria_Hillsdale6_jan14_h3",
        "captures/mllm_baseline/lookout_Hillsdale6_jan14_gemini.json"),
    ("SLOPER4D", "seq003_street_002",
        "captures/sloper4d_seq003_street_002_h3",
        "captures/mllm_baseline/seq003_street_002_gemini.json"),
    ("SLOPER4D", "seq008_running_001",
        "captures/sloper4d_seq008_running_001_h3",
        "captures/mllm_baseline/seq008_running_001_gemini.json"),
    ("SLOPER4D", "seq009_running_002",
        "captures/sloper4d_seq009_h3",
        "captures/mllm_baseline/seq009_running_002_gemini.json"),
    ("Nymeria", "20230608_s0_shelby_arroyo_act0_3ciwl8",
        "captures/sloper4d_20230608_s0_shelby_arroyo_act0_3ciwl8_h3",
        "captures/mllm_baseline/nymeria_shelby_gemini.json"),
]


def _psm_clipl_r12_hit5(captures_dir: Path, sequence: str) -> float | None:
    """Mean Hit @5 across 5 seeds of clipL @ H3 r12 for the given session."""
    rx = re.compile(
        rf"^h3_res_12_{re.escape(sequence)}_clipL_s(?P<seed>\d+)\.json$"
    )
    rates = []
    for p in sorted(captures_dir.glob("h3_res_12_*_clipL_s*.json")):
        if not rx.match(p.name):
            continue
        r = json.loads(p.read_text()).get("summary", {}).get("exemplar_hit_rate_at_5")
        if r is not None:
            rates.append(r)
    if not rates:
        return None
    return float(np.mean(rates))


def _vanilla_mllm_hit5(path: Path) -> float | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("summary", {}).get("exemplar_hit_rate_at_5")


def _short_name(corpus: str, sequence: str) -> str:
    if corpus == "Nymeria":
        return "shelby_arroyo_act0"
    # Trim trailing _feb5 / _jan10 etc for LookOut shorter labels.
    s = sequence.rstrip("_")
    for tail in ("_jan10", "_jan11", "_jan12", "_jan14", "_jan16", "_jan17", "_feb5", "_Jan12"):
        if s.endswith(tail):
            s = s[: -len(tail)]
            break
    return s


def _aggregate(repo_root: Path) -> list[dict]:
    rows = []
    for corpus, seq, caps_rel, mllm_rel in _V1_SESSIONS:
        caps = repo_root / caps_rel
        mllm = repo_root / mllm_rel
        psm = _psm_clipl_r12_hit5(caps, seq) if caps.is_dir() else None
        vm = _vanilla_mllm_hit5(mllm)
        rows.append({
            "corpus": corpus,
            "sequence": seq,
            "short": _short_name(corpus, seq),
            "psm": psm,
            "mllm": vm,
        })
    return rows


def _render_svg(rows: list[dict]) -> str:
    # Filter to sessions where BOTH PSM and MLLM exist (apples-to-apples).
    valid = [r for r in rows if r["psm"] is not None and r["mllm"] is not None]
    missing = [r for r in rows if r["psm"] is None or r["mllm"] is None]

    n = len(valid)
    # Figure size + margins.
    W = max(900, 60 * n + 200)
    H = 460
    ML, MR, MT, MB = 80, 200, 50, 110
    PW = W - ML - MR
    PH = H - MT - MB

    # Y range: 0 to nearest 5% above max.
    max_y = max(
        (max(r["psm"], r["mllm"]) for r in valid),
        default=0.5,
    )
    y_max = max(0.10, round(max_y * 100 / 5 + 0.5) * 5 / 100)

    # Bars: per session, two bars side-by-side.
    bar_w = PW / max(1, n) * 0.35
    group_w = bar_w * 2 + 8

    def x_group(i: int) -> float:
        return ML + (i + 0.5) * PW / max(1, n) - group_w / 2

    def y_top(rate: float) -> float:
        return MT + PH - (rate / y_max) * PH

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )

    # Title.
    lines.append(
        f'<text x="{W/2:.0f}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="bold">PSM vs vanilla MLLM (Hit @ 5, K=8 candidates)</text>'
    )

    # Axes.
    lines.append(
        f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}" '
        f'stroke="black" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" '
        f'stroke="black" stroke-width="1"/>'
    )

    # Y ticks every 5%.
    n_ticks = int(y_max * 100 / 5)
    for i in range(n_ticks + 1):
        pct = i * 5 / 100
        y = y_top(pct)
        lines.append(
            f'<line x1="{ML - 5}" y1="{y:.1f}" x2="{ML}" y2="{y:.1f}" '
            f'stroke="black" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{ML - 10}" y="{y + 4:.1f}" text-anchor="end">{int(pct*100)}%</text>'
        )
        lines.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML + PW}" y2="{y:.1f}" '
            f'stroke="#dddddd" stroke-width="1" stroke-dasharray="2,2"/>'
        )

    # Bars + labels.
    psm_color = "#1f77b4"
    mllm_color = "#d62728"
    for i, r in enumerate(valid):
        gx = x_group(i)
        # PSM bar
        psm_h = (r["psm"] / y_max) * PH
        lines.append(
            f'<rect x="{gx:.1f}" y="{y_top(r["psm"]):.1f}" '
            f'width="{bar_w:.1f}" height="{psm_h:.1f}" '
            f'fill="{psm_color}" />'
        )
        # MLLM bar
        lines.append(
            f'<rect x="{gx + bar_w + 4:.1f}" y="{y_top(r["mllm"]):.1f}" '
            f'width="{bar_w:.1f}" height="{(r["mllm"] / y_max) * PH:.1f}" '
            f'fill="{mllm_color}" />'
        )
        # Value labels above bars (small).
        lines.append(
            f'<text x="{gx + bar_w/2:.1f}" y="{y_top(r["psm"]) - 4:.1f}" '
            f'text-anchor="middle" font-size="10" fill="{psm_color}" font-weight="bold">'
            f'{r["psm"]*100:.0f}%</text>'
        )
        lines.append(
            f'<text x="{gx + bar_w + 4 + bar_w/2:.1f}" y="{y_top(r["mllm"]) - 4:.1f}" '
            f'text-anchor="middle" font-size="10" fill="{mllm_color}" font-weight="bold">'
            f'{r["mllm"]*100:.0f}%</text>'
        )
        # Session label (rotated 35° down).
        label_x = gx + group_w / 2
        label_y = MT + PH + 18
        lines.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="end" font-size="10" '
            f'transform="rotate(-35 {label_x:.1f} {label_y:.1f})">'
            f'{r["short"]}</text>'
        )
        # Corpus tag under label.
        lines.append(
            f'<text x="{label_x:.1f}" y="{label_y + 60:.1f}" '
            f'text-anchor="middle" font-size="8" fill="#666">{r["corpus"]}</text>'
        )

    # Legend.
    lgx = ML + PW + 30
    lgy = MT + 30
    lines.append(
        f'<text x="{lgx}" y="{lgy - 12}" font-weight="bold">Method</text>'
    )
    for i, (color, label, desc) in enumerate([
        (psm_color, "PSM", "clipL @ H3 r12, top-5"),
        (mllm_color, "vanilla MLLM", "Gemini 3.1 Pro, K=8"),
    ]):
        y = lgy + i * 36
        lines.append(
            f'<rect x="{lgx}" y="{y - 8}" width="14" height="12" fill="{color}" />'
        )
        lines.append(
            f'<text x="{lgx + 20}" y="{y + 2}">{label}</text>'
        )
        lines.append(
            f'<text x="{lgx + 20}" y="{y + 16}" font-size="10" fill="#666">{desc}</text>'
        )

    # Y-axis label.
    lines.append(
        f'<text x="22" y="{MT + PH/2:.0f}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 22 {MT + PH/2:.0f})">exemplar Hit @ 5 (%)</text>'
    )

    # Footnote about missing.
    if missing:
        lines.append(
            f'<text x="{ML}" y="{H - 12}" font-size="9" fill="#666">'
            f'{len(missing)} session(s) omitted (no MLLM baseline yet): '
            f'{", ".join(_short_name(r["corpus"], r["sequence"]) for r in missing)}'
            f'</text>'
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
            / "journal" / "figures" / "f2_psm_vs_mllm.svg",
    )
    args = ap.parse_args()

    rows = _aggregate(args.repo_root)

    print(f"\n{'corpus':10s}  {'session':35s}  {'PSM':>7s}  {'MLLM':>7s}  {'×':>5s}")
    print("-" * 75)
    for r in rows:
        psm = f"{r['psm']*100:.1f}%" if r["psm"] is not None else "  —  "
        mllm = f"{r['mllm']*100:.1f}%" if r["mllm"] is not None else "  —  "
        ratio = ""
        if r["psm"] is not None and r["mllm"] is not None:
            if r["mllm"] > 0:
                ratio = f"{r['psm']/r['mllm']:.1f}×"
            else:
                ratio = "∞"
        print(f"{r['corpus']:10s}  {r['sequence'][:35]:35s}  {psm:>7s}  {mllm:>7s}  {ratio:>5s}")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_svg(rows))
    print(f"[f2] wrote {args.out}")
    pdf = args.out.with_suffix(".pdf")
    print(f"[f2] render PDF for paper: rsvg-convert -f pdf -o {pdf} {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
