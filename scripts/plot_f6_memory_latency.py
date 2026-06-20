#!/usr/bin/env python3
"""F6 paper figure: memory + latency vs session length.

X-axis: session length in frames (proxy for session duration at 1 fps).
Two panels (or one combined plot):
  - Memory in MB to hold the per-session retrieval state:
      brute-force CLIP: N × dim × 4 bytes (linear in N)
      PSM: fixed-per-cell × n_cells (bounded; effectively flat)
  - Latency (median µs per query): brute-force grows linearly in N
    in principle, but our bench shows ~40 µs floor across the
    measurement range due to cache + setup overhead.

Inputs:
  - benchmarks/nymeria/*.json: 30 Nymeria sessions, brute-force
    CLIP-L median+p99 latency + bytes_in_ram per session.
  - benchmarks/bench_brute_force_clip_{l,bigg}_features.json: 2
    Aria-internal sessions.
  - PSM reference: hardcoded ~1 MB/cell × ~10 cells ≈ 13 MB
    at the deployment configuration (R=128 exemplars × 1024-d ×
    4 bytes ≈ 0.5 MB per cell; with ring buffer + HLL ~1 MB
    total). See section_5_results.tex sec:results-memory-latency.

Usage:

    python scripts/plot_f6_memory_latency.py \\
        --out journal/figures/f6_memory_latency.svg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_bench(repo_root: Path) -> list[dict]:
    """Return [{n_frames, median_us, bytes_in_ram, dim}, ...] across
    all bench JSONs."""
    rows = []
    for p in sorted((repo_root / "benchmarks" / "nymeria").glob("*.json")):
        d = json.loads(p.read_text())
        rows.append({
            "n_frames": d["n_frames"],
            "median_us": d["median_us"],
            "bytes_in_ram": d["bytes_in_ram"],
            "dim": d["dim"],
            "src": "Nymeria",
        })
    for f in ("bench_brute_force_clip_l_features.json", "bench_brute_force_clip_bigg_features.json"):
        p = repo_root / "benchmarks" / f
        if p.exists():
            d = json.loads(p.read_text())
            rows.append({
                "n_frames": d["n_frames"],
                "median_us": d["median_us"],
                "bytes_in_ram": d["bytes_in_ram"],
                "dim": d["dim"],
                "src": "Aria",
            })
    return rows


# PSM bounded-memory deployment reference: 10 cells × ~1 MB/cell at
# R=128, dim=1024 reservoir + ring buffer + HLL. Matches the §5
# memory text. Constant in session length.
_PSM_MB = 13.0


# Plot constants — same style as F3.
_W = 720
_H = 360
_ML = 80
_MR = 160
_MT = 50
_MB = 70
_PW = _W - _ML - _MR
_PH = _H - _MT - _MB


def _render_svg(rows: list[dict]) -> str:
    if not rows:
        raise RuntimeError("no bench rows")

    # X-axis: n_frames. Pad domain to nearest 250.
    n_vals = [r["n_frames"] for r in rows]
    x_max = max(n_vals)
    x_max = int(((x_max + 250) // 250) * 250)

    # Y-axis: MB. PSM constant ~13; brute-force grows up to ~4 MB at
    # N=1325 for CLIP-L (768d). Scale to nearest 5 MB above max of
    # PSM line + brute-force max.
    y_max_data = max(_PSM_MB, max(r["bytes_in_ram"] / 1024 / 1024 for r in rows))
    y_max = max(15.0, round(y_max_data + 2.5))

    def x_pos(n: int) -> float:
        return _ML + (n / x_max) * _PW

    def y_pos(mb: float) -> float:
        return _MT + _PH - (mb / y_max) * _PH

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="12">'
    )
    lines.append(
        f'<text x="{_W/2:.0f}" y="22" text-anchor="middle" font-size="14" '
        f'font-weight="bold">Memory footprint vs session length</text>'
    )

    # Axes.
    lines.append(
        f'<line x1="{_ML}" y1="{_MT + _PH}" x2="{_ML + _PW}" y2="{_MT + _PH}" '
        f'stroke="black" stroke-width="1"/>'
    )
    lines.append(
        f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + _PH}" '
        f'stroke="black" stroke-width="1"/>'
    )

    # Y ticks every 5 MB.
    n_ticks = int(y_max // 5)
    for i in range(n_ticks + 1):
        mb = i * 5
        y = y_pos(mb)
        lines.append(
            f'<line x1="{_ML - 5}" y1="{y:.1f}" x2="{_ML}" y2="{y:.1f}" '
            f'stroke="black" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{_ML - 10}" y="{y + 4:.1f}" text-anchor="end">{mb}</text>'
        )
        lines.append(
            f'<line x1="{_ML}" y1="{y:.1f}" x2="{_ML + _PW}" y2="{y:.1f}" '
            f'stroke="#dddddd" stroke-width="1" stroke-dasharray="2,2"/>'
        )

    # X ticks at 250-frame intervals.
    n_xticks = int(x_max // 250)
    for i in range(n_xticks + 1):
        n = i * 250
        x = x_pos(n)
        lines.append(
            f'<line x1="{x:.1f}" y1="{_MT + _PH}" x2="{x:.1f}" y2="{_MT + _PH + 5}" '
            f'stroke="black" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{_MT + _PH + 18}" text-anchor="middle">{n}</text>'
        )
    lines.append(
        f'<text x="{_ML + _PW/2:.0f}" y="{_MT + _PH + 45}" text-anchor="middle" font-size="13">'
        f'session length (frames @ 1 fps)</text>'
    )
    lines.append(
        f'<text x="22" y="{_MT + _PH/2:.0f}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 22 {_MT + _PH/2:.0f})">memory (MB)</text>'
    )

    # PSM bounded-memory reference: horizontal line at _PSM_MB.
    psm_y = y_pos(_PSM_MB)
    lines.append(
        f'<line x1="{_ML}" y1="{psm_y:.1f}" x2="{_ML + _PW}" y2="{psm_y:.1f}" '
        f'stroke="#1f77b4" stroke-width="2.5" stroke-dasharray="8,4"/>'
    )
    # Brute-force linear fit + scatter.
    xs = np.array([r["n_frames"] for r in rows], dtype=np.float64)
    ys = np.array([r["bytes_in_ram"] / 1024 / 1024 for r in rows], dtype=np.float64)
    # Linear regression through origin: y = m*x.
    m = float((xs * ys).sum() / (xs * xs).sum())
    # Scatter points.
    for r in rows:
        bx = x_pos(r["n_frames"])
        by = y_pos(r["bytes_in_ram"] / 1024 / 1024)
        color = "#d62728" if r["src"] == "Nymeria" else "#ff7f0e"
        lines.append(
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="3.5" fill="{color}" '
            f'fill-opacity="0.7" stroke="white" stroke-width="0.8"/>'
        )
    # Brute-force linear fit line.
    x_fit = np.linspace(0, x_max, 50)
    y_fit = m * x_fit
    pts = " ".join(f"{x_pos(int(x)):.1f},{y_pos(y):.1f}" for x, y in zip(x_fit, y_fit) if y <= y_max)
    lines.append(
        f'<polyline points="{pts}" stroke="#d62728" stroke-width="2" fill="none"/>'
    )

    # Legend (right side).
    lgx = _ML + _PW + 20
    lgy = _MT + 30
    lines.append(
        f'<text x="{lgx}" y="{lgy - 12}" font-weight="bold">Method</text>'
    )
    # PSM
    lines.append(
        f'<line x1="{lgx}" y1="{lgy + 6}" x2="{lgx + 24}" y2="{lgy + 6}" '
        f'stroke="#1f77b4" stroke-width="2.5" stroke-dasharray="8,4"/>'
    )
    lines.append(f'<text x="{lgx + 30}" y="{lgy + 10}">PSM (bounded)</text>')
    lines.append(
        f'<text x="{lgx + 30}" y="{lgy + 24}" font-size="10" fill="#666">'
        f'~{_PSM_MB:.0f} MB at R=128, 10 cells</text>'
    )
    # Brute-force
    lines.append(
        f'<line x1="{lgx}" y1="{lgy + 56}" x2="{lgx + 24}" y2="{lgy + 56}" '
        f'stroke="#d62728" stroke-width="2"/>'
    )
    lines.append(
        f'<circle cx="{lgx + 12}" cy="{lgy + 56}" r="3.5" fill="#d62728" '
        f'fill-opacity="0.7" stroke="white" stroke-width="0.8"/>'
    )
    lines.append(f'<text x="{lgx + 30}" y="{lgy + 60}">brute-force CLIP</text>')
    lines.append(
        f'<text x="{lgx + 30}" y="{lgy + 74}" font-size="10" fill="#666">'
        f'N × dim × 4 B, growing</text>'
    )

    # Note at bottom.
    lines.append(
        f'<text x="{_W - 10}" y="{_H - 12}" text-anchor="end" font-size="10" fill="#666">'
        f'{len(rows)} Nymeria + Aria CLIP-L bench JSONs</text>'
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
            / "journal" / "figures" / "f6_memory_latency.svg",
    )
    args = ap.parse_args()

    rows = _load_bench(args.repo_root)
    if not rows:
        raise SystemExit("no bench JSONs found under benchmarks/")

    # Print summary table.
    rows_s = sorted(rows, key=lambda r: r["n_frames"])
    print(f"\n{'src':10s}  {'n_frames':>9s}  {'median_us':>10s}  {'memory_MB':>10s}")
    print("-" * 50)
    for r in rows_s[:5]:
        mb = r["bytes_in_ram"] / 1024 / 1024
        print(f"{r['src']:10s}  {r['n_frames']:>9d}  {r['median_us']:>10.1f}  {mb:>9.2f}")
    print(f"  ...  ({len(rows_s)} rows total)")
    for r in rows_s[-5:]:
        mb = r["bytes_in_ram"] / 1024 / 1024
        print(f"{r['src']:10s}  {r['n_frames']:>9d}  {r['median_us']:>10.1f}  {mb:>9.2f}")
    print()
    n_max = max(r["n_frames"] for r in rows)
    mb_max = max(r["bytes_in_ram"] for r in rows) / 1024 / 1024
    print(f"PSM constant @ {_PSM_MB} MB; brute-force at N={n_max} uses {mb_max:.2f} MB")
    crossover_n = int(_PSM_MB * 1024 * 1024 / (768 * 4))
    print(f"Brute-force=PSM crossover at N ≈ {crossover_n} (CLIP-L 768-d × 4 B)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_render_svg(rows))
    print(f"\n[f6] wrote {args.out}")
    pdf = args.out.with_suffix(".pdf")
    print(f"[f6] render PDF for paper: rsvg-convert -f pdf -o {pdf} {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
