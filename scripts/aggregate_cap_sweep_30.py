#!/usr/bin/env python3
"""Compute per-cap mean Hit@5 across all 30 Nymeria sessions.

Reads captures/multisession_pcc_sweep/<session>/eval_<sid>_pcc<N>.json
files (the output of scripts/multisession_per_cell_cap_sweep.py) and
emits the 4-value row that fills the "30-session mean" placeholder in
section_5_results.tex's tab:multisession.

Run after scripts/multisession_cap_sweep_30.sh completes:
  python scripts/aggregate_cap_sweep_30.py

Prints a markdown row + LaTeX row + summary statistics.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWEEP_DIR = ROOT / "captures" / "multisession_pcc_sweep"
CAPS = [1, 2, 3, 5]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--sweep-dir",
        type=Path,
        default=DEFAULT_SWEEP_DIR,
        help="Directory containing <session>/eval_<sid>_pcc<N>.json files.",
    )
    args = ap.parse_args()
    sweep_dir = args.sweep_dir

    if not sweep_dir.exists():
        print(
            f"!!! {sweep_dir} not found; run multisession_cap_sweep_30.sh first",
            file=sys.stderr,
        )
        return 1

    per_cap: dict[int, list[float]] = {c: [] for c in CAPS}
    sessions = sorted(d.name for d in sweep_dir.iterdir() if d.is_dir())
    for sid in sessions:
        for cap in CAPS:
            p = sweep_dir / sid / f"eval_{sid}_pcc{cap}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            s = d.get("summary") or {}
            h = s.get("exemplar_hit_rate_at_5")
            if h is not None:
                per_cap[cap].append(float(h) * 100.0)

    # 95% CI of the mean across sessions: mean +/- 1.96 * (std / sqrt(n))
    # (normal approx; n=30 is large enough that the t-correction is <5%).
    def ci95(xs: list[float]) -> tuple[float, float, float]:
        n = len(xs)
        m = st.mean(xs)
        sd = st.stdev(xs) if n > 1 else 0.0
        half = 1.96 * sd / (n**0.5) if n > 1 else 0.0
        return m, sd, half

    print(f"n sessions discovered: {len(sessions)}")
    for c in CAPS:
        m, sd, half = ci95(per_cap[c])
        print(
            f"  cap={c}: n={len(per_cap[c]):2d}  "
            f"mean={m:5.2f}%  std={sd:4.2f}  95%CI=[{m-half:5.2f}, {m+half:5.2f}]  "
            f"median={st.median(per_cap[c]):5.2f}%  "
            f"max={max(per_cap[c]):5.1f}%  min={min(per_cap[c]):4.1f}%"
        )

    # Ready-to-paste sentence for supplementary.tex §A.1 (pattern 4) and/or the
    # abstract -- fill the CI values into the existing 30-session sentence.
    print("\nSentence for supp §A.1 (mean +/- 95% CI across 30 sessions, R=128):")
    parts = []
    for c in CAPS:
        m, _sd, half = ci95(per_cap[c])
        parts.append(
            f"\\caponek{{=}}{c if c != CAPS[-1] else 'K'}: ${m:.2f}\\pm{half:.2f}\\%$"
        )
    print("  mean Hit@5 = " + "; ".join(parts) + ".")

    print("\nMarkdown table row (paste into journal/results_v1.md):")
    cells = "  ".join(f"{st.mean(per_cap[c]):.2f}%" for c in CAPS)
    print(f"| **30-session mean** | {cells.replace('  ', ' | ')} |")

    print("\nLaTeX table row (paste into section_5_results.tex):")
    cells_tex = " & ".join(f"\\textbf{{{st.mean(per_cap[c]):.2f}\\%}}" for c in CAPS)
    print(f"\\textbf{{30-session mean}} & {cells_tex} & --- \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
