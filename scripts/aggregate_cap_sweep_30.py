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
        print(f"!!! {sweep_dir} not found; run multisession_cap_sweep_30.sh first",
              file=sys.stderr)
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

    print(f"n sessions discovered: {len(sessions)}")
    for c in CAPS:
        print(f"  cap={c}: n={len(per_cap[c]):2d}  "
              f"mean={st.mean(per_cap[c]):5.2f}%  "
              f"median={st.median(per_cap[c]):5.2f}%  "
              f"max={max(per_cap[c]):5.1f}%  min={min(per_cap[c]):4.1f}%")

    print("\nMarkdown table row (paste into journal/results_v1.md):")
    cells = "  ".join(f"{st.mean(per_cap[c]):.2f}%" for c in CAPS)
    print(f"| **30-session mean** | {cells.replace('  ', ' | ')} |")

    print("\nLaTeX table row (paste into section_5_results.tex):")
    cells_tex = " & ".join(f"\\textbf{{{st.mean(per_cap[c]):.2f}\\%}}" for c in CAPS)
    print(f"\\textbf{{30-session mean}} & {cells_tex} & --- \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
