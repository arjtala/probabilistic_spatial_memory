#!/usr/bin/env python3
"""Acceptance check for a SLOPER4D H3-resolution sweep.

Reads `<captures>/h3_res_<R>_<SEQ>_<encoder>_s<seed>.json` files and
prints a per-encoder verdict:

  PASS iff the Hit@5 curve across H3 resolutions is **strictly
  non-decreasing** AND the **absolute lift from r10 to r12 is
  ≥ ABS_LIFT_PP** (default 5 pp).

Rationale (why not a ratio threshold):
  A ratio rule like "r12 ≥ 2× r10" penalizes stronger encoders that
  already discriminate well at r10. On seq009 bigG's r10 baseline is
  2× clipL's, so the same +7pp absolute lift looks like a 1.5× ratio
  vs clipL's 2.5×. Absolute-lift + monotonicity tests the spatial-axis
  claim independent of encoder baseline strength.

Usage:
    python scripts/h3_acceptance.py \\
        --captures captures/sloper4d_seq009_running_002_h3 \\
        --sequence seq009_running_002

Exits 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# H3 resolutions we sweep over. Hard-coded to the canonical Aria /
# Nymeria / SLOPER4D set; if you add r13 etc later, extend here.
SWEEP_RESOLUTIONS = (8, 9, 10, 11, 12)


def evaluate(captures: Path, sequence: str, abs_lift_pp: float) -> tuple[bool, dict]:
    """Return (all_pass, report_dict). report_dict maps encoder → means + flags."""
    fname_re = re.compile(
        rf"^h3_res_(?P<label>\d+)_{re.escape(sequence)}_(?P<enc>clipL|bigG)_s(?P<seed>\d+)\.json$"
    )
    by_cell: dict[tuple[str, int], list[float]] = defaultdict(list)
    for p in sorted(captures.glob("h3_res_*.json")):
        m = fname_re.match(p.name)
        if not m:
            continue
        rate = json.loads(p.read_text()).get("summary", {}).get("exemplar_hit_rate_at_5")
        if rate is None:
            continue
        by_cell[(m["enc"], int(m["label"]))].append(rate)

    report: dict[str, dict] = {}
    all_pass = True
    for enc in ("clipL", "bigG"):
        rows = [(r, by_cell.get((enc, r), [])) for r in SWEEP_RESOLUTIONS]
        if any(len(rates) == 0 for _, rates in rows):
            report[enc] = {"missing": [r for r, v in rows if not v]}
            all_pass = False
            continue
        means = [sum(rates) / len(rates) for _, rates in rows]
        mono = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
        lift_pp = (means[SWEEP_RESOLUTIONS.index(12)] - means[SWEEP_RESOLUTIONS.index(10)]) * 100
        enc_pass = mono and lift_pp >= abs_lift_pp
        report[enc] = {
            "means": dict(zip(SWEEP_RESOLUTIONS, means)),
            "monotone": mono,
            "lift_pp": lift_pp,
            "pass": enc_pass,
        }
        if not enc_pass:
            all_pass = False
    return all_pass, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--captures", type=Path, required=True,
                    help="dir of h3_res_*.json from the sweep")
    ap.add_argument("--sequence", required=True,
                    help="session_id (e.g. seq009_running_002)")
    ap.add_argument("--abs-lift-pp", type=float, default=5.0,
                    help="minimum r10→r12 absolute lift, in percentage points (default 5)")
    args = ap.parse_args()

    all_pass, report = evaluate(args.captures, args.sequence, args.abs_lift_pp)

    header_cells = "  ".join(f"r{r:>2d} mean" for r in SWEEP_RESOLUTIONS)
    print(f"{'encoder':6s}  {header_cells}  {'mono':6s}  {'lift10→12':9s}  verdict")
    print("-" * (16 + 9 * len(SWEEP_RESOLUTIONS) + 30))
    for enc in ("clipL", "bigG"):
        info = report.get(enc, {})
        if "missing" in info:
            print(f"{enc:6s}  MISSING resolutions: {info['missing']}")
            continue
        cells = "  ".join(f"{info['means'][r]*100:6.1f}%" for r in SWEEP_RESOLUTIONS)
        mono_s = "yes" if info["monotone"] else "no"
        verdict = "PASS" if info["pass"] else "FAIL"
        print(f"{enc:6s}  {cells}  {mono_s:6s}  {info['lift_pp']:+5.1f}pp   {verdict}")
    print()
    if all_pass:
        print(f"ACCEPTANCE: PASS — both encoders show monotone H3 curve + ≥{args.abs_lift_pp:.0f}pp lift r10→r12 ({args.sequence})")
        return 0
    else:
        print(f"ACCEPTANCE: FAIL — at least one encoder is not monotone or lift <{args.abs_lift_pp:.0f}pp ({args.sequence})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
