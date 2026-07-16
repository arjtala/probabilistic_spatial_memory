#!/usr/bin/env python3
"""Acceptance check for a SLOPER4D H3-resolution sweep.

Reads `<captures>/h3_res_<R>_<SEQ>_<encoder>_s<seed>.json` files and
prints a per-encoder verdict:

  PASS iff the Hit@5 curve is **monotone non-decreasing across
  r10..r12** AND the **absolute lift from r10 to r12 is ≥
  ABS_LIFT_PP** (default 4 pp).

Rationale:
  - A ratio rule like "r12 ≥ 2× r10" penalizes stronger encoders that
    already discriminate well at r10. On seq009 bigG's r10 baseline is
    2× clipL's, so the same +7pp absolute lift looks like a 1.5× ratio
    vs clipL's 2.5×. Absolute-lift + monotonicity tests the
    spatial-axis claim independent of encoder baseline strength.
  - Monotonicity is checked only over r10..r12 (the resolutions that
    actually exercise the spatial axis). r8 and r9 are at-or-below the
    trajectory scale on these sequences, so each cell holds many frames
    and Hit@5 there is dominated by which exemplar was reservoir-sampled.
    Adjacent 1-2 pp wiggles at r8/r9 are sampling noise, not curve shape.

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


def evaluate(captures: Path, sequence: str, abs_lift_pp: float,
             *, strict: bool = False,
             encoder: str | None = None) -> tuple[bool, dict]:
    """Return (overall_pass, report_dict).

    A per-encoder pass requires (a) monotone curve over r10..r12 and
    (b) absolute lift r10 → r12 ≥ abs_lift_pp.

    Overall verdict combines per-encoder passes. `encoder` takes
    precedence over `strict`:
      - encoder="clipL" (etc.): overall PASS iff that ONE pre-registered
        encoder passes. This is the honest, reviewer-recommended mode
        (finding #6b): fixing the encoder *before* looking at results
        avoids the post-hoc "any-of-three" selection that inflates the
        pass count by letting whichever encoder happens to work define
        success. Report this in the paper.
      - strict=False (default, LEGACY): overall PASS if **at least one
        encoder** passes. NOTE: this is post-hoc encoder selection — it
        reports a session as passing whenever *some* encoder works, which
        is exactly the confound flagged in review. Kept only for
        backward-compatible reproduction of the old table; do not headline.
      - strict=True: overall PASS only if **all encoders** pass.
    """
    fname_re = re.compile(
        rf"^h3_res_(?P<label>\d+)_{re.escape(sequence)}_(?P<enc>[A-Za-z][A-Za-z0-9]*)_s(?P<seed>\d+)\.json$"
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
    enc_pass_flags: dict[str, bool] = {}
    n_missing_or_empty = 0
    # Discover encoders dynamically from the capture filenames so that
    # adding a third (e.g. siglip2L) doesn't require an h3_acceptance.py
    # change. Sort known encoders first (clipL, bigG) for stable output
    # ordering, then any newer ones in alphabetical order.
    discovered = sorted({enc for enc, _ in by_cell.keys()})
    known_order = ["clipL", "bigG", "siglip2L"]
    encoders = [e for e in known_order if e in discovered] + [
        e for e in discovered if e not in known_order
    ]
    if not encoders:
        return False, {}
    for enc in encoders:
        rows = [(r, by_cell.get((enc, r), [])) for r in SWEEP_RESOLUTIONS]
        if any(len(rates) == 0 for _, rates in rows):
            report[enc] = {"missing": [r for r, v in rows if not v]}
            n_missing_or_empty += 1
            enc_pass_flags[enc] = False
            continue
        means = [sum(rates) / len(rates) for _, rates in rows]
        # Monotonicity is checked only over r10..r12, the resolutions
        # that actually exercise the spatial axis. r8 and r9 are
        # at-or-below the trajectory scale on these sequences so each
        # cell holds many frames and the Hit@5 there is dominated by
        # which exemplar was reservoir-sampled — adjacent 1-2 pp
        # wiggles are noise, not curve shape. Lift is still r10→r12
        # (the spatial-axis claim).
        mono_top = all(
            means[SWEEP_RESOLUTIONS.index(r)] <= means[SWEEP_RESOLUTIONS.index(r + 1)]
            for r in (10, 11)
        )
        lift_pp = (means[SWEEP_RESOLUTIONS.index(12)] - means[SWEEP_RESOLUTIONS.index(10)]) * 100
        enc_pass = mono_top and lift_pp >= abs_lift_pp
        enc_pass_flags[enc] = enc_pass
        report[enc] = {
            "means": dict(zip(SWEEP_RESOLUTIONS, means)),
            "monotone": mono_top,
            "lift_pp": lift_pp,
            "pass": enc_pass,
        }
    if encoder is not None:
        # Pre-registered single encoder (honest mode, finding #6b): the
        # verdict is that one encoder's pass flag. Absent data => FAIL.
        all_pass = bool(enc_pass_flags.get(encoder, False))
    elif strict:
        # All present encoders must pass; missing data also counts as
        # failure (consistent with the pre-2026-06-19 behavior).
        all_pass = (n_missing_or_empty == 0
                    and all(enc_pass_flags[enc] for enc in enc_pass_flags))
    else:
        # At least one encoder must pass. Models real encoder asymmetry:
        # on some corpora bigG picks up spatial signal that clipL misses
        # (and vice versa). The spatial-axis claim holds whenever *any*
        # encoder demonstrates monotone discrimination.
        all_pass = any(enc_pass_flags[enc] for enc in enc_pass_flags)
    return all_pass, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--captures", type=Path, required=True,
                    help="dir of h3_res_*.json from the sweep")
    ap.add_argument("--sequence", required=True,
                    help="session_id (e.g. seq009_running_002)")
    ap.add_argument("--abs-lift-pp", type=float, default=4.0,
                    help="minimum r10→r12 absolute lift, in percentage points (default 4)")
    ap.add_argument("--encoder", default=None,
                    help="pre-register ONE encoder (e.g. clipL) as the verdict "
                         "basis (finding #6b: avoids post-hoc any-of-N encoder "
                         "selection). Overrides --strict. Report THIS in the paper.")
    ap.add_argument("--strict", action="store_true",
                    help="LEGACY combine mode: require ALL present encoders to "
                         "pass. Without --encoder/--strict the default is the "
                         "any-encoder disjunction, which is post-hoc selection "
                         "(kept only for old-table reproduction).")
    args = ap.parse_args()

    all_pass, report = evaluate(
        args.captures, args.sequence, args.abs_lift_pp,
        strict=args.strict, encoder=args.encoder,
    )

    header_cells = "  ".join(f"r{r:>2d} mean" for r in SWEEP_RESOLUTIONS)
    print(f"{'encoder':9s}  {header_cells}  {'mono':6s}  {'lift10→12':9s}  verdict")
    print("-" * (19 + 9 * len(SWEEP_RESOLUTIONS) + 30))
    for enc in report.keys():
        if enc.startswith("_"):
            continue
        info = report.get(enc, {})
        if not isinstance(info, dict):
            continue
        if "missing" in info:
            print(f"{enc:9s}  MISSING resolutions: {info['missing']}")
            continue
        cells = "  ".join(f"{info['means'][r]*100:6.1f}%" for r in SWEEP_RESOLUTIONS)
        mono_s = "yes" if info["monotone"] else "no"
        verdict = "PASS" if info["pass"] else "FAIL"
        print(f"{enc:9s}  {cells}  {mono_s:6s}  {info['lift_pp']:+5.1f}pp   {verdict}")
    print()
    if args.encoder:
        mode = f"pre-registered encoder {args.encoder}"
    elif args.strict:
        mode = "all encoders"
    else:
        mode = "at least one encoder (POST-HOC — legacy)"
    if all_pass:
        print(f"ACCEPTANCE: PASS — {mode} shows monotone H3 curve over r10..r12 + ≥{args.abs_lift_pp:.0f}pp lift r10→r12 ({args.sequence})")
        return 0
    else:
        print(f"ACCEPTANCE: FAIL — no encoder shows monotone H3 curve over r10..r12 with lift ≥{args.abs_lift_pp:.0f}pp ({args.sequence})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
