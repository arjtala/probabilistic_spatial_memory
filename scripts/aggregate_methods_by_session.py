#!/usr/bin/env python3
"""Aggregate Hit@5 / mIoU@5 per (method, session) across captures/.

Walks captures/eval_<session>_<tag>.json files, groups by session
and method-tag, and emits a TSV with one row per session and one
column per method.

Methods recognized (by filename suffix):
  _pool0s_p50   -> psm_W0      (PSM baseline; equiv to per-frame brute force)
  _pool30s_p50  -> psm_W30     (PSM + W=30 query-time pool rerank)
  _sliding_w3s  -> sw_3s
  _sliding_w5s  -> sw_5s
  _sliding_w10s -> sw_10s
  _uniform_r30s -> uniform_30s
  _uniform_r75s -> uniform_75s

Writes a markdown summary table at the bottom for paste-into-paper
use.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPTURES = ROOT / "captures"

METHOD_TAGS = {
    "_pool0s_p50":   "psm_W0",
    "_pool30s_p50":  "psm_W30",
    "_sliding_w3s":  "sw_3s",
    "_sliding_w5s":  "sw_5s",
    "_sliding_w10s": "sw_10s",
    "_uniform_r30s": "uniform_30s",
    "_uniform_r75s": "uniform_75s",
    "_longclip_text": "longclip",
}


def _summary(p: Path) -> dict:
    d = json.loads(p.read_text())
    return d.get("summary") or {}


def main() -> int:
    by_session: dict[str, dict[str, float]] = {}
    for p in sorted(CAPTURES.glob("eval_*.json")):
        name = p.name[len("eval_"):-len(".json")]
        tag = None
        sid = None
        for suffix, method in METHOD_TAGS.items():
            if name.endswith(suffix):
                tag = method
                sid = name[: -len(suffix)]
                break
        if not tag:
            continue
        s = _summary(p)
        if not s:
            continue
        by_session.setdefault(sid, {})[tag] = s.get("exemplar_hit_rate_at_5", 0.0) * 100

    if not by_session:
        print("no eval_*.json files matched recognized method tags", file=sys.stderr)
        return 1

    methods = sorted({m for d in by_session.values() for m in d})
    sessions = sorted(by_session)

    # TSV
    print("\t".join(["session"] + methods))
    for sid in sessions:
        row = by_session[sid]
        print("\t".join([sid] + [f"{row.get(m, float('nan')):.1f}" for m in methods]))

    # Means
    print("\nMEAN  Hit@5 (%) across sessions:")
    for m in methods:
        vals = [d[m] for d in by_session.values() if m in d]
        if not vals:
            continue
        print(f"  {m:14s}  n={len(vals):2d}  mean={sum(vals)/len(vals):5.2f}  "
              f"median={sorted(vals)[len(vals)//2]:5.2f}  "
              f"max={max(vals):5.2f}  min={min(vals):5.2f}")

    # If both psm_W0 and psm_W30 present, compute per-session delta.
    if "psm_W0" in methods and "psm_W30" in methods:
        deltas = []
        improved = 0; neutral = 0; worsened = 0
        for sid, d in by_session.items():
            if "psm_W0" in d and "psm_W30" in d:
                delta = d["psm_W30"] - d["psm_W0"]
                deltas.append(delta)
                if delta > 0.05: improved += 1
                elif delta < -0.05: worsened += 1
                else: neutral += 1
        if deltas:
            mean = sum(deltas) / len(deltas)
            print(f"\nApproach-1 delta (psm_W30 - psm_W0):")
            print(f"  n={len(deltas)}  mean={mean:+.2f}pp  "
                  f"improved={improved}  neutral={neutral}  worsened={worsened}")

    # If sw_10s present, compare to psm_W0.
    if "sw_10s" in methods and "psm_W0" in methods:
        diffs = []
        for d in by_session.values():
            if "sw_10s" in d and "psm_W0" in d:
                diffs.append(d["sw_10s"] - d["psm_W0"])
        if diffs:
            print(f"\nSliding-window @ 10s vs psm_W0:")
            print(f"  n={len(diffs)}  mean diff={sum(diffs)/len(diffs):+.2f}pp")

    return 0


if __name__ == "__main__":
    sys.exit(main())
