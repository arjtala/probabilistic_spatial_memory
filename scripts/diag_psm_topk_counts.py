"""Probe PSM top-k candidate counts per query, at two operating points.

Diagnoses: when PSM at r10 single-cell returns only 1 candidate but
the user asked --top 5, is that a single-cell collapse (architectural)
or a real bug? Print median/mean/max candidates per query at r12
vs r10.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

R12 = Path("/tmp/full_psm_ex1024_20230608_s0_shelby_arroyo_act0_3ciwl8.json")
R10 = Path("/tmp/full_psm_r10_unbounded_20230608_s0_shelby_arroyo_act0_3ciwl8.json")


def summarize(path: Path) -> None:
    d = json.loads(path.read_text())
    recs = d["records"]
    npreds = [len(q["preds"]) for q in recs]
    nq_5 = sum(1 for n in npreds if n >= 5)
    nq_3 = sum(1 for n in npreds if n >= 3)
    nq_2 = sum(1 for n in npreds if n >= 2)
    nq_1 = sum(1 for n in npreds if n >= 1)
    print(f"{path.name}")
    print(f"  median={statistics.median(npreds):.0f}  "
          f"mean={statistics.mean(npreds):.1f}  "
          f"max={max(npreds)}  min={min(npreds)}")
    print(f"  >=1: {nq_1}/{len(recs)}  >=2: {nq_2}/{len(recs)}  "
          f">=3: {nq_3}/{len(recs)}  >=5: {nq_5}/{len(recs)}")


for p in (R12, R10):
    summarize(p)
