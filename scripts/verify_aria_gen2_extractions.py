#!/usr/bin/env python3
"""Quick verifier: print track_mode + frame count for each Aria Gen 2 features.h5.

Usage:
  python scripts/verify_aria_gen2_extractions.py [ROOT]

ROOT defaults to /checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot.
Reports one line per session; missing files or unexpected track_modes
print loudly so they're easy to grep for.

Expected per journal/manifests/aria_gen2_subset_v1.yaml:
  walk_0, walk_1 -> vrs_gps
  others          -> vrs_slam
Anything else (synthetic_snake_grid, missing file, error) means
something went wrong in extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py

EXPECTED = {
    "walk_0": "vrs_gps",
    "walk_1": "vrs_gps",
    "clean_0": "vrs_slam",
    "cook_0": "vrs_slam",
    "eat_0": "vrs_slam",
    "eat_1": "vrs_slam",
    "eat_2": "vrs_slam",
    "eat_3": "vrs_slam",
    "play_0": "vrs_slam",
    "play_1": "vrs_slam",
    "play_2": "vrs_slam",
    "play_3": "vrs_slam",
}


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/checkpoint/dream/arjangt/video_retrieval/aria_gen2_pilot"
    )
    bad = 0
    for name in sorted(EXPECTED):
        h5 = root / name / "clip_l_features.h5"
        if not h5.exists():
            print(f"{name:9s}  MISSING  ({h5})")
            bad += 1
            continue
        try:
            with h5py.File(h5) as f:
                g = f["clip"]
                mode = dict(g.attrs).get("track_mode", "(none)")
                ts = g["timestamps"]
                n = ts.shape[0]
                dur = float(ts[-1] - ts[0]) if n else 0.0
        except Exception as exc:
            print(f"{name:9s}  ERR     {exc}")
            bad += 1
            continue
        expected = EXPECTED[name]
        flag = "OK " if mode == expected else "!! "
        if mode != expected:
            bad += 1
        print(f"{name:9s}  {flag} track_mode={mode!s:<22}  N={n:5d}  dur={dur:6.1f}s  (expected {expected})")
    print()
    print(f"{12 - bad}/12 sessions match expectations.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
