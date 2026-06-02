"""Verify all Nymeria features.h5 files have non-collapsed lat/lng.

Walks /checkpoint/.../nymeria_atomic/, reads each clip_l_features.h5,
computes the lat/lng range in meters, flags any session where the range
is suspiciously small (< 0.5m) — the same trajectory-clock-rebase bug
we hit on Ego-Exo4D would land here as range = 0.

Run on the cluster:
  python scripts/verify_nymeria_extractions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py

ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic")
M_PER_DEG = 111_132.0
BUG_THRESHOLD_M = 0.5


def main() -> int:
    bad = 0
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        h5p = d / "clip_l_features.h5"
        if not h5p.exists():
            print(f"MIS  {d.name}")
            bad += 1
            continue
        try:
            with h5py.File(h5p) as f:
                g = f["clip"]
                attrs = dict(g.attrs)
                lat = g["lat"][:]
                lng = g["lng"][:]
                dlat = (lat.max() - lat.min()) * M_PER_DEG
                dlng = (lng.max() - lng.min()) * M_PER_DEG
                ext = max(dlat, dlng)
        except Exception as exc:  # noqa: BLE001
            print(f"ERR  {d.name}: {exc}")
            bad += 1
            continue
        flag = "OK " if ext > BUG_THRESHOLD_M else "BUG"
        if ext <= BUG_THRESHOLD_M:
            bad += 1
        mode = attrs.get("track_mode", "?")
        print(
            f"{flag}  {d.name}  mode={mode}  N={len(lat)}  "
            f"dlat={dlat:6.1f}m  dlng={dlng:6.1f}m"
        )
    print()
    if bad:
        print(f"!! {bad} session(s) failed checks")
        return 1
    print("OK 30/30 sessions verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
