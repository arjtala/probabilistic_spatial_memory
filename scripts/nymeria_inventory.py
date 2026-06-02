"""Inventory the Nymeria sessions on the cluster.

Walks `/checkpoint/.../nymeria_partial/`, reads each session's
metadata.json, counts narrations, and checks that the VRS + SLAM
trajectory are on disk. Prints a sortable table.

Run from the repo root:
  python scripts/nymeria_inventory.py

Override the root if needed:
  python scripts/nymeria_inventory.py /path/to/other/nymeria_dir
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/nymeria_partial")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    if not root.is_dir():
        print(f"ERR: {root} not a directory", file=sys.stderr)
        return 1

    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"{'session':<55s} {'act':<6s} {'traj_m':>8s} {'dur_s':>7s} "
          f"{'narr':>5s} {'vrs':<4s} {'slam':<5s}  script @ location")
    print("-" * 140)
    rows = []
    for d in sessions:
        meta_path = d / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            m = json.loads(meta_path.read_text())
        except Exception as exc:
            print(f"# {d.name}: metadata read failed: {exc}", file=sys.stderr)
            continue
        traj = m.get("head_trajectory_m") or 0.0
        dur = m.get("action_duration_sec") or 0.0
        act = m.get("act_id", "?")
        script = m.get("script", "?")
        loc = m.get("location", "?")
        narr_csv = d / "narration" / "atomic_action.csv"
        vrs = d / "recording_head" / "data" / "data.vrs"
        slam = d / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv"
        n_narr = 0
        if narr_csv.exists():
            try:
                with narr_csv.open() as f:
                    n_narr = sum(1 for _ in f) - 1  # minus header
            except OSError:
                n_narr = 0
        rows.append((traj, d.name, act, dur, n_narr, vrs.exists(), slam.exists(), script, loc))

    rows.sort(key=lambda r: -r[0])
    for traj, name, act, dur, n_narr, has_vrs, has_slam, script, loc in rows:
        v = "Y" if has_vrs else "-"
        s = "Y" if has_slam else "-"
        print(f"{name:<55s} {act:<6s} {traj:>7.1f} {dur:>7.0f} "
              f"{n_narr:>5d} {v:<4s} {s:<5s}  {script} @ {loc}")

    total_narr = sum(r[4] for r in rows)
    n_with_vrs = sum(1 for r in rows if r[5])
    n_with_slam = sum(1 for r in rows if r[6])
    print()
    print(f"# {len(rows)} sessions; {total_narr} total narrations; "
          f"{n_with_vrs} with VRS; {n_with_slam} with SLAM")
    return 0


if __name__ == "__main__":
    sys.exit(main())
