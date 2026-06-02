"""Per-session SLAM displacement probe for Nymeria.

metadata.json reports `head_trajectory_m` as an integrated path
length (Σ‖step‖), which inflates 5-10x from per-sample jitter at
Aria's ~1000 Hz sampling. The metric that actually matters for PSM
H3 cell carving is the *bounding-box extent* — max - min on each
axis. That's what tells us whether the wearer's positions span
multiple H3 cells at a given resolution.

Outputs (per session, sorted by extent desc):
  - integrated_m: same as metadata.json's head_trajectory_m
  - extent_m:     3D bbox diagonal across (tx, ty, tz)
  - extent_xy_m:  2D bbox diagonal across (tx, ty) only (ground plane)
  - h3_cells_*:   how many H3 cells the path covers at various
                  resolutions, assuming fake (0,0) origin projection

Run from repo root:
  python scripts/nymeria_slam_displacement.py

Override the root if needed:
  python scripts/nymeria_slam_displacement.py /path/to/nymeria_partial
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/nymeria_partial")

# Average H3 cell edge lengths in meters from the H3 docs:
# https://h3geo.org/docs/core-library/restable/#cell-areas
# Edge length ~ avg distance between adjacent cell centers; we use it
# as the "is this bbox larger than one cell?" threshold. The threshold
# is a lower bound on whether PSM might carve into multiple cells —
# actual carving depends on the path crossing boundaries, not just
# spanning a diagonal that exceeds the cell edge.
_H3_EDGE_M = {
    10: 65.9,
    11: 24.9,
    12: 9.4,
    13: 3.56,
    14: 1.35,
    15: 0.51,
}


def parse_trajectory(csv_path: Path) -> tuple[list[float], list[float], list[float]] | None:
    """Read (tx, ty, tz) lists from closed_loop_trajectory.csv. None on missing/bad."""
    if not csv_path.is_file():
        return None
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            need = {"tx_world_device", "ty_world_device", "tz_world_device"}
            if not need.issubset(set(reader.fieldnames or [])):
                return None
            for row in reader:
                try:
                    xs.append(float(row["tx_world_device"]))
                    ys.append(float(row["ty_world_device"]))
                    zs.append(float(row["tz_world_device"]))
                except (KeyError, ValueError):
                    continue
    except OSError:
        return None
    if not xs:
        return None
    return xs, ys, zs


def integrated_length_m(xs: list[float], ys: list[float], zs: list[float]) -> float:
    """Σ‖step‖ in meters — what metadata.json reports."""
    total = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dz = zs[i] - zs[i - 1]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def bbox_extent_m(vals: list[float]) -> float:
    return max(vals) - min(vals) if vals else 0.0


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    if not root.is_dir():
        print(f"ERR: {root} not a directory", file=sys.stderr)
        return 1

    sessions = sorted(p for p in root.iterdir() if p.is_dir())

    header = (
        f"{'session':<55s} "
        f"{'integ_m':>8s} {'ext3D_m':>8s} {'ext_xy':>7s} "
        f"{'dx':>6s} {'dy':>6s} {'dz':>6s} "
        f"{'>r10':>5s} {'>r11':>5s} {'>r12':>5s} {'>r13':>5s} {'>r14':>5s}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for d in sessions:
        csv_path = d / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv"
        parsed = parse_trajectory(csv_path)
        if parsed is None:
            rows.append((0.0, d.name, None))
            continue
        xs, ys, zs = parsed
        integ = integrated_length_m(xs, ys, zs)
        dx = bbox_extent_m(xs)
        dy = bbox_extent_m(ys)
        dz = bbox_extent_m(zs)
        ext3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        ext_xy = math.sqrt(dx * dx + dy * dy)
        spans = {r: ("Y" if ext3d >= _H3_EDGE_M[r] else "-") for r in (10, 11, 12, 13, 14)}
        rows.append((ext3d, d.name, (integ, ext3d, ext_xy, dx, dy, dz, spans)))

    rows.sort(key=lambda r: -r[0])
    for _, name, info in rows:
        if info is None:
            print(f"{name:<55s} (no SLAM trajectory)")
            continue
        integ, ext3d, ext_xy, dx, dy, dz, spans = info
        print(f"{name:<55s} "
              f"{integ:>8.1f} {ext3d:>8.2f} {ext_xy:>7.2f} "
              f"{dx:>6.2f} {dy:>6.2f} {dz:>6.2f} "
              f"{spans[10]:>5s} {spans[11]:>5s} {spans[12]:>5s} {spans[13]:>5s} {spans[14]:>5s}")

    print()
    print("# integ_m   = Σ‖step‖, matches metadata.json's head_trajectory_m")
    print("# ext3D_m   = sqrt(dx² + dy² + dz²), bounding-box diagonal")
    print("# ext_xy    = ground-plane bbox diagonal (tx, ty only)")
    print("# >r10..14  = does ext3D_m exceed avg H3 cell edge at that resolution?")
    print("#   r10=65.9m  r11=24.9m  r12=9.4m  r13=3.56m  r14=1.35m")
    print("#   (cell areas: r10=15047m²  r11=2150m²  r12=307m²  r13=43.9m²  r14=6.27m²)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
