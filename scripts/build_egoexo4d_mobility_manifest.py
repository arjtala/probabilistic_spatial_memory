"""Score every Ego-Exo4D take by wearer trajectory length, emit a manifest.

The PSM look-back claim depends on the wearer actually moving. Sums per-step
SLAM displacements (`closed_loop_trajectory.csv` -> tx/ty/tz_world_device)
into a total path length per take. Writes a JSON manifest of all takes
with their trajectory length, then a derived text list of takes above the
mobility threshold (default 50m) for the PSM eval sweeps to iterate.

Stats banner prints the distribution (median, p25/p75/p90/p99, count above
threshold) so the threshold choice is informed by what the data actually
looks like.

Usage:
  python scripts/build_egoexo4d_mobility_manifest.py \\
    [--takes-root /datasets/egoexo4d/v2/takes] \\
    [--questions-root /checkpoint/.../egoexo4d_atomic] \\
    [--threshold-m 50] \\
    [--out-manifest /checkpoint/.../egoexo4d_mobility.json] \\
    [--out-take-list /checkpoint/.../egoexo4d_mobile_takes.txt]

Manifest schema:
  {
    "threshold_m": 50,
    "n_total": 696,
    "n_above_threshold": <N>,
    "takes": [
      {"take_name": "...", "traj_len_m": 123.4, "duration_s": 1083.0,
       "n_questions": 229, "above_threshold": true},
      ...  # sorted by traj_len_m desc
    ],
  }

The take-list is just the take_names sorted by traj_len_m desc, one per
line — what `extract_egoexo4d.sbatch`-style array jobs can read directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def trajectory_length_m(csv_path: Path) -> float:
    """Sum 3D step distances from an Aria MPS closed_loop_trajectory.csv.

    Schema columns of interest (same as aria_vrs._read_slam_trajectory):
      - tx_world_device, ty_world_device, tz_world_device (meters)

    Sums sqrt((dx)^2 + (dy)^2 + (dz)^2) across consecutive rows. Returns
    0.0 for malformed/empty files (caller should treat as no-data, not
    zero-mobility).
    """
    prev: tuple[float, float, float] | None = None
    total = 0.0
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            required = {"tx_world_device", "ty_world_device", "tz_world_device"}
            if not required.issubset(set(reader.fieldnames or [])):
                return 0.0
            for row in reader:
                try:
                    p = (
                        float(row["tx_world_device"]),
                        float(row["ty_world_device"]),
                        float(row["tz_world_device"]),
                    )
                except (KeyError, ValueError):
                    continue
                if prev is not None:
                    dx = p[0] - prev[0]
                    dy = p[1] - prev[1]
                    dz = p[2] - prev[2]
                    total += (dx * dx + dy * dy + dz * dz) ** 0.5
                prev = p
    except OSError:
        return 0.0
    return total


def trajectory_duration_s(csv_path: Path) -> float:
    """Wall time spanned by the trajectory CSV. 0.0 if unparseable."""
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            if "tracking_timestamp_us" not in (reader.fieldnames or []):
                return 0.0
            first_t: float | None = None
            last_t: float | None = None
            for row in reader:
                try:
                    t = float(row["tracking_timestamp_us"])
                except (KeyError, ValueError):
                    continue
                if first_t is None:
                    first_t = t
                last_t = t
            if first_t is None or last_t is None:
                return 0.0
            return (last_t - first_t) / 1e6
    except OSError:
        return 0.0


def count_questions(questions_yaml: Path) -> int:
    if not questions_yaml.is_file():
        return 0
    return sum(
        1 for line in questions_yaml.read_text().splitlines()
        if line.lstrip().startswith("- id:")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--takes-root", type=Path,
        default=Path("/datasets/egoexo4d/v2/takes"),
    )
    ap.add_argument(
        "--questions-root", type=Path,
        default=Path("/checkpoint/dream/arjangt/video_retrieval/egoexo4d_atomic"),
    )
    ap.add_argument("--threshold-m", type=float, default=50.0)
    ap.add_argument(
        "--out-manifest", type=Path,
        default=Path("/checkpoint/dream/arjangt/video_retrieval/egoexo4d_mobility.json"),
    )
    ap.add_argument(
        "--out-take-list", type=Path,
        default=Path("/checkpoint/dream/arjangt/video_retrieval/egoexo4d_mobile_takes.txt"),
    )
    args = ap.parse_args()

    if not args.questions_root.is_dir():
        print(f"ERR: {args.questions_root} not found", file=sys.stderr)
        return 1

    take_names = sorted(
        d.name for d in args.questions_root.iterdir() if d.is_dir()
    )
    print(f"[mobility] scoring {len(take_names)} takes...", file=sys.stderr)

    entries: list[dict] = []
    for i, name in enumerate(take_names, start=1):
        csv_path = args.takes_root / name / "trajectory" / "closed_loop_trajectory.csv"
        traj_m = trajectory_length_m(csv_path)
        dur_s = trajectory_duration_s(csv_path)
        n_q = count_questions(args.questions_root / name / "questions.yaml")
        entries.append({
            "take_name": name,
            "traj_len_m": round(traj_m, 2),
            "duration_s": round(dur_s, 2),
            "n_questions": n_q,
            "above_threshold": traj_m >= args.threshold_m,
        })
        if i % 50 == 0 or i == len(take_names):
            print(f"  scored {i}/{len(take_names)}", file=sys.stderr)

    entries.sort(key=lambda e: e["traj_len_m"], reverse=True)
    lengths = [e["traj_len_m"] for e in entries if e["traj_len_m"] > 0]
    above = [e for e in entries if e["above_threshold"]]

    # Distribution snapshot — drives the threshold-choice rationale in the paper.
    if lengths:
        sorted_l = sorted(lengths)
        def pct(p: float) -> float:
            idx = int(round(p * (len(sorted_l) - 1)))
            return sorted_l[idx]
        print("", file=sys.stderr)
        print(f"[mobility] trajectory length distribution (n={len(lengths)}):", file=sys.stderr)
        print(f"  min    = {min(lengths):8.1f} m", file=sys.stderr)
        print(f"  p25    = {pct(0.25):8.1f} m", file=sys.stderr)
        print(f"  median = {statistics.median(lengths):8.1f} m", file=sys.stderr)
        print(f"  p75    = {pct(0.75):8.1f} m", file=sys.stderr)
        print(f"  p90    = {pct(0.90):8.1f} m", file=sys.stderr)
        print(f"  p99    = {pct(0.99):8.1f} m", file=sys.stderr)
        print(f"  max    = {max(lengths):8.1f} m", file=sys.stderr)
        print(f"  above {args.threshold_m:g} m: {len(above)}/{len(entries)} takes "
              f"({100 * len(above) / max(len(entries), 1):.1f}%)", file=sys.stderr)
    else:
        print("[mobility] WARN: no usable trajectories found.", file=sys.stderr)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps({
        "threshold_m": args.threshold_m,
        "n_total": len(entries),
        "n_above_threshold": len(above),
        "takes": entries,
    }, indent=2))
    print(f"[mobility] wrote {args.out_manifest}", file=sys.stderr)

    args.out_take_list.write_text("".join(f"{e['take_name']}\n" for e in above))
    print(f"[mobility] wrote {args.out_take_list} "
          f"({len(above)} mobile takes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
