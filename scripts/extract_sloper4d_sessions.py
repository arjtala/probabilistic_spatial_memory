#!/usr/bin/env python3
"""Extract CLIP-L features from SLOPER4D sequences for PSM.

Per sequence:
1. Read lidar_trajectory.txt → timestamps + XYZ → project to WGS84
2. Extract frames from rgb_data/*.mp4 at --fps
3. Embed frames with CLIP-L
4. Interpolate projected lat/lng onto frame timestamps
5. Write features.h5 (v2 schema)

The heavy lifting reuses the existing psm_extraction pipeline;
this script just provides the SLOPER4D-specific data loading.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", type=Path, required=True,
                     help="SLOPER4D dataset root")
    ap.add_argument("--out-root", type=Path, default=None,
                     help="Output root (default: same as --root)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--origin-lat", type=float, default=24.4381)
    ap.add_argument("--origin-lng", type=float, default=118.0992)
    ap.add_argument("--checkpoint", type=str,
                     default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    ap.add_argument("--sequences", nargs="*", default=None,
                     help="Specific sequence names to extract (default: all)")
    args = ap.parse_args()

    out_root = args.out_root or args.root

    # Import after argparse so --help is fast
    from psm_extraction.io.sloper4d import (
        discover_sequences,
        find_video,
        load_lidar_trajectory,
    )

    # Discover sequences
    all_seqs = discover_sequences(args.root)
    if args.sequences:
        all_seqs = [s for s in all_seqs if s.name in args.sequences]

    if not all_seqs:
        print(f"No valid sequences found in {args.root}", file=sys.stderr)
        return 1

    print(f"[sloper4d] {len(all_seqs)} sequences to extract", file=sys.stderr)

    for seq_dir in all_seqs:
        name = seq_dir.name
        out_dir = out_root / name
        out_h5 = out_dir / "clip_l_features.h5"

        if out_h5.exists():
            print(f"[sloper4d] {name}: already extracted, skipping",
                  file=sys.stderr)
            continue

        print(f"\n[sloper4d] ━━━ {name} ━━━", file=sys.stderr)

        # Load trajectory
        traj = load_lidar_trajectory(
            seq_dir,
            origin_lat=args.origin_lat,
            origin_lng=args.origin_lng,
        )
        print(f"  trajectory: {traj.trajectory_length_m:.0f}m, "
              f"{traj.bbox_extent_m:.0f}m bbox, "
              f"{len(traj.timestamps)} LiDAR frames",
              file=sys.stderr)

        # Find video
        video = find_video(seq_dir)
        if video is None:
            print(f"  WARN: no video found, skipping", file=sys.stderr)
            continue

        # Use the existing extraction pipeline
        # Build a synthetic GPS sidecar from the projected trajectory
        # so the orchestrator can interpolate onto frame timestamps
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write a temporary GPS JSON sidecar in Aria format so the
        # existing orchestrator's align.py can consume it
        _write_gps_sidecar(traj, out_dir / "gps.json")

        # Run extraction via the standard CLI
        import subprocess
        cmd = [
            sys.executable, "-m", "psm_extraction", "extract",
            "--video", str(video),
            "--output", str(out_h5),
            "--models", "clip",
            "--checkpoint", f"clip:{args.checkpoint}",
            "--sample-fps", str(args.fps),
            "--segment-sec", "1",
            "--session-id", name,
        ]
        print(f"  running: {' '.join(cmd[-8:])}", file=sys.stderr)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"  ERR: extraction failed (rc={result.returncode})",
                  file=sys.stderr)
            continue

        # Clean up temporary sidecar
        (out_dir / "gps.json").unlink(missing_ok=True)

        print(f"  ✓ {out_h5}", file=sys.stderr)

    return 0


def _write_gps_sidecar(traj, path: Path) -> None:
    """Write an Aria-style gps.json from a SLOPER4D trajectory.

    The orchestrator's json_sidecar.py reader expects:
    [{stream_id, samples: [{timestamp, latitude, longitude, ...}, ...]}]

    We write the projected lat/lng with the LiDAR timestamps (already
    in seconds). The orchestrator's align step will interpolate these
    onto frame timestamps.
    """
    import json

    samples = []
    for i in range(len(traj.timestamps)):
        samples.append({
            "timestamp": float(traj.timestamps[i]),
            "latitude": float(traj.lat[i]),
            "longitude": float(traj.lng[i]),
            "accuracy": 1.0,  # LiDAR-SLAM is sub-metre
        })

    sidecar = [{
        "stream_id": "sloper4d-lidar",
        "samples": samples,
    }]

    with open(path, "w") as f:
        json.dump(sidecar, f)


if __name__ == "__main__":
    raise SystemExit(main())
