#!/usr/bin/env python3
"""Extract CLIP features from LookOut sequences for PSM.

LookOut ships pre-extracted 20-fps PNG frames at 512×512, an Aria MPS
SLAM trajectory, and per-frame depth maps. We bypass the orchestrator
because (a) frames already exist, no ffmpeg / VRS-read needed; (b) the
orchestrator's MP4/VRS-source paths don't fit the PNG-on-disk shape.
Writes the v2 H5 schema directly so all downstream eval (eval_lookback,
eval_psm_mllm, h3_acceptance) works unchanged.

Per sequence:
1. Read slam/closed_loop_trajectory.csv → device-clock timestamps + (tx, ty)
2. Project (tx, ty) → WGS84 lat/lng via per-session origin override
   (e.g. SanmateoDT2_Jan12 → San Mateo downtown). PSM math is invariant
   to origin; the choice only matters for map visualization.
3. Read rgb_data/rgb_info.pkl → per-frame (frame_idx, capture_ts_ns)
4. Subsample to --fps Hz (every ~20th frame at default 1 fps)
5. Embed kept frames with CLIPPyTorchRunner
6. Linearly interpolate trajectory lat/lng onto the kept frame timestamps
7. Write features.h5

Usage:
    GEMINI_API_KEY not needed (no MLLM calls). Run via SLURM h200 sbatch
    because CLIP-L embedding is GPU-bound for ~600+ frames per session.

    python scripts/extract_lookout_sessions.py \\
        --root /checkpoint/dream/arjangt/LookOut-unzipped/aria_navigation_blurred \\
        --out-root /checkpoint/dream/arjangt/video_retrieval/lookout \\
        --checkpoint laion/CLIP-ViT-L-14-laion2B-s32B-b82K \\
        --sequences Mainquad_jan10 Sanmateopark_garage_jan11
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np


_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "extraction"))

from psm_extraction.io.lookout import (  # noqa: E402
    discover_sequences,
    find_frame_dir,
    load_slam_trajectory,
)


_ENCODER_MAP = {
    "clipL": ("laion/CLIP-ViT-L-14-laion2B-s32B-b82K", 768, "clip_l_features.h5"),
    "bigG": ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 1280, "clip_bigg_features.h5"),
}


def _pick_subsample_indices(timestamps_s: np.ndarray, target_fps: float) -> np.ndarray:
    """Greedy: pick the first index whose ts is past the previous pick +
    1/target_fps. Robust to non-uniform sampling (Aria has occasional
    dropped frames). Returns int indices into `timestamps_s`."""
    if len(timestamps_s) == 0:
        return np.array([], dtype=np.int64)
    period = 1.0 / target_fps
    kept = [0]
    next_t = float(timestamps_s[0]) + period
    for i in range(1, len(timestamps_s)):
        if float(timestamps_s[i]) >= next_t:
            kept.append(i)
            next_t = float(timestamps_s[i]) + period
    return np.array(kept, dtype=np.int64)


def _interpolate_latlng(
    traj_ts: np.ndarray, traj_lat: np.ndarray, traj_lng: np.ndarray,
    frame_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of trajectory lat/lng onto frame timestamps,
    clamped at the trajectory edges. Both clocks are in seconds-from-
    session-start (we zero-shifted both in load_slam_trajectory and the
    rgb_info reader)."""
    lat = np.interp(frame_ts, traj_ts, traj_lat)
    lng = np.interp(frame_ts, traj_ts, traj_lng)
    return lat, lng


def extract_one(
    sequence_dir: Path,
    out_dir: Path,
    *,
    encoder: str,
    target_fps: float = 1.0,
    force: bool = False,
) -> Path:
    """Extract one LookOut sequence. Returns path to written H5."""
    seq_name = sequence_dir.name
    checkpoint, embed_dim, h5_name = _ENCODER_MAP[encoder]
    out_sess = out_dir / seq_name
    out_sess.mkdir(parents=True, exist_ok=True)
    out_h5 = out_sess / h5_name

    if out_h5.exists() and not force:
        print(f"[lookout] {seq_name}: {h5_name} already exists; skipping", file=sys.stderr)
        return out_h5

    print(f"[lookout] {seq_name} encoder={encoder} -> {out_h5}", file=sys.stderr)

    # Trajectory.
    traj = load_slam_trajectory(sequence_dir)
    print(f"  trajectory: {len(traj.timestamps)} samples, "
          f"{traj.trajectory_length_m:.0f}m path, {traj.bbox_extent_m:.0f}m bbox, "
          f"origin=({traj.origin_lat:.4f}, {traj.origin_lng:.4f})",
          file=sys.stderr)

    # Frame index.
    rgb_info_path = sequence_dir / "rgb_data" / "rgb_info.pkl"
    with rgb_info_path.open("rb") as f:
        rgb_info = pickle.load(f)
    frame_ids = np.array([int(r[0]) for r in rgb_info], dtype=np.int64)
    frame_ts_ns = np.array([int(r[1]) for r in rgb_info], dtype=np.int64)
    # Convert to seconds from the first frame.
    frame_ts_s = (frame_ts_ns - frame_ts_ns[0]) / 1e9
    print(f"  rgb_info: {len(frame_ids)} frames at "
          f"{(len(frame_ids)-1)/max(1e-9, frame_ts_s[-1]):.1f} fps native, "
          f"target fps={target_fps}", file=sys.stderr)

    # Subsample to target fps.
    kept_idx = _pick_subsample_indices(frame_ts_s, target_fps)
    kept_ts = frame_ts_s[kept_idx]
    kept_frame_ids = frame_ids[kept_idx]
    print(f"  subsampled to {len(kept_idx)} frames", file=sys.stderr)

    # Build PNG paths.
    frames_dir = find_frame_dir(sequence_dir)
    if frames_dir is None:
        raise SystemExit(f"no frames dir at {sequence_dir}/rgb_data/undistorted_aa")
    png_paths = [
        frames_dir / f"{fid}_undistorted_512_243.png" for fid in kept_frame_ids
    ]
    missing = [p for p in png_paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"{seq_name}: {len(missing)} of {len(png_paths)} expected PNGs missing "
            f"(first missing: {missing[0]})"
        )

    # Embed.
    from psm_extraction.models.clip_pytorch import CLIPPyTorchRunner
    print(f"  loading CLIP ({checkpoint})", file=sys.stderr)
    runner = CLIPPyTorchRunner(checkpoint=checkpoint)
    print(f"  embedding on {runner.backend}", file=sys.stderr)

    n = len(png_paths)
    def progress(done: int) -> None:
        print(f"  [embed] {done}/{n} ({100*done/n:.1f}%)", file=sys.stderr)
    emb = runner.embed_images(png_paths, progress=progress)
    if emb.shape != (n, embed_dim):
        raise RuntimeError(
            f"embedding shape {emb.shape} != expected ({n}, {embed_dim})"
        )

    # Interpolate trajectory onto kept frame timestamps.
    lat, lng = _interpolate_latlng(traj.timestamps, traj.lat, traj.lng, kept_ts)

    # Write H5.
    print(f"  writing {out_h5}", file=sys.stderr)
    with h5py.File(out_h5, "w") as h:
        h.attrs["timestamp_unit"] = "unix_seconds_f64"
        h.attrs["coord_system"] = "WGS84_degrees"
        h.attrs["schema_version"] = 2
        h.attrs["producer"] = "psm-extraction-lookout"
        h.attrs["producer_version"] = "0.1.0"
        h.attrs["source_video"] = str(sequence_dir)
        h.attrs["session_id"] = seq_name
        h.attrs["created_at_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )

        g = h.create_group("clip")
        g.create_dataset("embeddings", data=emb.astype(np.float32))
        g.create_dataset("lat", data=lat.astype(np.float64))
        g.create_dataset("lng", data=lng.astype(np.float64))
        g.create_dataset("timestamps", data=kept_ts.astype(np.float64))
        g.attrs["model"] = "openai/clip"
        g.attrs["checkpoint"] = checkpoint
        g.attrs["embedding_dim"] = int(embed_dim)
        g.attrs["sample_fps"] = float(target_fps)
        g.attrs["sampling"] = f"video_fps={target_fps}"
        g.attrs["preprocess"] = "clip_default(resize=224,center_crop=224,normalize=clip)"
        g.attrs["normalized"] = True
        g.attrs["interpolation"] = "linear,clipped_at_edges,from=lookout_slam"
        g.attrs["track_mode"] = "lookout_aria_slam"

    print(f"[lookout] done: {out_h5}", file=sys.stderr)
    return out_h5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--root", type=Path,
        default=Path("/checkpoint/dream/arjangt/LookOut-unzipped/aria_navigation_blurred"),
    )
    ap.add_argument(
        "--out-root", type=Path,
        default=Path("/checkpoint/dream/arjangt/video_retrieval/lookout"),
    )
    ap.add_argument(
        "--encoder", choices=list(_ENCODER_MAP.keys()), default="clipL",
    )
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if output H5 exists")
    ap.add_argument(
        "--sequences", nargs="*", default=None,
        help="specific session names to extract; default = all sessions "
             "with both slam/closed_loop_trajectory.csv and undistorted_aa/",
    )
    args = ap.parse_args()

    if args.sequences:
        sequence_dirs = [args.root / s for s in args.sequences]
        for d in sequence_dirs:
            if not d.is_dir():
                raise SystemExit(f"requested sequence not found: {d}")
    else:
        sequence_dirs = discover_sequences(args.root)

    print(f"[lookout] {len(sequence_dirs)} sequences to extract; "
          f"encoder={args.encoder}, fps={args.fps}", file=sys.stderr)

    for d in sequence_dirs:
        extract_one(
            d, args.out_root,
            encoder=args.encoder,
            target_fps=args.fps,
            force=args.force,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
