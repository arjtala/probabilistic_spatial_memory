#!/usr/bin/env python3
"""Extract CLIP/SigLIP features from Honda HDD drives for PSM.

HDD ships real RTK GPS, so per drive we:
1. Read general/csv/rtk_pos.csv -> real (lat, lng) (io.hdd, no projection)
2. Extract frames from camera/center/*.mp4 at --fps
3. Embed with the requested encoder
4. Interpolate GPS onto frame timestamps
5. Write features.h5 (v2 schema)

Reuses the standard psm_extraction pipeline (`python -m psm_extraction extract`
+ a gps.json sidecar), exactly like extract_sloper4d_sessions.py.

Embed-sanity gate (run FIRST, before committing GPU to 104 h):
    python scripts/extract_hdd_sessions.py --root <HDD> --sanity-only
Front-facing windshield frames (reflections, motion blur, sky-dominated) are a
different distribution from the wearable/LiDAR corpora. --sanity-only extracts
~50 frames from one drive, embeds them, and reports the pairwise cosine-
similarity spread. If frames collapse to near-identical embeddings (mean cosine
~1, tiny spread), F-HDD-3's cross-session retrieval signal would die -- catch it
at 50 frames, not after 104 h.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _h5_basename(checkpoint: str) -> str:
    if "ViT-L-14" in checkpoint:
        return "clip_l_features.h5"
    if "bigG-14" in checkpoint:
        return "clip_bigg_features.h5"
    if "siglip2-large" in checkpoint:
        return "siglip2_l_features.h5"
    slug = checkpoint.rsplit("/", 1)[-1].lower().replace("-", "_")
    return f"{slug}_features.h5"


def _frames_dirname(checkpoint: str) -> str:
    if "bigG-14" in checkpoint:
        return "frames_bigg"
    if "ViT-L-14" in checkpoint:
        return "frames_clipl"
    if "siglip" in checkpoint:
        return "frames_siglip"
    return "frames_hdd"


def _write_gps_sidecar(traj, path: Path) -> None:
    """Aria-style gps.json on the video clock (RTK t0 subtracted).

    HDD's video and RTK share the wall clock; the GPS first fix lands ~3 s
    after the video starts (GPS warmup), so subtracting t0 aligns GPS to
    video PTS within a few seconds -- well inside an r10 cell at road speed.
    """
    t0 = float(traj.timestamps[0])
    samples = [{
        "timestamp": float(t) - t0,
        "latitude": float(la),
        "longitude": float(lo),
        "accuracy": 1.0,  # RTK is sub-metre
    } for t, la, lo in zip(traj.timestamps, traj.lat, traj.lng)]
    path.write_text(json.dumps([{"stream_id": "hdd-rtk", "samples": samples}]))


def _run_extract(video: Path, out_h5: Path, sidecar: Path, frames_dir: Path,
                 name: str, family: str, checkpoint: str, fps: float) -> int:
    cmd = [
        sys.executable, "-m", "psm_extraction", "extract",
        "--video", str(video),
        "--output", str(out_h5),
        "--models", family,
        "--checkpoint", f"{family}:{checkpoint}",
        "--sample-fps", str(fps),
        "--segment-sec", "1",
        "--session-id", name,
        "--gps-json", str(sidecar),
        "--frames-dir", str(frames_dir),
        "--keep-frames",
    ]
    print(f"  running: {' '.join(cmd[-14:])}", file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode


def _normalize_group(out_h5: Path) -> None:
    try:
        import h5py
        with h5py.File(out_h5, "a") as h:
            for g in list(h.keys()):
                if g != "clip" and g not in ("imu", "gps") and "embeddings" in h[g]:
                    h.move(g, "clip")
                    break
    except Exception as e:  # noqa: BLE001
        print(f"  WARN: could not normalize H5 group: {e}", file=sys.stderr)


def _embed_sanity(out_h5: Path) -> dict:
    """Report pairwise cosine-similarity spread over the drive's embeddings."""
    import h5py
    with h5py.File(out_h5, "r") as h:
        grp = "clip" if "clip" in h else next(iter(h))
        emb = np.asarray(h[grp]["embeddings"], dtype=np.float64)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sims = emb @ emb.T
    iu = np.triu_indices(sims.shape[0], k=1)
    off = sims[iu]
    return {
        "n_frames": int(emb.shape[0]),
        "cos_mean": float(off.mean()),
        "cos_std": float(off.std()),
        "cos_p05": float(np.percentile(off, 5)),
        "cos_p95": float(np.percentile(off, 95)),
        "cos_min": float(off.min()),
        "cos_max": float(off.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", type=Path, required=True, help="HDD release root")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--model-family", choices=["clip", "siglip"], default="clip")
    ap.add_argument("--checkpoint", default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    ap.add_argument("--drives", nargs="*", default=None,
                    help="specific drive_ids to extract (default: all)")
    ap.add_argument("--drive-index", type=int, default=None,
                    help="extract only the Nth discovered drive (for SLURM "
                         "arrays); applied after --drives filtering")
    ap.add_argument("--sanity-only", action="store_true",
                    help="extract ~50 frames from ONE drive, report cosine "
                         "spread, and exit (embed-sanity gate)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path("extraction").resolve()))
    from psm_extraction.io.hdd import (  # noqa: E402
        discover_drives, find_video, load_rtk_trajectory,
    )

    out_root = args.out_root or args.root
    drives = discover_drives(args.root)
    if args.drives:
        drives = [d for d in drives if d.name in args.drives]
    if args.drive_index is not None:
        if args.drive_index >= len(drives):
            print(f"[hdd] drive-index {args.drive_index} >= {len(drives)} "
                  f"drives; nothing to do", file=sys.stderr)
            return 0
        drives = [drives[args.drive_index]]
    if not drives:
        print(f"No HDD drives found under {args.root}", file=sys.stderr)
        return 1

    if args.sanity_only:
        drives = drives[:1]
        print(f"[hdd] embed-sanity on {drives[0].name}", file=sys.stderr)

    h5_basename = _h5_basename(args.checkpoint)
    n_done = 0
    for drive_dir in drives:
        name = drive_dir.name
        out_dir = out_root / name
        out_h5 = out_dir / (("sanity_" if args.sanity_only else "") + h5_basename)
        if out_h5.exists() and not args.sanity_only:
            print(f"[hdd] {name}: already extracted, skipping", file=sys.stderr)
            n_done += 1
            continue

        try:
            traj = load_rtk_trajectory(drive_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"[hdd] {name}: skip ({e})", file=sys.stderr)
            continue
        video = find_video(drive_dir)
        if video is None:
            print(f"[hdd] {name}: no center video, skip", file=sys.stderr)
            continue

        duration = float(traj.timestamps[-1] - traj.timestamps[0])
        fps = args.fps
        if args.sanity_only:  # aim for ~50 frames
            fps = max(0.02, min(1.0, 50.0 / max(duration, 1.0)))

        out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = video.parent / "gps.json"
        _write_gps_sidecar(traj, sidecar)
        frames_dir = out_dir / (("sanity_" if args.sanity_only else "")
                                + _frames_dirname(args.checkpoint))
        rc = _run_extract(video, out_h5, sidecar, frames_dir, name,
                          args.model_family, args.checkpoint, fps)
        sidecar.unlink(missing_ok=True)
        if rc != 0:
            print(f"[hdd] {name}: extraction failed (rc={rc})", file=sys.stderr)
            continue
        _normalize_group(out_h5)

        if args.sanity_only:
            stats = _embed_sanity(out_h5)
            print("\n=== EMBED-SANITY (windshield frames) ===")
            print(json.dumps(stats, indent=2))
            degenerate = stats["cos_mean"] > 0.95 and stats["cos_std"] < 0.03
            print(f"\nVERDICT: {'DEGENERATE -- frames collapse; F-HDD-3 at risk' if degenerate else 'OK -- embeddings show usable spread'}")
            return 1 if degenerate else 0

        n_done += 1
        print(f"[hdd] ✓ {out_h5}", file=sys.stderr)

    print(f"[hdd] done: {n_done}/{len(drives)} drives", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
