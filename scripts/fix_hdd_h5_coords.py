#!/usr/bin/env python3
"""Correct ``clip/lat,lng`` at rest in the HDD CLIP-L feature files.

Background (docs/HDD.md, 2026-08 postmortem). The 2026-07-05 extraction wrote
per-frame coordinates carrying a ~3 s video/GPS skew (the sidecar's warmup
correction was cancelled by ``extract.py``'s unconditional rebase). The
documented remedy was a read-time ``--realign-gps`` flag, but that path was
itself broken: ``video_start_skew`` called argless ``astimezone()``, which on a
UTC host inflated the skew by the capture's whole 7-8 h UTC offset, pushing
every frame outside the RTK span where ``np.interp`` silently clamps. Result:
tracks collapsed to a single coordinate with no error raised.

This script fixes the coordinates **at rest** instead, so no reader has to
remember a flag on the one variable the fixed-budget study is most sensitive
to. Embeddings are untouched — the GPS bug never affected them, so this is a
coords-only recompute (no GPU, minutes not hours).

Per file:
  * recompute ``clip/lat,lng`` from the raw RTK track via
    ``psm_extraction.io.hdd.realign_frames``, which now checks track coverage
    instead of letting ``np.interp`` clamp: frames outside the RTK span become
    NaN (``on_gap="nan"``) and are counted, never fabricated;
  * preserve the 2026-07-05 arrays under ``provenance/gps_20260705/{lat,lng}``
    — outside ``clip/`` so no downstream reader picks them up by accident;
  * stamp provenance attrs on the root group (fix commit, tool, skew applied,
    displacement summary);
  * emit a per-file sha256 over the corrected coordinates for the manifest.

Idempotent: a file already carrying ``gps_realign_fix`` is skipped unless
``--force`` is passed.

Usage:
    python scripts/fix_hdd_h5_coords.py --dry-run          # report, write nothing
    python scripts/fix_hdd_h5_coords.py                    # apply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "extraction"))

from psm_extraction.io.hdd import (  # noqa: E402
    load_rtk_trajectory,
    realign_frames,
    video_start_skew,
)

_FEATURES = "/checkpoint/dream/arjangt/video_retrieval/hdd/features"
_RAW = "/checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08"
_BACKUP_GROUP = "provenance/gps_20260705"
#: RTK bbox extent below which the vehicle simply never moved (parked/idling).
_STATIONARY_BBOX_M = 5.0
_M_PER_DEG_LAT = 110_540.0


def _fix_commit() -> str:
    """Commit that owns the current io/hdd.py, for the provenance stamp."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), "log", "-1", "--format=%H", "--",
             "extraction/psm_extraction/io/hdd.py"],
            capture_output=True, text=True, check=True,
        )
        sha = out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "-C", str(_REPO), "status", "--porcelain", "--",
         "extraction/psm_extraction/io/hdd.py"],
        capture_output=True, text=True,
    ).stdout.strip()
    return f"{sha}{'+dirty' if dirty else ''}"


def _displacement_m(lat0, lng0, lat1, lng1) -> np.ndarray:
    m_per_deg_lng = 111_320.0 * float(np.cos(np.deg2rad(float(np.mean(lat0)))))
    return np.hypot((lng1 - lng0) * m_per_deg_lng, (lat1 - lat0) * _M_PER_DEG_LAT)


def process(h5_path: Path, raw_root: Path, *, fix_commit: str,
            dry_run: bool, force: bool) -> dict:
    drive = h5_path.parent.name
    day = h5_path.parent.parent.name
    drive_dir = raw_root / day / drive
    rec: dict = {"drive": drive, "day": day}

    with h5py.File(h5_path, "r") as f:
        if "gps_realign_fix" in f.attrs and not force:
            rec["status"] = "already-fixed"
            rec["gps_realign_fix"] = str(f.attrs["gps_realign_fix"])
            return rec
        ts = f["clip/timestamps"][:].astype(np.float64)
        old_lat = f["clip/lat"][:].astype(np.float64)
        old_lng = f["clip/lng"][:].astype(np.float64)
        has_backup = _BACKUP_GROUP in f

    # ``on_gap="nan"`` rather than the default raise: a few drives have RTK
    # acquisition lagging the video by minutes, which is a real property of
    # the capture, not a clock bug. NaN keeps the frame axis aligned with the
    # embeddings while making uncovered frames impossible to use by accident.
    # The clamp the guard exists to prevent still cannot happen either way.
    coords = realign_frames(drive_dir, ts, on_gap="nan")
    if coords is None:
        rec["status"] = "no-raw-gps"
        return rec
    if coords.shape[0] != ts.size:
        rec["status"] = "length-mismatch"
        return rec
    new_lat, new_lng = coords[:, 0], coords[:, 1]
    covered = np.isfinite(new_lat) & np.isfinite(new_lng)
    rec["n_uncovered_frames"] = int((~covered).sum())

    traj = load_rtk_trajectory(drive_dir)
    skew = video_start_skew(drive_dir, traj)

    # Distinguish "the vehicle never moved" from "the track collapsed". Both
    # look identical in the H5; only the raw RTK extent tells them apart.
    if traj.bbox_extent_m < _STATIONARY_BBOX_M:
        rec.update(status="stationary-track", n_frames=int(ts.size),
                   skew_sec=round(float(skew), 3),
                   rtk_bbox_extent_m=round(float(traj.bbox_extent_m), 2))
        return rec
    if np.unique(np.round(new_lat[covered], 6)).size < 2:
        rec["status"] = "degenerate-track"
        return rec

    d = _displacement_m(old_lat[covered], old_lng[covered],
                        new_lat[covered], new_lng[covered])
    sha = hashlib.sha256(
        np.ascontiguousarray(np.stack([new_lat, new_lng], 1), dtype="<f8").tobytes()
    ).hexdigest()
    rec.update(
        n_frames=int(ts.size),
        skew_sec=round(float(skew), 3),
        displacement_m_median=round(float(np.median(d)), 2),
        displacement_m_p95=round(float(np.percentile(d, 95)), 2),
        displacement_m_max=round(float(d.max()), 2),
        coords_sha256=sha,
        status="would-fix" if dry_run else "fixed",
    )
    if dry_run:
        return rec

    with h5py.File(h5_path, "a") as f:
        if not has_backup:
            g = f.require_group(_BACKUP_GROUP)
            g.create_dataset("lat", data=old_lat)
            g.create_dataset("lng", data=old_lng)
            g.attrs["note"] = (
                "clip/lat,lng as written by the 2026-07-05 extraction, before "
                "the video/GPS skew correction. Kept for audit only — do not "
                "read these for analysis."
            )
        f["clip/lat"][...] = new_lat
        f["clip/lng"][...] = new_lng
        f.attrs["gps_realign_fix"] = f"extraction/psm_extraction/io/hdd.py@{fix_commit}"
        f.attrs["gps_realign_tool"] = "scripts/fix_hdd_h5_coords.py"
        f.attrs["gps_realign_skew_sec"] = float(skew)
        f.attrs["gps_realign_source"] = str(drive_dir)
        f.attrs["gps_coords_sha256"] = sha
        f.attrs["gps_uncovered_frames"] = int(rec["n_uncovered_frames"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-root", type=Path, default=Path(_FEATURES))
    ap.add_argument("--raw-root", type=Path, default=Path(_RAW))
    ap.add_argument("--manifest", type=Path,
                    default=_REPO / "captures/hdd/coords_fix_manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="re-fix files already carrying gps_realign_fix")
    args = ap.parse_args()

    files = sorted(args.features_root.glob("*/*/clip_l_features.h5"))
    if not files:
        print(f"no feature files under {args.features_root}", file=sys.stderr)
        return 1
    fix_commit = _fix_commit()
    print(f"[hdd-coords] {len(files)} files; fix commit {fix_commit}", file=sys.stderr)

    records, failures = [], []
    for i, p in enumerate(files, 1):
        try:
            rec = process(p, args.raw_root, fix_commit=fix_commit,
                          dry_run=args.dry_run, force=args.force)
        except Exception as exc:  # guard trips land here — report, never silently skip
            rec = {"drive": p.parent.name, "day": p.parent.parent.name,
                   "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        records.append(rec)
        if rec["status"] in {"error", "no-raw-gps", "length-mismatch",
                             "degenerate-track"}:
            failures.append(rec)
        if i % 25 == 0 or i == len(files):
            print(f"[hdd-coords] {i}/{len(files)}", file=sys.stderr)

    by_status: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    d = np.array([r["displacement_m_median"] for r in records
                  if "displacement_m_median" in r])
    manifest = {
        "fix_commit": fix_commit,
        "tool": "scripts/fix_hdd_h5_coords.py",
        "features_root": str(args.features_root),
        "raw_root": str(args.raw_root),
        "dry_run": args.dry_run,
        "n_files": len(files),
        "status_counts": by_status,
        "displacement_m_median_over_drives": (
            round(float(np.median(d)), 2) if d.size else None),
        "drives": records,
    }
    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"[hdd-coords] manifest -> {args.manifest}", file=sys.stderr)
    print(json.dumps(by_status, indent=2))
    for f in failures:
        print(f"  FAIL {f['day']}/{f['drive']}: "
              f"{f.get('error', f['status'])}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
