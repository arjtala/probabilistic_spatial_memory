"""Ego-Exo4D take-directory helpers.

Ego-Exo4D ships each "take" as a directory with:

  <take_dir>/
    aria01.vrs                                   (full ego-camera VRS)
    aria01_noimagestreams.vrs                    (image-stripped twin)
    frame_aligned_videos/
      aria01_214-1.mp4                           ego RGB stream, pre-extracted
      aria01_211-1.mp4                           ego SLAM camera (grayscale)
      aria01_1201-{1,2}.mp4                      ego SLAM secondary
      cam0{1..4}.mp4                             exo GoPros
    trajectory/
      closed_loop_trajectory.csv                 MPS SLAM (same schema as Aria Gen 2)
      open_loop_trajectory.csv
      ...

For PSM look-back retrieval against the ego perspective, we use:
  - `frame_aligned_videos/aria*_214-1.mp4` as the source video.
    Stream ID `214-1` is Aria's primary RGB camera (same on Gen 1 and
    Gen 2); the `aria*` prefix varies because different captures use
    different aria devices (aria01..aria06+ seen across the val set).
  - `trajectory/closed_loop_trajectory.csv` for SLAM-projected
    (lat, lng) — Ego-Exo4D is all indoor so there's no GPS to prefer.

This module's only public surface is `is_egoexo4d_take(dir)`,
`ego_rgb_mp4(dir)`, and `load_egoexo4d_trajectory(dir)`. The
orchestrator detects + dispatches; the trajectory parsing reuses
`_read_slam_trajectory` and `_project_xy_to_latlng` from `aria_vrs`
so there's no second copy of the schema knowledge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .aria_vrs import (
    _interpolate_track_to_frames,
    _project_xy_to_latlng,
    _read_slam_trajectory,
)

_EGO_RGB_GLOB = "frame_aligned_videos/aria*_214-1.mp4"
_TRAJ_CSV = "trajectory/closed_loop_trajectory.csv"


def _find_ego_rgb_mp4(take_dir: Path) -> Path | None:
    """Locate the ego RGB MP4. None when no candidate exists.

    Glob matches all aria<N>_214-1.mp4 variants. In the rare case
    multiple match (multi-Aria recordings), pick the lexicographically
    first — Ego-Exo4D conventionally numbers the wearer's device
    lowest, and the atomic_descriptions annotations are tied to that
    primary ego perspective.
    """
    matches = sorted(take_dir.glob(_EGO_RGB_GLOB))
    return matches[0] if matches else None


def is_egoexo4d_take(take_dir: Path) -> bool:
    """True when `take_dir` looks like an Ego-Exo4D take directory.

    Detection requires BOTH a usable ego RGB MP4 and the closed-loop
    SLAM trajectory — partial layouts (e.g. someone copied just one)
    aren't routed through this path, which keeps the autodetect strict
    and predictable.
    """
    if not take_dir.is_dir():
        return False
    return _find_ego_rgb_mp4(take_dir) is not None and (take_dir / _TRAJ_CSV).exists()


def ego_rgb_mp4(take_dir: Path) -> Path:
    """Return the ego RGB MP4 for `take_dir`. Raises if absent.

    Callers that want a tri-state should use `is_egoexo4d_take` first;
    raising here keeps the orchestrator from having to handle a None
    after it's already committed to the egoexo4d branch.
    """
    p = _find_ego_rgb_mp4(take_dir)
    if p is None:
        raise FileNotFoundError(
            f"no ego RGB MP4 under {take_dir}/frame_aligned_videos/ "
            f"(glob: aria*_214-1.mp4)"
        )
    return p


def load_egoexo4d_trajectory(
    take_dir: Path,
    frame_ts_s: np.ndarray,
    *,
    origin_lat: float = 0.0,
    origin_lng: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read SLAM trajectory + project onto frame timestamps -> (lats, lngs).

    `frame_ts_s` is relative-seconds since video start (`arange(N) /
    sample_fps` from extract.py). The trajectory CSV's
    `tracking_timestamp_us` is device-clock microseconds starting from
    boot (~10^8 us into the recording for typical takes), NOT zero —
    `frame_aligned_videos/aria*_214-1.mp4` is the same image stream
    re-muxed to start at MP4 wall-time 0.

    We re-base the trajectory clock by subtracting its first timestamp,
    matching what `aria_vrs.read_vrs_session` does for VRS-side
    timestamps. Without this every frame interp-clamps to the first
    trajectory row, collapsing lat/lng to a single (constant) point and
    making PSM's spatial axis degenerate for every Ego-Exo4D take.

    Returns (None, None) when the CSV is missing or malformed; caller
    should fall back to synthetic snake-grid.
    """
    csv = take_dir / _TRAJ_CSV
    if not csv.exists():
        return None, None
    parsed = _read_slam_trajectory(csv)
    if parsed is None:
        return None, None
    track_ts_s, tx, ty = parsed
    if track_ts_s.size:
        # Re-base to start at 0 so it shares an origin with the MP4's
        # arange/fps timeline. Without this `np.interp` clamps every
        # frame (all < track_ts_s[0]) to the trajectory's first row.
        track_ts_s = track_ts_s - track_ts_s[0]
    fr_tx, fr_ty = _interpolate_track_to_frames(frame_ts_s, track_ts_s, tx, ty)
    lats, lngs = _project_xy_to_latlng(
        fr_tx, fr_ty, origin_lat=origin_lat, origin_lng=origin_lng,
    )
    return lats, lngs
