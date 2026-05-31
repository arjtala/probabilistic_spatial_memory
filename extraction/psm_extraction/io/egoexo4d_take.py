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
  - frame_aligned_videos/aria01_214-1.mp4  as the source video (much
    faster to decode than the full VRS, and time-aligned to the
    trajectory CSV by construction).
  - trajectory/closed_loop_trajectory.csv  for SLAM-projected
    (lat, lng) — Ego-Exo4D is all indoor so there's no GPS to prefer.

This module's only public surface is `is_egoexo4d_take(dir)` and
`load_egoexo4d_trajectory(dir)`. The orchestrator detects + dispatches;
the trajectory parsing reuses `_read_slam_trajectory` and
`_project_xy_to_latlng` from `aria_vrs` so there's no second copy of
the schema knowledge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .aria_vrs import (
    _interpolate_track_to_frames,
    _project_xy_to_latlng,
    _read_slam_trajectory,
)

_EGO_RGB_MP4 = "frame_aligned_videos/aria01_214-1.mp4"
_TRAJ_CSV = "trajectory/closed_loop_trajectory.csv"


def is_egoexo4d_take(take_dir: Path) -> bool:
    """True when `take_dir` looks like an Ego-Exo4D take directory.

    Detection requires BOTH the ego RGB MP4 and the closed-loop SLAM
    trajectory — partial layouts (e.g. someone copied just the MP4)
    aren't routed through this path, which keeps the autodetect strict
    and predictable.
    """
    if not take_dir.is_dir():
        return False
    return (take_dir / _EGO_RGB_MP4).exists() and (take_dir / _TRAJ_CSV).exists()


def ego_rgb_mp4(take_dir: Path) -> Path:
    return take_dir / _EGO_RGB_MP4


def load_egoexo4d_trajectory(
    take_dir: Path,
    frame_ts_s: np.ndarray,
    *,
    origin_lat: float = 0.0,
    origin_lng: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read SLAM trajectory + project onto frame timestamps -> (lats, lngs).

    `frame_ts_s` must be relative-seconds since video start (i.e.
    `arange(N) / sample_fps` — what extract.py produces for MP4
    inputs). The trajectory CSV's `tracking_timestamp_us` is device-
    clock microseconds, also relative to take start, so the two clocks
    line up directly (frame_aligned_videos is "frame aligned" to the
    trajectory by construction).

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
    fr_tx, fr_ty = _interpolate_track_to_frames(frame_ts_s, track_ts_s, tx, ty)
    lats, lngs = _project_xy_to_latlng(
        fr_tx, fr_ty, origin_lat=origin_lat, origin_lng=origin_lng,
    )
    return lats, lngs
