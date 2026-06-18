"""LookOut / Aria Navigation Dataset reader.

LookOut (Pan et al., ICCV 2025) is captured with Project Aria glasses
across diverse outdoor environments and shipped as a privacy-blurred
release with pre-extracted PNG frames and MPS SLAM trajectories. Same
schema as Nymeria's `recording_head/mps/slam/closed_loop_trajectory.csv`
so we reuse the existing Aria MPS trajectory parser; the only new
shape is the pre-extracted-PNG frames layout.

Per-sequence layout (from the LookOut README):

    aria_navigation_blurred/<session>/
    ├── rgb_data/
    │   ├── device_T_rgbcam.npy
    │   ├── pix_T_rgbcam_ver_aa.npy
    │   ├── rgb_info.pkl                          # list[(frame_idx, capture_timestamp_ns)]
    │   └── undistorted_aa/
    │       ├── <N>_undistorted_512_243.png             # 512×512 RGB
    │       ├── <N>_undistorted_512_243_depth.png       # depth visualisation
    │       └── <N>_undistorted_512_243_depth_metric.npy
    ├── slam/
    │   ├── closed_loop_trajectory.csv            # MPS schema, same as Nymeria
    │   ├── open_loop_trajectory.csv
    │   ├── online_calibration.jsonl
    │   ├── semidense_{points,observations}.csv.gz
    │   └── summary.json
    └── xyz_world.npy

Trajectory `(tx_world_device, ty_world_device)` is in metres in an
arbitrary world frame (Aria MPS post-loop-closure convention). We
project to WGS84 via a flat-earth fake origin — same convention as
SLOPER4D + Nymeria-SLAM. Default origin near Stanford (the
collection-team home institution) so the cells land on a real map
position for visualisation; the choice doesn't affect PSM math.

Reference:
    @InProceedings{Pan_2025_ICCV,
      author    = {Pan, Boxiao and Harley, Adam W. and Engelmann, Francis
                   and Liu, C. Karen and Guibas, Leonidas J.},
      title     = {LookOut: Real-World Humanoid Egocentric Navigation},
      booktitle = {ICCV},
      year      = {2025},
      pages     = {24977--24988},
    }
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from psm_extraction.io.aria_vrs import _read_slam_trajectory

# Stanford University main quad — collection-team home institution.
# Some LookOut sessions are San Mateo / other Bay Area locations, but
# the origin only matters for visualisation; PSM is invariant.
DEFAULT_ORIGIN_LAT = 37.4275
DEFAULT_ORIGIN_LNG = -122.1697


@dataclass(frozen=True)
class LookOutTrajectory:
    """Per-sequence Aria MPS trajectory projected to WGS84."""

    timestamps: np.ndarray   # float64, seconds (device clock)
    lat: np.ndarray          # float64, degrees (projected from tx/ty)
    lng: np.ndarray          # float64, degrees (projected from tx/ty)
    tx: np.ndarray           # float64, metres (raw MPS tx_world_device)
    ty: np.ndarray           # float64, metres (raw MPS ty_world_device)
    sequence_name: str
    origin_lat: float
    origin_lng: float
    trajectory_length_m: float
    bbox_extent_m: float


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(lat_deg))
    return m_per_deg_lat, m_per_deg_lng


def load_slam_trajectory(
    sequence_dir: Path,
    *,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
    origin_lng: float = DEFAULT_ORIGIN_LNG,
) -> LookOutTrajectory:
    """Load and project a LookOut SLAM trajectory to WGS84.

    Reuses the Aria MPS parser from `aria_vrs._read_slam_trajectory`
    (LookOut ships the standard MPS schema). Returns a typed
    LookOutTrajectory the orchestrator + sidecar wrappers can consume.
    """
    csv_path = sequence_dir / "slam" / "closed_loop_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No trajectory file at {csv_path}")
    parsed = _read_slam_trajectory(csv_path)
    if parsed is None:
        raise ValueError(f"Could not parse {csv_path}")

    timestamps, tx, ty = parsed
    # Zero-shift the device clock so all downstream tools see a
    # video-clock timeline starting at 0 (same convention we settled
    # on for SLOPER4D after the seq008 LiDAR-clock bug).
    t0 = float(timestamps[0])
    timestamps = timestamps - t0

    m_per_deg_lat, m_per_deg_lng = _meters_per_degree(origin_lat)
    lat = origin_lat + ty / m_per_deg_lat
    lng = origin_lng + tx / m_per_deg_lng

    dx = np.diff(tx)
    dy = np.diff(ty)
    trajectory_length_m = float(np.sum(np.sqrt(dx**2 + dy**2)))
    bbox_extent_m = float(max(tx.max() - tx.min(), ty.max() - ty.min()))

    return LookOutTrajectory(
        timestamps=timestamps,
        lat=lat,
        lng=lng,
        tx=tx,
        ty=ty,
        sequence_name=sequence_dir.name,
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        trajectory_length_m=trajectory_length_m,
        bbox_extent_m=bbox_extent_m,
    )


def load_frame_index(sequence_dir: Path) -> list[tuple[int, int]]:
    """Read `rgb_data/rgb_info.pkl` → list of `(frame_idx, capture_ts_ns)`.

    `rgb_info.pkl` is a small pickle of (frame_number, capture_timestamp_ns)
    tuples — one entry per PNG in `undistorted_aa/`. Used to map a
    target timestamp to the nearest available frame index.
    """
    pkl_path = sequence_dir / "rgb_data" / "rgb_info.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"No rgb_info.pkl at {pkl_path}")
    with pkl_path.open("rb") as f:
        records = pickle.load(f)
    # Be tolerant of whatever exact tuple layout the file ships; the
    # README says (frame_number, capture_timestamp_ns) but variants exist.
    out: list[tuple[int, int]] = []
    for r in records:
        if isinstance(r, tuple) and len(r) >= 2:
            out.append((int(r[0]), int(r[1])))
    if not out:
        raise ValueError(f"rgb_info.pkl at {pkl_path} parsed to empty list")
    return out


def find_frame_dir(sequence_dir: Path) -> Path | None:
    """Locate the pre-extracted RGB frames dir, or None.

    LookOut ships PNGs at `rgb_data/undistorted_aa/` named
    `<N>_undistorted_512_243.png`.
    """
    cand = sequence_dir / "rgb_data" / "undistorted_aa"
    return cand if cand.is_dir() else None


def discover_sequences(root: Path) -> list[Path]:
    """Discover all valid LookOut sequences under a root directory.

    A valid sequence has both `slam/closed_loop_trajectory.csv` and
    a populated `rgb_data/undistorted_aa/` PNG dir.

    `root` should be the parent that contains the per-session dirs.
    LookOut's release zip extracts to `aria_navigation_blurred/<session>/`,
    so callers typically pass `<extracted>/aria_navigation_blurred`.
    """
    sequences = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        traj = d / "slam" / "closed_loop_trajectory.csv"
        if not traj.exists():
            continue
        if find_frame_dir(d) is None:
            continue
        sequences.append(d)
    return sequences


def probe_sequences(
    root: Path,
    *,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
    origin_lng: float = DEFAULT_ORIGIN_LNG,
) -> list[dict]:
    """Probe all sequences and return summary statistics.

    Same shape as SLOPER4D's probe — useful for confirming trajectory
    extents before committing to extraction on a multi-hundred-GB
    release.
    """
    results = []
    for seq_dir in discover_sequences(root):
        try:
            traj = load_slam_trajectory(
                seq_dir, origin_lat=origin_lat, origin_lng=origin_lng,
            )
            results.append({
                "sequence": traj.sequence_name,
                "frames": len(traj.timestamps),
                "duration_s": float(traj.timestamps[-1] - traj.timestamps[0]),
                "trajectory_m": round(traj.trajectory_length_m, 1),
                "bbox_extent_m": round(traj.bbox_extent_m, 1),
                "origin": f"{traj.origin_lat:.4f},{traj.origin_lng:.4f}",
            })
        except (FileNotFoundError, ValueError) as e:
            results.append({
                "sequence": seq_dir.name,
                "error": str(e),
            })
    return results
