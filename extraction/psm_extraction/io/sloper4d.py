"""SLOPER4D dataset reader.

Reads per-sequence data from SLOPER4D (Dai et al., CVPR 2023):
  - LiDAR-SLAM global trajectory (lidar_trajectory.txt)
  - Egocentric RGB video (rgb_data/*.mp4)

Global XYZ positions from the LiDAR-SLAM trajectory are projected to
WGS84 lat/lng via a configurable fake origin (default: Xiamen University,
Fujian, China — where the dataset was captured). The projection is
locally accurate for the 200m-1.3km trajectory scales in the dataset.

Data structure per sequence (from the SLOPER4D README):

    root_folder/
    ├── lidar_data/
    │   ├── lidar_frames_rot/        # *.pcd point clouds (not used here)
    │   ├── lidar_trajectory.txt     # framenum X Y Z qx qy qz qw timestamp
    │   └── tracking_traj.txt        # X Y Z framenum timestamp
    ├── rgb_data/
    │   └── *.mp4                    # egocentric video
    ├── *_labels.pkl                 # 2D/3D pose labels (not used here)
    └── dataset_params.json          # meta info

Reference:
    @InProceedings{Dai_2023_CVPR,
        author  = {Dai, Yudi and Lin, Yitai and Lin, Xiping and Wen, Chenglu
                   and Xu, Lan and Yi, Hongwei and Shen, Siqi and Ma, Yuexin
                   and Wang, Cheng},
        title   = {SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose
                   Estimation in Urban Environments},
        booktitle = {CVPR},
        year    = {2023},
        pages   = {682--692},
    }
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Xiamen University, Siming Campus — where the dataset was captured.
# Used as the WGS84 origin for projecting LiDAR-SLAM metric XYZ to
# lat/lng. The choice only matters for H3 cell IDs (which are globally
# unique); relative inter-cell distances are preserved regardless of
# the origin.
DEFAULT_ORIGIN_LAT = 24.4381
DEFAULT_ORIGIN_LNG = 118.0992


@dataclass(frozen=True)
class SLOPER4DTrajectory:
    """Per-sequence LiDAR-SLAM trajectory projected to WGS84."""

    timestamps: np.ndarray   # float64, seconds (from LiDAR clock)
    lat: np.ndarray          # float64, degrees (projected from XYZ)
    lng: np.ndarray          # float64, degrees (projected from XYZ)
    x: np.ndarray            # float64, metres (raw global X)
    y: np.ndarray            # float64, metres (raw global Y)
    z: np.ndarray            # float64, metres (raw global Z)
    sequence_name: str
    origin_lat: float
    origin_lng: float
    trajectory_length_m: float
    bbox_extent_m: float


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Return (m_per_deg_lat, m_per_deg_lng) at the given latitude."""
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(lat_deg))
    return m_per_deg_lat, m_per_deg_lng


def load_lidar_trajectory(
    sequence_dir: Path,
    *,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
    origin_lng: float = DEFAULT_ORIGIN_LNG,
) -> SLOPER4DTrajectory:
    """Load and project a SLOPER4D LiDAR trajectory to WGS84.

    Reads ``lidar_data/lidar_trajectory.txt`` which has the format::

        framenum X Y Z qx qy qz qw timestamp

    (space-separated, one row per LiDAR frame at ~20 Hz).

    Args:
        sequence_dir: Path to a SLOPER4D sequence root (contains
            ``lidar_data/``, ``rgb_data/``, etc.).
        origin_lat: WGS84 latitude for the XYZ origin.
        origin_lng: WGS84 longitude for the XYZ origin.

    Returns:
        A ``SLOPER4DTrajectory`` with projected lat/lng and raw XYZ.

    Raises:
        FileNotFoundError: if the trajectory file is missing.
        ValueError: if the file is empty or unparseable.
    """
    traj_path = sequence_dir / "lidar_data" / "lidar_trajectory.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"No trajectory file at {traj_path}")

    # framenum X Y Z qx qy qz qw timestamp
    raw = np.loadtxt(traj_path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[0] == 0 or raw.shape[1] < 9:
        raise ValueError(
            f"Expected ≥9 columns (framenum X Y Z qx qy qz qw timestamp), "
            f"got shape {raw.shape} in {traj_path}"
        )

    x = raw[:, 1]
    y = raw[:, 2]
    z = raw[:, 3]
    timestamps = raw[:, 8]

    # Sort by timestamp (should already be sorted, but be safe)
    order = np.argsort(timestamps)
    x, y, z, timestamps = x[order], y[order], z[order], timestamps[order]

    # Project metric XYZ to WGS84 degrees.
    # SLOPER4D's global frame uses XYZ in metres with an arbitrary origin;
    # we map X → longitude offset, Y → latitude offset (the conventional
    # ENU-like mapping). Z is elevation (not used for H3).
    m_per_deg_lat, m_per_deg_lng = _meters_per_degree(origin_lat)

    lat = origin_lat + y / m_per_deg_lat
    lng = origin_lng + x / m_per_deg_lng

    # Trajectory statistics
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    trajectory_length_m = float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))
    bbox_extent_m = float(max(
        x.max() - x.min(),
        y.max() - y.min(),
    ))

    return SLOPER4DTrajectory(
        timestamps=timestamps,
        lat=lat,
        lng=lng,
        x=x,
        y=y,
        z=z,
        sequence_name=sequence_dir.name,
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        trajectory_length_m=trajectory_length_m,
        bbox_extent_m=bbox_extent_m,
    )


def find_video(sequence_dir: Path) -> Path | None:
    """Find the egocentric RGB video in a SLOPER4D sequence.

    Returns the first .mp4 under ``rgb_data/``, or None if missing.
    """
    rgb_dir = sequence_dir / "rgb_data"
    if not rgb_dir.exists():
        return None
    videos = sorted(rgb_dir.glob("*.mp4"))
    return videos[0] if videos else None


def discover_sequences(root: Path) -> list[Path]:
    """Discover all valid SLOPER4D sequences under a root directory.

    A valid sequence has both ``lidar_data/lidar_trajectory.txt`` and
    at least one ``.mp4`` in ``rgb_data/``.
    """
    sequences = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        traj = d / "lidar_data" / "lidar_trajectory.txt"
        if not traj.exists():
            continue
        if find_video(d) is None:
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

    Useful for confirming trajectory extents before running extraction.
    """
    results = []
    for seq_dir in discover_sequences(root):
        try:
            traj = load_lidar_trajectory(
                seq_dir, origin_lat=origin_lat, origin_lng=origin_lng
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
