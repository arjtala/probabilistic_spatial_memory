"""HDD (Honda Research Institute Driving Dataset) IO for PSM extraction.

HDD ships real RTK GPS, so unlike SLOPER4D/Nymeria there is no fake-origin
projection: we read (lat, lng) straight from ``rtk_pos.csv``. This module
mirrors the SLOPER4D reader's shape so the standard extraction pipeline
(``python -m psm_extraction extract`` + a ``gps.json`` sidecar) applies
unchanged.

Layout (release_2019_07_08)::

    <root>/
    └── 2017_MM_DD_ITS1/                 # drive-day
        └── <YYYYMMDDHHMM>/              # a single drive
            ├── general/csv/rtk_pos.csv  # unix_ts, iso_ts, lat, lng  (see note)
            │                 vel.csv     # unix_ts, iso_ts, speed
            └── camera/center/*.mp4       # front-facing center camera

Data gotcha: ``rtk_pos.csv``'s header reads ``...,lng,lat`` but the columns
are actually ``lat,lng`` (col 3 = 37.x Palo Alto latitude, col 4 = -122.x
longitude). We read col 3 -> lat, col 4 -> lng and range-guard to SF Bay.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# SF Bay (incl. Santa Cruz ~36.97) guard against a bad file / swapped columns.
_LAT_RANGE = (36.5, 38.5)
_LNG_RANGE = (-123.5, -120.5)


@dataclass(frozen=True)
class HDDTrajectory:
    """RTK GPS trajectory for one HDD drive (real WGS84, no projection)."""
    timestamps: np.ndarray   # float64, unix seconds (RTK clock)
    lat: np.ndarray          # float64, degrees
    lng: np.ndarray          # float64, degrees
    drive_id: str            # e.g. "201703061033"
    drive_day: str           # e.g. "2017_03_06_ITS1"
    bbox_extent_m: float


def load_rtk_trajectory(drive_dir: Path) -> HDDTrajectory:
    """Read ``general/csv/rtk_pos.csv`` for one drive.

    Args:
        drive_dir: a single-drive directory (contains ``general/`` and
            ``camera/``), e.g. ``.../2017_03_06_ITS1/201703061033``.

    Returns:
        An ``HDDTrajectory`` in real WGS84 degrees.

    Raises:
        FileNotFoundError: if ``rtk_pos.csv`` is missing.
        ValueError: if the file is empty or the coordinates fall outside
            the SF Bay guard box (likely a swapped-column / bad-fix file).
    """
    csv_path = drive_dir / "general" / "csv" / "rtk_pos.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No rtk_pos.csv at {csv_path}")

    # header: # unix_timestamp,iso_timestamp,lng,lat  (values are lat,lng)
    raw = np.genfromtxt(csv_path, delimiter=",", comments="#",
                        usecols=(0, 2, 3), dtype=np.float64)
    raw = np.atleast_2d(raw)
    if raw.size == 0:
        raise ValueError(f"empty rtk_pos.csv at {csv_path}")

    ts, lat, lng = raw[:, 0], raw[:, 1], raw[:, 2]
    finite = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lng)
    ts, lat, lng = ts[finite], lat[finite], lng[finite]
    if ts.size == 0:
        raise ValueError(f"no finite rows in {csv_path}")

    in_box = ((lat >= _LAT_RANGE[0]) & (lat <= _LAT_RANGE[1]) &
              (lng >= _LNG_RANGE[0]) & (lng <= _LNG_RANGE[1]))
    if in_box.mean() < 0.5:
        raise ValueError(
            f"{csv_path}: {(~in_box).sum()}/{in_box.size} points outside the "
            f"SF Bay box (first lat={lat[0]:.3f}, lng={lng[0]:.3f}); likely a "
            f"swapped-column or bad-fix file.")
    ts, lat, lng = ts[in_box], lat[in_box], lng[in_box]

    order = np.argsort(ts)
    ts, lat, lng = ts[order], lat[order], lng[order]

    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * np.cos(np.radians(float(np.median(lat))))
    bbox_extent_m = float(max(
        (lat.max() - lat.min()) * m_per_deg_lat,
        (lng.max() - lng.min()) * m_per_deg_lng,
    ))

    return HDDTrajectory(
        timestamps=ts, lat=lat, lng=lng,
        drive_id=drive_dir.name,
        drive_day=drive_dir.parent.name,
        bbox_extent_m=bbox_extent_m,
    )


def find_video(drive_dir: Path) -> Path | None:
    """Return the center-camera MP4 for a drive, or None if missing."""
    center = drive_dir / "camera" / "center"
    if not center.is_dir():
        return None
    vids = sorted(center.glob("*.mp4")) + sorted(center.glob("*.MP4"))
    return vids[0] if vids else None


def discover_drives(root: Path) -> list[Path]:
    """Discover all HDD drives with both RTK GPS and a center video.

    A valid drive has ``general/csv/rtk_pos.csv`` and a ``camera/center``
    MP4. Drive-days are the ``2017_*_ITS1`` subdirectories of ``root``.
    """
    drives: list[Path] = []
    for day_dir in sorted(root.glob("2017_*_ITS1")):
        if not day_dir.is_dir():
            continue
        for drive_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            if not (drive_dir / "general" / "csv" / "rtk_pos.csv").is_file():
                continue
            if find_video(drive_dir) is None:
                continue
            drives.append(drive_dir)
    return drives
