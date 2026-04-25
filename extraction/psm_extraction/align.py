"""GPS / track resolution for frame timestamps.

Two paths:
- `load_session_track` reads a per-frame (or per-fix) track from an existing
  features.h5 (preferred groups in order: dino, jepa, gps) and normalizes it
  to relative seconds since the first sample.
- `synthetic_snake_grid` lays segments onto a deterministic H3-friendly
  snake-grid for plain videos that have no GPS.

Pure numpy + h5py — safe to import without torch/transformers.
"""

import dataclasses
from pathlib import Path

import h5py
import numpy as np

DEFAULT_CELL_STEP_DEG = 0.02
DEFAULT_GRID_COLUMNS = 128
DEFAULT_BASE_LAT = 37.0
DEFAULT_BASE_LNG = -122.0


@dataclasses.dataclass(frozen=True)
class SessionTrack:
    """Relative-time GPS track loaded from a session HDF5."""

    rel_seconds: np.ndarray
    lat: np.ndarray
    lng: np.ndarray
    source_group: str


def load_session_track(features_path: Path) -> SessionTrack | None:
    """Load (relative_seconds, lat, lng, group_name) from a session HDF5.

    Tries 'dino' first (per-frame aligned, ~30 fps), then 'jepa', then the
    raw 'gps' group. All paths are normalized to relative seconds since the
    first sample so callers can match against video-relative frame
    timestamps without needing the absolute video start time. Returns None
    if no usable group is present.
    """
    if not features_path.exists():
        return None
    try:
        with h5py.File(features_path, "r") as handle:
            for candidate in ("dino", "jepa", "gps"):
                if candidate not in handle:
                    continue
                group = handle[candidate]
                if not all(name in group for name in ("timestamps", "lat", "lng")):
                    continue
                ts = np.asarray(group["timestamps"], dtype=np.float64)
                lat = np.asarray(group["lat"], dtype=np.float64)
                lng = np.asarray(group["lng"], dtype=np.float64)
                if ts.shape != lat.shape or ts.shape != lng.shape or ts.size == 0:
                    continue
                order = np.argsort(ts)
                ts = ts[order]
                lat = lat[order]
                lng = lng[order]
                mask = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lng)
                if not mask.any():
                    continue
                ts = ts[mask]
                lat = lat[mask]
                lng = lng[mask]
                return SessionTrack(
                    rel_seconds=ts - ts[0],
                    lat=lat,
                    lng=lng,
                    source_group=candidate,
                )
    except (OSError, KeyError):
        return None
    return None


def map_frames_to_gps(
    frame_timestamps: np.ndarray, track: SessionTrack
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate the track onto frame timestamps (clipped to range)."""
    if track.rel_seconds.size == 0:
        raise ValueError("track is empty; cannot interpolate")
    clipped = np.clip(frame_timestamps, track.rel_seconds[0], track.rel_seconds[-1])
    lats = np.interp(clipped, track.rel_seconds, track.lat)
    lngs = np.interp(clipped, track.rel_seconds, track.lng)
    return lats, lngs


def synthetic_snake_grid(
    timestamps: np.ndarray,
    *,
    segment_sec: float,
    grid_columns: int = DEFAULT_GRID_COLUMNS,
    cell_step_deg: float = DEFAULT_CELL_STEP_DEG,
    base_lat: float = DEFAULT_BASE_LAT,
    base_lng: float = DEFAULT_BASE_LNG,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Map timestamps to a deterministic snake-grid of pseudo-cells.

    Frames inside the same `segment_sec`-wide window share a pseudo-cell so
    PSM returns narrow temporal intervals rather than collapsing the whole
    video into one tile. Returns `(lats, lngs, segment_count)`.
    """
    if segment_sec <= 0.0:
        raise ValueError("segment_sec must be > 0")
    n = int(timestamps.shape[0])
    lats = np.empty(n, dtype=np.float64)
    lngs = np.empty(n, dtype=np.float64)
    segment_ids = np.floor(timestamps / segment_sec).astype(np.int64)
    max_segment = int(segment_ids.max()) if n else 0

    for idx in range(n):
        segment = int(segment_ids[idx])
        row, col = divmod(segment, grid_columns)
        if row % 2 == 1:
            col = grid_columns - 1 - col
        lat = base_lat + row * cell_step_deg
        lng = base_lng + col * cell_step_deg
        if lat > 89.0 or lng < -179.0 or lng > 179.0:
            raise RuntimeError(
                "synthetic snake-grid overflowed valid lat/lng bounds; "
                "increase grid_columns or reduce cell_step_deg"
            )
        lats[idx] = lat
        lngs[idx] = lng
    return lats, lngs, max_segment + 1
