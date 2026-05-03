"""Aria-style JSON sidecar reader.

Aria recordings get post-processed into per-stream JSON files alongside the
muxed video — typically `gps.json` and `imu.json`. Each is a list of
`{stream_id, samples: [...]}` blobs; we pick the largest non-empty stream
by default and emit numpy arrays ready to feed straight into the v2 writer.

The reader does not depend on `projectaria_tools`. For raw VRS files, use
`io.aria_vrs` (Phase 4 follow-up).
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Aria's pre-fix GPS samples ship as denormalized (smallest positive) doubles
# with all coordinates at 0.0. Filter them rather than emit them as valid.
_DENORMAL_TIMESTAMP_THRESHOLD = 1e-300


@dataclass(frozen=True)
class GPSSidecar:
    timestamps: np.ndarray  # float64 seconds, monotonic non-decreasing
    lat: np.ndarray         # float64 degrees, WGS84
    lng: np.ndarray         # float64 degrees, WGS84
    stream_id: str          # e.g. "281-1"
    # True when timestamps were derived from `utc_time_ms` (absolute Unix
    # seconds); False when they came from `timestamp` (a session-clock /
    # device-monotonic value the orchestrator must add capture_time_epoch
    # to). Defaults to False for back-compat with sidecars that only carry
    # the relative form.
    timestamps_absolute: bool = False


@dataclass(frozen=True)
class IMUSidecar:
    timestamps: np.ndarray  # float64 seconds, monotonic non-decreasing
    accel: np.ndarray       # float32 (N, 3), m/s^2 in device frame
    gyro: np.ndarray        # float32 (N, 3), rad/s in device frame
    stream_id: str          # e.g. "1202-1"
    timestamps_absolute: bool = False


def _largest_stream(streams: list[dict]) -> dict:
    if not streams:
        raise RuntimeError("JSON sidecar has zero streams")
    return max(streams, key=lambda s: len(s.get("samples") or []))


def _utc_seconds(sample: dict) -> float | None:
    """Aria sidecars sometimes carry both a session-clock `timestamp` and
    an absolute `utc_time_ms`. Prefer the latter when present and non-zero
    because it's unambiguously Unix epoch (sub-second precision via the
    millisecond integer is sufficient for our purposes — the
    embedding-side timestamps are coarser than 1 ms anyway). Returns None
    when the field is absent, zero, or non-finite.
    """
    raw = sample.get("utc_time_ms")
    if raw is None:
        return None
    try:
        ms = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ms) or ms <= 0:
        return None
    return ms / 1000.0


def _valid_gps_sample(sample: dict) -> bool:
    ts = sample.get("timestamp")
    if not isinstance(ts, (int, float)):
        return False
    if not math.isfinite(ts) or abs(ts) < _DENORMAL_TIMESTAMP_THRESHOLD:
        return False
    lat = sample.get("latitude")
    lng = sample.get("longitude")
    if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
        return False
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return False
    if lat == 0.0 and lng == 0.0:
        return False
    return True


def read_gps_json(path: Path, *, stream_id: str | None = None) -> GPSSidecar:
    """Read an Aria-style gps.json file.

    Picks the stream with the most usable samples by default; pass
    `stream_id` to force a specific one (e.g. when the recording has multiple
    GPS providers and you want only one).
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise RuntimeError(f"{path}: expected a top-level JSON array of streams")
    if stream_id is not None:
        candidates = [s for s in raw if s.get("stream_id") == stream_id]
        if not candidates:
            raise RuntimeError(f"{path}: stream_id {stream_id!r} not found")
        stream = candidates[0]
    else:
        stream = _largest_stream(raw)
    samples = stream.get("samples") or []
    valid = [s for s in samples if _valid_gps_sample(s)]
    if not valid:
        raise RuntimeError(f"{path}: stream has no valid GPS fixes")

    utc = [_utc_seconds(s) for s in valid]
    use_utc = all(u is not None for u in utc)
    if use_utc:
        ts = np.fromiter((u for u in utc), dtype=np.float64)
    else:
        ts = np.fromiter((float(s["timestamp"]) for s in valid), dtype=np.float64)
    lat = np.fromiter((float(s["latitude"]) for s in valid), dtype=np.float64)
    lng = np.fromiter((float(s["longitude"]) for s in valid), dtype=np.float64)

    order = np.argsort(ts)
    return GPSSidecar(
        timestamps=ts[order],
        lat=lat[order],
        lng=lng[order],
        stream_id=str(stream.get("stream_id", "unknown")),
        timestamps_absolute=use_utc,
    )


def _valid_imu_sample(sample: dict) -> bool:
    ts = sample.get("timestamp")
    if not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return False
    accel = sample.get("accel")
    gyro = sample.get("gyro")
    if not (isinstance(accel, list) and len(accel) == 3):
        return False
    if not (isinstance(gyro, list) and len(gyro) == 3):
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in accel + gyro)


def read_imu_json(path: Path, *, stream_id: str | None = None) -> IMUSidecar:
    """Read an Aria-style imu.json file.

    These files can be large (hundreds of MB). The reader uses the standard
    library's `json` module for simplicity; for sessions over a couple of
    hours a streaming parser would be a worthwhile upgrade.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise RuntimeError(f"{path}: expected a top-level JSON array of streams")
    if stream_id is not None:
        candidates = [s for s in raw if s.get("stream_id") == stream_id]
        if not candidates:
            raise RuntimeError(f"{path}: stream_id {stream_id!r} not found")
        stream = candidates[0]
    else:
        stream = _largest_stream(raw)
    samples = stream.get("samples") or []
    valid = [s for s in samples if _valid_imu_sample(s)]
    if not valid:
        raise RuntimeError(f"{path}: stream has no valid IMU samples")

    n = len(valid)
    ts = np.empty(n, dtype=np.float64)
    accel = np.empty((n, 3), dtype=np.float32)
    gyro = np.empty((n, 3), dtype=np.float32)
    utc_seconds = [_utc_seconds(s) for s in valid]
    use_utc = all(u is not None for u in utc_seconds)
    for idx, sample in enumerate(valid):
        if use_utc:
            ts[idx] = utc_seconds[idx]  # type: ignore[assignment]
        else:
            ts[idx] = float(sample["timestamp"])
        accel[idx] = sample["accel"]
        gyro[idx] = sample["gyro"]

    order = np.argsort(ts)
    return IMUSidecar(
        timestamps=ts[order],
        accel=accel[order],
        gyro=gyro[order],
        stream_id=str(stream.get("stream_id", "unknown")),
        timestamps_absolute=use_utc,
    )


def read_metadata_json(path: Path) -> dict:
    """Read an Aria metadata.json file. Returns the parsed dict.

    Notable keys when present: `recording_id`, `device_serial`,
    `duration_seconds`, `tags.capture_time_epoch` (Unix seconds the relative
    timestamps in gps/imu sidecars are measured against).
    """
    return json.loads(Path(path).read_text())


def capture_time_epoch(metadata: dict) -> float | None:
    """Best-effort extraction of the absolute capture-time epoch (Unix s)."""
    tags = metadata.get("tags") or {}
    raw = tags.get("capture_time_epoch")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
