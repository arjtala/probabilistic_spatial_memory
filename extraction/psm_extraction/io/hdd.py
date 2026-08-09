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
# Per-drive outlier reject: drop teleport fixes >this far from the drive median
# (a real drive stays within one metro; isolated bad fixes otherwise bin to
# spurious H3 cells). Mirrors scripts/hdd_revisit_density.py.
_OUTLIER_KM = 80.0


def _haversine_km(lat, lng, lat0, lng0):
    """Vectorized great-circle distance (km) from each point to (lat0, lng0)."""
    r = 6371.0
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi = np.radians(lat - lat0)
    dlam = np.radians(lng - lng0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class HDDTrajectory:
    """RTK GPS trajectory for one HDD drive (real WGS84, no projection)."""
    timestamps: np.ndarray   # float64, unix seconds (RTK clock)
    lat: np.ndarray          # float64, degrees
    lng: np.ndarray          # float64, degrees
    drive_id: str            # e.g. "201703061033"
    drive_day: str           # e.g. "2017_03_06_ITS1"
    bbox_extent_m: float
    first_iso: str           # ISO timestamp of the first fix (for video-clock skew)


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
    # ISO timestamps (col 1) carried through the same filtering so first_iso
    # matches the earliest returned fix (used for the video-clock skew).
    iso = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", comments="#",
                                      usecols=(1,), dtype=str))

    ts, lat, lng = raw[:, 0], raw[:, 1], raw[:, 2]
    finite = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lng)
    ts, lat, lng = ts[finite], lat[finite], lng[finite]
    if iso.shape[0] == raw.shape[0]:
        iso = iso[finite]
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
    if iso.shape[0] == in_box.shape[0]:
        iso = iso[in_box]

    # Per-drive teleport-outlier reject (drop fixes far from the drive median).
    near = _haversine_km(lat, lng, float(np.median(lat)),
                         float(np.median(lng))) <= _OUTLIER_KM
    ts, lat, lng = ts[near], lat[near], lng[near]
    if iso.shape[0] == near.shape[0]:
        iso = iso[near]
    if ts.size == 0:
        raise ValueError(f"no in-region rows in {csv_path}")

    order = np.argsort(ts)
    ts, lat, lng = ts[order], lat[order], lng[order]
    first_iso = str(iso[order][0]) if iso.shape[0] == order.shape[0] else ""

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
        first_iso=first_iso,
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


def video_start_skew(drive_dir: Path, traj: HDDTrajectory) -> float:
    """Seconds the first GPS fix lags the video start (>=0 typical, RTK warmup).

    Center-camera files are named ``YYYY-MM-DD-HH-MM-SS_...mp4`` (local wall
    clock). ``traj.first_iso`` is the first fix's ISO time. When ``first_iso``
    is naive we assume it shares the video's local timezone (release_2019_07_08
    RTK CSVs write local wall-clock without an offset), so a naive-datetime
    difference is the true skew regardless of PST/PDT. When ``first_iso`` is
    tz-aware (e.g. ``2017-02-27T10:17:30-08:00``), its wall-clock fields are
    ALREADY the capture's local time — the offset only annotates which local
    zone that is. Dropping the tzinfo therefore yields exactly the local wall
    clock the video filename is in, and the naive difference is the true skew.

    Do NOT ``astimezone()`` here (see the 2026-08 postmortem in docs/HDD.md).
    Argless ``astimezone()`` converts to the *host* timezone, which on a UTC
    cluster node re-expresses a ``-08:00`` fix as UTC and inflates the skew by
    the whole 7-8 h offset. That fed ``realign_frames`` a ``video_start_unix``
    hours outside the RTK span, where ``np.interp`` silently clamps every
    frame to the first fix — collapsing the track to a single coordinate with
    no error raised. ``assert_track_coverage`` below now makes that loud, but
    the correct conversion is simply to drop the offset.
    Returns 0.0 if either is unparseable.
    """
    from datetime import datetime
    video = find_video(drive_dir)
    if video is None or not traj.first_iso:
        return 0.0
    try:
        vstart = datetime.strptime(video.name[:19], "%Y-%m-%d-%H-%M-%S")
        gps_first = datetime.fromisoformat(traj.first_iso)
        if gps_first.tzinfo is not None:
            # The wall-clock fields are already capture-local; just drop the
            # zone annotation so the subtraction matches the naive filename.
            gps_first = gps_first.replace(tzinfo=None)
        return (gps_first - vstart).total_seconds()
    except (ValueError, TypeError):
        return 0.0


#: Max seconds a frame may fall outside the RTK span before we call the
#: alignment broken. Real drives overhang by a frame or two at each end
#: (video starts marginally before the first fix / ends after the last);
#: anything beyond this means the clocks disagree, not that the track is short.
_MAX_OVERHANG_SEC = 30.0


def assert_track_coverage(
    frame_abs: np.ndarray,
    xp: np.ndarray,
    *,
    drive_id: str = "",
    max_overhang_sec: float = _MAX_OVERHANG_SEC,
) -> None:
    """Fail loudly when interpolation times fall outside the RTK track.

    ``np.interp`` clamps out-of-range queries to the endpoint value instead of
    erroring, so a clock-alignment bug does not surface as an exception — it
    surfaces as a track that has silently collapsed onto its first (or last)
    fix, which then reads downstream as a real, if degenerate, trajectory.
    That is precisely how the 2026-07 ``astimezone()`` skew bug survived a
    full 132-drive run. Anything computed from a clamped track is void, so we
    raise rather than let it propagate.
    """
    if frame_abs.size == 0 or xp.size == 0:
        raise ValueError(f"{drive_id or 'drive'}: empty frame or RTK time axis")
    lo, hi = float(xp[0]), float(xp[-1])
    before = float(np.maximum(0.0, lo - frame_abs.min()))
    after = float(np.maximum(0.0, frame_abs.max() - hi))
    overhang = max(before, after)
    if overhang > max_overhang_sec:
        n_out = int(((frame_abs < lo) | (frame_abs > hi)).sum())
        raise ValueError(
            f"{drive_id or 'drive'}: frame times fall {overhang:.1f} s outside "
            f"the RTK span ({n_out}/{frame_abs.size} frames out of range; "
            f"{before:.1f} s before / {after:.1f} s after), exceeding the "
            f"{max_overhang_sec:.0f} s tolerance. np.interp would clamp these "
            "to an endpoint and silently collapse the track — refusing. This "
            "is a video/GPS clock-alignment bug (check video_start_skew), not "
            "a short track."
        )


def realign_frames(
    drive_dir: Path,
    frame_pts: np.ndarray,
    *,
    on_gap: str = "raise",
) -> np.ndarray | None:
    """Correct per-frame (lat, lng) from raw RTK GPS on the true video clock.

    The 2026-07-05 extraction wrote ``clip/lat,lng`` with a ~3.4 s skew (the
    sidecar's warmup correction was cancelled by extract.py's unconditional
    rebase) plus a few teleport outliers (no outlier guard at extract time).
    This recomputes correct coords non-destructively for analysis:

        frame i (video PTS ``frame_pts[i]``, 0-based) is at real time
        ``video_start_unix + frame_pts[i]`` where
        ``video_start_unix = gps_unix[0] - skew``; interpolate the (outlier-
        filtered) raw RTK track at that absolute time.

    Returns an (N, 2) array of [lat, lng] aligned to ``frame_pts``, or None if
    the drive's raw GPS is unreadable.

    ``on_gap`` controls what happens when frame times fall outside the RTK
    span — which is real on a handful of drives whose RTK acquisition lagged
    the video start by minutes:

    * ``"raise"`` (default) — ``ValueError`` via ``assert_track_coverage``. A
      clamped track is worse than no track, so callers must opt in to gaps.
    * ``"nan"`` — interpolate where covered and write NaN elsewhere. Keeps the
      frame axis aligned with the embeddings while making uncovered frames
      unmistakable to anything downstream (H3 conversion fails on NaN rather
      than binning a fabricated coordinate).

    As of the 2026-08 coords-only rewrite the shipped H5s carry corrected
    ``clip/lat,lng`` at rest, so this is a verification/recompute path rather
    than a correction every reader must remember to apply.
    """
    try:
        traj = load_rtk_trajectory(drive_dir)
    except (FileNotFoundError, ValueError):
        return None
    skew = video_start_skew(drive_dir, traj)
    video_start_unix = float(traj.timestamps[0]) - skew
    frame_abs = video_start_unix + np.asarray(frame_pts, dtype=np.float64)
    # np.interp needs strictly increasing xp; timestamps are sorted, drop dups.
    keep = np.concatenate(([True], np.diff(traj.timestamps) > 0))
    xp, la, lo = traj.timestamps[keep], traj.lat[keep], traj.lng[keep]
    if on_gap not in {"raise", "nan"}:
        raise ValueError(f"on_gap must be 'raise' or 'nan', got {on_gap!r}")
    # np.interp clamps out-of-range queries instead of raising; check first.
    if on_gap == "raise":
        assert_track_coverage(frame_abs, xp, drive_id=drive_dir.name)
    lat = np.interp(frame_abs, xp, la)
    lng = np.interp(frame_abs, xp, lo)
    if on_gap == "nan":
        outside = (frame_abs < xp[0]) | (frame_abs > xp[-1])
        lat = np.where(outside, np.nan, lat)
        lng = np.where(outside, np.nan, lng)
    return np.stack([lat, lng], axis=1)

