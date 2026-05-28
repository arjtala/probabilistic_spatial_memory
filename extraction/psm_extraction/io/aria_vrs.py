"""Project Aria VRS reader for Aria Gen 1 (Nymeria, AEA) and Aria Gen 2 Pilot.

Bridges `projectaria-tools`' VRS playback into the
`extraction/psm_extraction/extract.py` orchestration pipeline. The
function `read_vrs_session(session_dir, sample_fps, output_dir)` is the
public entry point — given an Aria session directory (the format used by
both Nymeria and Aria Gen 2 Pilot Dataset) and a target sample rate, it:

  1. Locates the primary VRS file (`recording_head/data/data.vrs` for
     Nymeria, top-level `<sequence>.vrs` for Aria Gen 2).
  2. Iterates the RGB camera stream at the requested rate using the
     real device-clock timestamps as the source of truth.
  3. Writes JPEGs to `output_dir/frame_%06d.jpg` matching the file-name
     convention of the existing ffmpeg-backed `extract_frames` cache,
     so downstream model runners are unchanged.
  4. Writes a `.extract_manifest.json` so subsequent calls with matching
     `(vrs_path, sample_fps)` skip the re-decode.
  5. Loads the closed-loop MPS SLAM trajectory if present, returning
     per-frame `(lat, lng)` via a flat-earth projection from `(tx, ty)`
     world coordinates (origin = first sample). Indoor sessions land
     at a per-session fake origin; outdoor sessions with a real GPS
     sidecar should use the GPS loader in `extract.py` instead.

Stream IDs:
  - Aria Gen 1 RGB:    `214-1`
  - Aria Gen 2 RGB:    same `214-1` (the device + SDK preserve stream
    IDs across generations for the principal camera).

Both Nymeria and Aria Gen 2 Pilot share the VRS format and MPS layout,
so this module works for both. The session-directory layout
differences are handled in `_locate_vrs_file`.

Dependencies: `projectaria-tools` (`pip install -e ./extraction[aria]`).
The import is deferred to the function body so importing this module
does not require projectaria-tools to be installed (helpful for
type-checking / docstring tooling).
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Earth radius is unused below; the flat-earth projection uses a fixed
# meters-per-degree at the chosen origin latitude, which is more accurate
# at room scale than the Haversine alternative.
_METERS_PER_DEG_LAT = 111_132.0  # average; varies by latitude but room-scale doesn't care

# Aria's principal RGB camera stream — same on Gen 1 and Gen 2.
_RGB_STREAM_ID = "214-1"

# Per-frame JPEG quality; matches what `extract_frames` produces with ffmpeg's
# defaults. Higher is wasted because CLIP/DINO preprocessing downsamples to 224.
_JPEG_QUALITY = 90

# Manifest filename mirrors `extract_frames` so the orchestrator's
# cache-validation code doesn't need a separate path.
_MANIFEST_FILENAME = ".extract_manifest.json"


@dataclass
class VrsExtractResult:
    """Output of `read_vrs_session` — slots into ExtractOptions downstream.

    `lats` and `lngs` are None when no SLAM trajectory was loadable; the
    orchestrator should fall back to GPS sidecars or synthetic tracks in
    that case (same as the current MP4 path).
    """
    frame_paths: list[Path]
    timestamps_s: np.ndarray            # (N,) device-clock seconds, monotonic
    lats: np.ndarray | None             # (N,) fake degrees (SLAM-projected) or None
    lngs: np.ndarray | None
    source_vrs: Path
    trajectory_origin: tuple[float, float] | None  # (lat0, lng0) used for projection


def _locate_vrs_file(session_dir: Path) -> Path:
    """Pick the primary RGB VRS file for an Aria session directory.

    Handles both layouts:
      - Nymeria / AEA: <session_dir>/recording_head/data/data.vrs
      - Aria Gen 2 Pilot: <session_dir>/main.vrs (or <sequence>.vrs)
    """
    candidates = [
        session_dir / "recording_head" / "data" / "data.vrs",
        session_dir / "main.vrs",
    ]
    # Aria Gen 2 sometimes names the main VRS after the sequence — fall
    # back to "any .vrs file at session root that isn't an aux stream."
    for p in candidates:
        if p.exists():
            return p
    root_vrs = [
        p for p in session_dir.glob("*.vrs")
        if p.name not in {"et.vrs", "motion.vrs"}  # eye-tracking / body-motion aux
    ]
    if len(root_vrs) == 1:
        return root_vrs[0]
    if not root_vrs:
        raise FileNotFoundError(
            f"no VRS file found under {session_dir} (looked at recording_head/data/data.vrs, "
            f"main.vrs, and any top-level *.vrs excluding et/motion)"
        )
    raise FileNotFoundError(
        f"multiple top-level VRS files under {session_dir}; specify one explicitly: "
        f"{[p.name for p in root_vrs]}"
    )


def _locate_slam_trajectory(session_dir: Path) -> Path | None:
    """Find the MPS closed-loop trajectory CSV if present.

    Layout:
      - Nymeria / AEA: <session_dir>/recording_head/mps/slam/closed_loop_trajectory.csv
      - Aria Gen 2 Pilot: <session_dir>/mps/slam/closed_loop_trajectory.csv

    Returns None when no trajectory is on disk; callers should fall back to
    the GPS sidecar path or synthetic geometry.
    """
    candidates = [
        session_dir / "recording_head" / "mps" / "slam" / "closed_loop_trajectory.csv",
        session_dir / "mps" / "slam" / "closed_loop_trajectory.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _read_manifest(output_dir: Path) -> dict | None:
    p = output_dir / _MANIFEST_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(
    output_dir: Path,
    *,
    vrs_path: Path,
    sample_fps: float,
    frame_count: int,
    timestamps_s: np.ndarray,
) -> None:
    """Mirror `extract_frames`' manifest shape + add a `timestamps_s` array.

    The orchestrator needs real per-frame timestamps for VRS-sourced
    frames (the synthetic `arange / fps` it uses for MP4 frames is
    wrong for VRS because the device clock isn't perfectly periodic).
    Stash them in the manifest so a cache hit can recover them without
    re-opening the VRS file.
    """
    (output_dir / _MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "source": "vrs",
                "video": str(vrs_path.resolve()),
                "sample_fps": float(sample_fps),
                "frame_count": int(frame_count),
                "timestamps_s": timestamps_s.tolist(),
            },
            indent=2,
        )
    )


def _read_slam_trajectory(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Parse Aria MPS closed_loop_trajectory.csv → (t_s, tx, ty) in meters.

    The MPS schema columns of interest:
      - tracking_timestamp_us:   device-clock microseconds (matches VRS frame
                                 capture timestamps)
      - tx_world_device, ty_world_device, tz_world_device:  world-frame
        position in meters

    Other columns (orientation quaternion, gravity vector, etc.) are
    ignored — PSM only cares about (lat, lng) and we're projecting from
    (tx, ty). Returns None if the file is empty or malformed.
    """
    try:
        import csv as _csv
    except ImportError:
        return None

    t_us: list[int] = []
    tx: list[float] = []
    ty: list[float] = []
    with csv_path.open() as f:
        reader = _csv.DictReader(f)
        # Skim the columns once to surface a clear error if the schema drifts.
        required = {"tracking_timestamp_us", "tx_world_device", "ty_world_device"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(
                f"[vrs] WARN: closed_loop_trajectory.csv at {csv_path} missing "
                f"columns {missing}; skipping SLAM trajectory load.",
                file=sys.stderr,
            )
            return None
        for row in reader:
            try:
                t_us.append(int(row["tracking_timestamp_us"]))
                tx.append(float(row["tx_world_device"]))
                ty.append(float(row["ty_world_device"]))
            except (KeyError, ValueError):
                continue
    if not t_us:
        return None
    return (
        np.array(t_us, dtype=np.int64) / 1e6,  # → seconds
        np.array(tx, dtype=np.float64),
        np.array(ty, dtype=np.float64),
    )


def _project_xy_to_latlng(
    tx_world: np.ndarray,
    ty_world: np.ndarray,
    origin_lat: float = 0.0,
    origin_lng: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat-earth projection: world-frame (tx, ty) meters → fake (lat, lng).

    At room scale (10s-100s of meters), the curvature error vs a
    proper geodetic projection is negligible (<<1 cm). H3 cell
    boundaries at res 11-12 are at the meter scale, which dwarfs any
    projection imprecision.

    `(origin_lat, origin_lng)` defaults to (0, 0) — sessions land on
    the equator near the prime meridian on a global map. That's an
    "obviously fake" location, which is the honest visualization
    semantic for indoor data. If the paper figure needs sessions
    placed somewhere visually interesting, pick origin per-session.
    """
    lat0_rad = math.radians(origin_lat)
    meters_per_deg_lng_at_origin = _METERS_PER_DEG_LAT * math.cos(lat0_rad)
    # PSM convention: tx_world is "east" (lng), ty_world is "north" (lat).
    # Aria MPS world frame is right-handed +Z up, but the exact +X/+Y
    # orientation depends on the per-session VIO bootstrap. We don't
    # care — H3 cell ID is invariant under any global rotation, so
    # whichever axis we call "east" is fine as long as it's consistent
    # within a session.
    lats = origin_lat + ty_world / _METERS_PER_DEG_LAT
    lngs = origin_lng + tx_world / meters_per_deg_lng_at_origin
    return lats, lngs


def _interpolate_track_to_frames(
    frame_ts_s: np.ndarray,
    track_ts_s: np.ndarray,
    track_tx: np.ndarray,
    track_ty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate (tx, ty) onto frame timestamps; clamp at edges.

    Matches the policy `extract.py.map_frames_to_gps` uses for GPS:
    interpolation between the two nearest track samples; frames before
    the first or after the last track sample get clamped (= the edge
    track sample's (tx, ty)). This is the right behavior for the few
    hundred ms of clock skew at session start/end that the MPS pipeline
    sometimes produces.
    """
    if track_ts_s.size == 0:
        nan = np.full_like(frame_ts_s, np.nan)
        return nan, nan
    # np.interp handles the clamp-at-edges behavior natively.
    tx = np.interp(frame_ts_s, track_ts_s, track_tx)
    ty = np.interp(frame_ts_s, track_ts_s, track_ty)
    return tx, ty


def read_vrs_session(
    session_dir: Path,
    sample_fps: float,
    output_dir: Path,
    *,
    origin_lat: float = 0.0,
    origin_lng: float = 0.0,
    verbose: bool = False,
    force: bool = False,
) -> VrsExtractResult:
    """Extract sampled JPEGs + (timestamp, lat, lng) per frame from a VRS session.

    See module docstring for the full pipeline. This is the public entry
    point that `extract.py` should call when the input is an Aria
    session directory rather than an MP4.

    Cache behavior matches `extract_frames`: on a hit (manifest's
    `(video, sample_fps)` matches and the JPEG count is right), the
    decoded JPEGs and cached timestamps are returned without re-opening
    the VRS file. SLAM trajectory is re-read either way since it's a
    small CSV.
    """
    session_dir = Path(session_dir).resolve()
    output_dir = Path(output_dir)
    if sample_fps <= 0.0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")

    vrs_path = _locate_vrs_file(session_dir)
    if verbose:
        print(f"[vrs] using {vrs_path}", file=sys.stderr)

    # Cache check — same shape as extract_frames so the orchestrator's
    # incremental-rebuild logic works.
    if not force:
        existing = _read_manifest(output_dir)
        if (
            existing is not None
            and existing.get("source") == "vrs"
            and existing.get("video") == str(vrs_path.resolve())
            and float(existing.get("sample_fps", 0.0)) == float(sample_fps)
        ):
            cached_paths = sorted(output_dir.glob("frame_*.jpg"))
            if cached_paths and len(cached_paths) == int(existing.get("frame_count", -1)):
                if verbose:
                    print(
                        f"[vrs] cache hit: {len(cached_paths)} JPEGs at {output_dir}",
                        file=sys.stderr,
                    )
                timestamps_s = np.array(
                    existing.get("timestamps_s", []), dtype=np.float64
                )
                lats, lngs, origin = _load_slam_optional(
                    session_dir, timestamps_s,
                    origin_lat=origin_lat, origin_lng=origin_lng,
                    verbose=verbose,
                )
                return VrsExtractResult(
                    frame_paths=cached_paths,
                    timestamps_s=timestamps_s,
                    lats=lats,
                    lngs=lngs,
                    source_vrs=vrs_path,
                    trajectory_origin=origin,
                )

    # Cold path: decode VRS, write JPEGs.
    # Lazy import so the module is loadable without projectaria-tools.
    try:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
    except ImportError as exc:
        raise ImportError(
            "projectaria-tools is required for VRS reading. "
            "Install with: pip install -e ./extraction[aria]"
        ) from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for VRS frame export. "
            "Install with: pip install Pillow"
        ) from exc

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if provider is None:
        raise RuntimeError(f"projectaria-tools failed to open VRS file {vrs_path}")
    rgb_stream = StreamId(_RGB_STREAM_ID)
    n_rgb = provider.get_num_data(rgb_stream)
    if n_rgb == 0:
        raise RuntimeError(
            f"{vrs_path} has zero RGB frames on stream {_RGB_STREAM_ID}; "
            "is this the right VRS file?"
        )

    # Compute the indices we want by walking timestamps and gating on
    # the inter-sample gap. This is robust to clock jitter — if the
    # device's nominal RGB rate drifts (which Aria Gen 1/2 do), we
    # still end up with sample_fps-ish output samples without rate
    # accumulation errors that a naive `range(0, n_rgb, stride)` would
    # produce.
    target_dt = 1.0 / float(sample_fps)
    selected_indices: list[int] = []
    selected_ts_s: list[float] = []
    next_t = -math.inf  # first frame always passes
    for i in range(n_rgb):
        # `get_image_data_by_index` returns (ImageData, ImageDataRecord).
        # We only need the timestamp here; image bytes come later.
        _, record = provider.get_image_data_by_index(rgb_stream, i)
        t_s = record.capture_timestamp_ns / 1e9
        if t_s >= next_t:
            selected_indices.append(i)
            selected_ts_s.append(t_s)
            next_t = t_s + target_dt

    if not selected_indices:
        raise RuntimeError(f"sample_fps={sample_fps} selected zero frames from {vrs_path}")
    if verbose:
        print(
            f"[vrs] decoding {len(selected_indices)} frames "
            f"(of {n_rgb} total) at target {sample_fps} fps",
            file=sys.stderr,
        )

    # Second pass: decode + JPEG-encode just the chosen indices. Done
    # second so the timestamp-selection loop above doesn't pay the
    # decode cost for frames we discard.
    frame_paths: list[Path] = []
    for out_idx, src_idx in enumerate(selected_indices):
        image_data, _ = provider.get_image_data_by_index(rgb_stream, src_idx)
        arr = image_data.to_numpy_array()
        img = Image.fromarray(arr)
        out_path = output_dir / f"frame_{out_idx:06d}.jpg"
        img.save(out_path, format="JPEG", quality=_JPEG_QUALITY)
        frame_paths.append(out_path)

    timestamps_s = np.array(selected_ts_s, dtype=np.float64)
    _write_manifest(
        output_dir,
        vrs_path=vrs_path,
        sample_fps=sample_fps,
        frame_count=len(frame_paths),
        timestamps_s=timestamps_s,
    )

    lats, lngs, origin = _load_slam_optional(
        session_dir, timestamps_s,
        origin_lat=origin_lat, origin_lng=origin_lng,
        verbose=verbose,
    )

    return VrsExtractResult(
        frame_paths=frame_paths,
        timestamps_s=timestamps_s,
        lats=lats,
        lngs=lngs,
        source_vrs=vrs_path,
        trajectory_origin=origin,
    )


def _load_slam_optional(
    session_dir: Path,
    timestamps_s: np.ndarray,
    *,
    origin_lat: float,
    origin_lng: float,
    verbose: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[float, float] | None]:
    """Try to load + project SLAM trajectory; return (None, None, None) if absent."""
    traj_csv = _locate_slam_trajectory(session_dir)
    if traj_csv is None:
        if verbose:
            print(
                f"[vrs] no MPS SLAM trajectory at {session_dir}; (lat, lng) will be None",
                file=sys.stderr,
            )
        return None, None, None

    parsed = _read_slam_trajectory(traj_csv)
    if parsed is None:
        return None, None, None
    track_ts_s, track_tx, track_ty = parsed
    if verbose:
        print(
            f"[vrs] loaded SLAM trajectory: {track_ts_s.size} samples spanning "
            f"{track_ts_s[-1] - track_ts_s[0]:.1f}s",
            file=sys.stderr,
        )

    tx, ty = _interpolate_track_to_frames(timestamps_s, track_ts_s, track_tx, track_ty)
    lats, lngs = _project_xy_to_latlng(tx, ty, origin_lat=origin_lat, origin_lng=origin_lng)
    return lats, lngs, (origin_lat, origin_lng)
