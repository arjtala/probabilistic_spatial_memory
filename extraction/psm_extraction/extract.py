"""End-to-end orchestrator: video → frames → embeddings → v2 features.h5.

Supports multi-model extraction (one frame pass, N model groups written into
the same file) and optional sensor-group writing from JSON sidecars.
"""

import dataclasses
import hashlib
import shutil
from pathlib import Path

import numpy as np

from . import schema
from .align import (
    SessionTrack,
    load_session_track,
    map_frames_to_gps,
    synthetic_snake_grid,
)
from .io import (
    GPSSidecar,
    IMUSidecar,
    capture_time_epoch,
    extract_frames,
    read_gps_json,
    read_imu_json,
    read_metadata_json,
    video_duration,
)
from .models import ModelRunner
from .progress import make_progress_logger, stage_banner
from .writer import FeaturesWriter


@dataclasses.dataclass
class ExtractOptions:
    """Inputs for a single multi-model extraction run."""

    video: Path
    output: Path
    runners: list[tuple[str, ModelRunner]]
    sample_fps: float = 1.0
    segment_sec: float = 2.0
    batch_size: int = 16
    use_gps: bool = True
    gps_source: Path | None = None
    gps_json: Path | None = None
    imu_json: Path | None = None
    metadata_json: Path | None = None
    grid_columns: int = 128
    cell_step_deg: float = 0.02
    keep_frames: bool = False
    frames_dir: Path | None = None
    source_video_attr: str | None = None
    session_id: str | None = None
    verbose: bool = False
    force_reextract: bool = False
    """If True, wipe and re-extract frames even if a matching cache exists."""
    force_reembed: bool = False
    """If True, ignore cached embeddings and re-run model inference."""
    cache_dir: Path | None = None
    """Where per-model embedding caches go; defaults to opts.output.parent."""


@dataclasses.dataclass
class ExtractResult:
    features_path: Path
    group_names: list[str]
    sample_fps: float
    segment_sec: float
    timestamps: np.ndarray
    embedding_dims: dict[str, int]
    frame_count: int
    duration_sec: float | None
    track_mode: str
    track_source: Path | None
    track_source_group: str | None
    sensor_groups_written: list[str]


def _resolve_frames_dir(opts: ExtractOptions) -> Path:
    if opts.frames_dir is not None:
        return opts.frames_dir
    return opts.output.parent / "frames"


def _resolve_track(opts: ExtractOptions) -> tuple[SessionTrack | None, Path | None]:
    """Pick a per-frame GPS track for interpolation onto frame timestamps.

    Order of preference:
        1. opts.gps_source (an existing features.h5)
        2. opts.gps_json (raw Aria sidecar, normalized to relative seconds)
        3. auto: <video_dir>/features.h5 if it has a usable group
        4. auto: <video_dir>/gps.json
    """
    if not opts.use_gps:
        return None, None

    if opts.gps_source is not None:
        track = load_session_track(opts.gps_source)
        if track is None:
            raise RuntimeError(
                f"--gps-source {opts.gps_source} has no usable dino/jepa/gps group"
            )
        return track, opts.gps_source

    if opts.gps_json is not None:
        sidecar = read_gps_json(opts.gps_json)
        return _track_from_sidecar(sidecar), opts.gps_json

    h5_guess = opts.video.parent / "features.h5"
    if h5_guess.exists():
        track = load_session_track(h5_guess)
        if track is not None:
            return track, h5_guess

    json_guess = opts.video.parent / "gps.json"
    if json_guess.exists():
        sidecar = read_gps_json(json_guess)
        return _track_from_sidecar(sidecar), json_guess

    return None, None


def _track_from_sidecar(sidecar: GPSSidecar) -> SessionTrack:
    return SessionTrack(
        rel_seconds=sidecar.timestamps - sidecar.timestamps[0],
        lat=sidecar.lat,
        lng=sidecar.lng,
        source_group=f"json_sidecar:{sidecar.stream_id}",
    )


def _load_imu(opts: ExtractOptions) -> tuple[IMUSidecar | None, Path | None]:
    if opts.imu_json is not None:
        return read_imu_json(opts.imu_json), opts.imu_json
    auto = opts.video.parent / "imu.json"
    if auto.exists():
        return read_imu_json(auto), auto
    return None, None


def _load_gps_sidecar(opts: ExtractOptions) -> tuple[GPSSidecar | None, Path | None]:
    """Pick a gps.json sidecar regardless of where the per-frame track came from.

    Sensor-group writing is independent of track resolution: even when the
    per-frame interpolation source was an existing features.h5 (auto-detected
    h5_guess), we still want a `gps` group in the output so the file matches
    the original Aria pipeline's shape and the viz can draw the trace.
    """
    if opts.gps_json is not None:
        return read_gps_json(opts.gps_json), opts.gps_json
    auto = opts.video.parent / "gps.json"
    if auto.exists():
        return read_gps_json(auto), auto
    return None, None


def _validate_opts(opts: ExtractOptions) -> None:
    if not opts.video.exists():
        raise RuntimeError(f"video not found: {opts.video}")
    if opts.sample_fps <= 0.0:
        raise ValueError("sample_fps must be > 0")
    if opts.segment_sec <= 0.0:
        raise ValueError("segment_sec must be > 0")
    if opts.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not opts.runners:
        raise ValueError("ExtractOptions.runners is empty")
    seen: set[str] = set()
    for group_name, _runner in opts.runners:
        if group_name in {"gps", "imu"}:
            raise ValueError(f"{group_name!r} is reserved for sensor groups")
        if group_name in seen:
            raise ValueError(f"duplicate group name: {group_name!r}")
        seen.add(group_name)


def _embed_with_optional_maps(runner: ModelRunner, paths, batch_size, *, progress=None):
    """Call runner.embed_images and normalize its return shape.

    Some runners (DINO) emit `(embeddings, attention_maps)`; others (CLIP,
    JEPA) emit just embeddings. This wrapper hides that asymmetry from the
    orchestrator.
    """
    out = runner.embed_images(paths, batch_size=batch_size, progress=progress)
    if isinstance(out, tuple):
        embeddings, maps = out
    else:
        embeddings, maps = out, None
    return embeddings, maps


def _embedding_cache_path(opts: "ExtractOptions", group_name: str, runner: ModelRunner) -> Path:
    """Stable path for a per-model embedding cache.

    Cache key includes everything that affects the produced embedding: the
    runner's model_id + checkpoint, video path, sample_fps, and group name.
    Different settings → different cache → no stale reuse.
    """
    key = "|".join(
        [
            runner.model_id,
            runner.checkpoint,
            str(opts.video.resolve()),
            f"{opts.sample_fps:.6f}",
            group_name,
        ]
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    base_dir = opts.cache_dir if opts.cache_dir is not None else opts.output.parent
    return base_dir / f".{opts.output.stem}.{group_name}.{digest}.npz"


def _save_embeddings_cache(path: Path, embeddings: np.ndarray, maps: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"embeddings": embeddings.astype(np.float32, copy=False)}
    if maps is not None:
        payload["maps"] = maps.astype(np.float32, copy=False)
        payload["has_maps"] = np.array([1], dtype=np.uint8)
    else:
        payload["has_maps"] = np.array([0], dtype=np.uint8)
    np.savez(path, **payload)


def _load_embeddings_cache(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    arr = np.load(path)
    has_maps = bool(arr["has_maps"][0]) if "has_maps" in arr.files else False
    embeddings = np.asarray(arr["embeddings"], dtype=np.float32)
    maps = np.asarray(arr["maps"], dtype=np.float32) if has_maps else None
    return embeddings, maps


def extract(opts: ExtractOptions) -> ExtractResult:
    """Run the full pipeline. Side effect: writes opts.output."""
    _validate_opts(opts)
    stage_banner(
        "extract",
        f"video={opts.video.name} sample_fps={opts.sample_fps} "
        f"models={[name for name, _ in opts.runners]} output={opts.output.name}",
    )
    duration = video_duration(opts.video, verbose=opts.verbose)

    frames_dir = _resolve_frames_dir(opts)
    stage_banner("frames", f"sampling at {opts.sample_fps} fps into {frames_dir}")
    frame_paths = extract_frames(
        opts.video,
        opts.sample_fps,
        frames_dir,
        verbose=opts.verbose,
        force=opts.force_reextract,
    )
    stage_banner(
        "frames",
        f"{len(frame_paths)} JPEGs ready"
        + (f" (duration={duration:.1f}s)" if duration is not None else ""),
    )
    timestamps = np.arange(len(frame_paths), dtype=np.float64) / opts.sample_fps

    track, track_source = _resolve_track(opts)
    if track is not None:
        lats, lngs = map_frames_to_gps(timestamps, track)
        track_mode = "real_gps"
        track_source_group = track.source_group
        interpolation = "linear,clipped_at_edges,from=session_track"
    else:
        lats, lngs, _segments = synthetic_snake_grid(
            timestamps,
            segment_sec=opts.segment_sec,
            grid_columns=opts.grid_columns,
            cell_step_deg=opts.cell_step_deg,
        )
        track_mode = "synthetic_snake_grid"
        track_source_group = None
        interpolation = None

    imu_sidecar, imu_source = _load_imu(opts)
    gps_sidecar, gps_sidecar_source = _load_gps_sidecar(opts)
    metadata = None
    if opts.metadata_json is not None:
        metadata = read_metadata_json(opts.metadata_json)
    elif (auto_meta := opts.video.parent / "metadata.json").exists():
        try:
            metadata = read_metadata_json(auto_meta)
        except (OSError, ValueError):
            metadata = None

    # JSON sidecars carry relative timestamps; if metadata.json gives us the
    # capture epoch, shift to absolute Unix seconds so the produced file is
    # directly comparable to the existing pipeline's output. The shift is
    # applied uniformly to model-group timestamps and to any sensor group
    # we write from a JSON sidecar (so all groups in the file share a clock).
    epoch_offset = 0.0
    using_json_sources = gps_sidecar is not None or imu_sidecar is not None or (
        track_source is not None and track_source.suffix == ".json"
    )
    if using_json_sources and metadata is not None:
        epoch = capture_time_epoch(metadata)
        if epoch is not None:
            epoch_offset = epoch

    if epoch_offset:
        timestamps_out = timestamps + epoch_offset
    else:
        timestamps_out = timestamps

    embedding_dims: dict[str, int] = {}
    runner_outputs: list[tuple[str, ModelRunner, np.ndarray, np.ndarray | None]] = []
    for group_name, runner in opts.runners:
        cache_path = _embedding_cache_path(opts, group_name, runner)
        cache_hit = False
        if cache_path.exists() and not opts.force_reembed:
            try:
                cached_emb, cached_maps = _load_embeddings_cache(cache_path)
            except Exception:  # noqa: BLE001
                cached_emb = None  # type: ignore[assignment]
                cached_maps = None
            else:
                if cached_emb.shape[0] == timestamps.shape[0]:
                    embeddings, maps = cached_emb, cached_maps
                    cache_hit = True
                    stage_banner(
                        f"embed:{group_name}",
                        f"reused {cache_path.name} ({cached_emb.shape[0]} vectors, dim={cached_emb.shape[1]})",
                    )
        if not cache_hit:
            stage_banner(
                f"embed:{group_name}",
                f"running {runner.model_id} (ckpt={runner.checkpoint}, "
                f"backend={runner.backend}) over {len(frame_paths)} frames",
            )
            progress = make_progress_logger(f"embed:{group_name}", len(frame_paths))
            embeddings, maps = _embed_with_optional_maps(
                runner, frame_paths, opts.batch_size, progress=progress
            )
            try:
                _save_embeddings_cache(cache_path, embeddings, maps)
                stage_banner(
                    f"embed:{group_name}", f"cached -> {cache_path.name}"
                )
            except OSError as exc:
                stage_banner(
                    f"embed:{group_name}",
                    f"warning: failed to write cache {cache_path}: {exc}",
                )
        if embeddings.shape[0] != timestamps.shape[0]:
            raise RuntimeError(
                f"runner for group {group_name!r}: embedding count "
                f"{embeddings.shape[0]} != frame count {timestamps.shape[0]}"
            )
        if embeddings.shape[1] != runner.embedding_dim:
            raise RuntimeError(
                f"runner for group {group_name!r}: embedding_dim mismatch "
                f"({runner.embedding_dim} declared vs {embeddings.shape[1]} produced)"
            )
        embedding_dims[group_name] = runner.embedding_dim
        runner_outputs.append((group_name, runner, embeddings, maps))

    source_video_attr = opts.source_video_attr or str(opts.video.resolve())
    sensor_groups_written: list[str] = []

    with FeaturesWriter(
        opts.output,
        source_video=source_video_attr,
        session_id=opts.session_id,
    ) as writer:
        if gps_sidecar is not None:
            gps_ts = gps_sidecar.timestamps
            if epoch_offset:
                gps_ts = gps_ts + epoch_offset
            writer.write_gps_group(
                timestamps=gps_ts,
                lat=gps_sidecar.lat,
                lng=gps_sidecar.lng,
                rate_hz_nominal=None,
            )
            sensor_groups_written.append("gps")
        if imu_sidecar is not None:
            imu_ts = imu_sidecar.timestamps
            if epoch_offset:
                imu_ts = imu_ts + epoch_offset
            writer.write_imu_group(
                timestamps=imu_ts,
                accel=imu_sidecar.accel,
                gyro=imu_sidecar.gyro,
                rate_hz_nominal=None,
            )
            sensor_groups_written.append("imu")

        for group_name, runner, embeddings, maps in runner_outputs:
            spec = schema.ModelGroupSpec(
                model=runner.model_id,
                checkpoint=runner.checkpoint,
                embedding_dim=runner.embedding_dim,
                sample_fps=opts.sample_fps,
                sampling=f"video_fps={opts.sample_fps}",
                preprocess=runner.preprocess,
                normalized=runner.normalized,
                patch_grid=runner.patch_grid,
                interpolation=interpolation,
            )
            attention_maps = maps if (
                runner.patch_grid is not None and maps is not None and maps.ndim == 3
            ) else None
            prediction_maps = None  # Phase 4: V-JEPA 2 prediction maps.
            writer.write_model_group(
                group_name,
                spec=spec,
                timestamps=timestamps_out,
                lat=lats,
                lng=lngs,
                embeddings=embeddings,
                attention_maps=attention_maps,
                prediction_maps=prediction_maps,
            )

    stage_banner("write", f"wrote {opts.output.name}")
    if not opts.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return ExtractResult(
        features_path=opts.output,
        group_names=[name for name, _ in opts.runners],
        sample_fps=opts.sample_fps,
        segment_sec=opts.segment_sec,
        timestamps=timestamps,
        embedding_dims=embedding_dims,
        frame_count=int(timestamps.shape[0]),
        duration_sec=duration,
        track_mode=track_mode,
        track_source=track_source,
        track_source_group=track_source_group,
        sensor_groups_written=sensor_groups_written,
    )
