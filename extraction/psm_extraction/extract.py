"""End-to-end orchestrator: video → frames → embeddings → v2 features.h5.

Lives behind a single `extract(opts)` call so both the `python -m
psm_extraction extract` CLI and the `scripts/e5_clip_demo.py` shim use one
code path.
"""

import dataclasses
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from . import schema
from .align import (
    SessionTrack,
    load_session_track,
    map_frames_to_gps,
    synthetic_snake_grid,
)
from .io import extract_frames, video_duration
from .models import ModelRunner
from .writer import FeaturesWriter


@dataclasses.dataclass
class ExtractOptions:
    """Inputs for a single extraction run."""

    video: Path
    output: Path
    runner: ModelRunner
    group_name: str = "clip"
    sample_fps: float = 1.0
    segment_sec: float = 2.0
    batch_size: int = 16
    use_gps: bool = True
    gps_source: Path | None = None
    grid_columns: int = 128
    cell_step_deg: float = 0.02
    keep_frames: bool = False
    frames_dir: Path | None = None
    source_video_attr: str | None = None
    session_id: str | None = None
    verbose: bool = False


@dataclasses.dataclass
class ExtractResult:
    """Summary returned to the caller after a successful extraction."""

    features_path: Path
    group_name: str
    sample_fps: float
    segment_sec: float
    timestamps: np.ndarray
    embedding_dim: int
    frame_count: int
    duration_sec: float | None
    track_mode: str
    track_source: Path | None
    track_source_group: str | None


def _resolve_frames_dir(opts: ExtractOptions) -> Path:
    if opts.frames_dir is not None:
        return opts.frames_dir
    return opts.output.parent / "frames"


def _resolve_track(opts: ExtractOptions) -> tuple[SessionTrack | None, Path | None]:
    if not opts.use_gps:
        return None, None
    candidate = opts.gps_source
    if candidate is None:
        guess = opts.video.parent / "features.h5"
        if guess.exists():
            candidate = guess
    if candidate is None:
        return None, None
    track = load_session_track(candidate)
    if track is None:
        if opts.gps_source is not None:
            raise RuntimeError(
                f"--gps-source {opts.gps_source} has no usable dino/jepa/gps group"
            )
        return None, None
    return track, candidate


def extract(opts: ExtractOptions) -> ExtractResult:
    """Run the full pipeline. Side effect: writes opts.output."""
    if not opts.video.exists():
        raise RuntimeError(f"video not found: {opts.video}")
    if opts.sample_fps <= 0.0:
        raise ValueError("sample_fps must be > 0")
    if opts.segment_sec <= 0.0:
        raise ValueError("segment_sec must be > 0")
    if opts.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if opts.group_name in {"gps", "imu"}:
        raise ValueError(f"{opts.group_name!r} is reserved for sensor groups")

    duration = video_duration(opts.video, verbose=opts.verbose)

    frames_dir = _resolve_frames_dir(opts)
    frame_paths = extract_frames(
        opts.video, opts.sample_fps, frames_dir, verbose=opts.verbose
    )
    timestamps = np.arange(len(frame_paths), dtype=np.float64) / opts.sample_fps
    embeddings = opts.runner.embed_images(frame_paths, batch_size=opts.batch_size)
    if embeddings.shape[0] != timestamps.shape[0]:
        raise RuntimeError("embedding count diverged from frame count")
    if embeddings.shape[1] != opts.runner.embedding_dim:
        raise RuntimeError(
            f"runner reported embedding_dim={opts.runner.embedding_dim} but "
            f"produced {embeddings.shape[1]}"
        )

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

    spec = schema.ModelGroupSpec(
        model=opts.runner.model_id,
        checkpoint=opts.runner.checkpoint,
        embedding_dim=opts.runner.embedding_dim,
        sample_fps=opts.sample_fps,
        sampling=f"video_fps={opts.sample_fps}",
        preprocess=opts.runner.preprocess,
        normalized=opts.runner.normalized,
        patch_grid=opts.runner.patch_grid,
        interpolation=interpolation,
    )

    source_video_attr = opts.source_video_attr or str(opts.video.resolve())

    with FeaturesWriter(
        opts.output,
        source_video=source_video_attr,
        session_id=opts.session_id,
    ) as writer:
        writer.write_model_group(
            opts.group_name,
            spec=spec,
            timestamps=timestamps,
            lat=lats,
            lng=lngs,
            embeddings=embeddings,
        )

    if not opts.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return ExtractResult(
        features_path=opts.output,
        group_name=opts.group_name,
        sample_fps=opts.sample_fps,
        segment_sec=opts.segment_sec,
        timestamps=timestamps,
        embedding_dim=opts.runner.embedding_dim,
        frame_count=int(timestamps.shape[0]),
        duration_sec=duration,
        track_mode=track_mode,
        track_source=track_source,
        track_source_group=track_source_group,
    )
