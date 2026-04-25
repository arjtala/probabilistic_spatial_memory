#!/usr/bin/env python3
"""Minimal E5 demo: CLIP text query -> psm --similar-to -> retrieved intervals."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore", message=r".*joblib will operate in serial mode.*"
)

from transformers import AutoProcessor, CLIPModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSM_BINARY = REPO_ROOT / "targets" / "psm"
DEFAULT_GROUP = "clip"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CELL_STEP_DEG = 0.02
DEFAULT_GRID_COLUMNS = 128
BASE_LAT = 37.0
BASE_LNG = -122.0


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a video, embed frames + query in CLIP, write a native PSM "
            "HDF5 group, run psm --similar-to, and print the retrieved intervals."
        )
    )
    parser.add_argument("video", type=Path, help="Video to index.")
    parser.add_argument("query", help="Text query to retrieve from the video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Artifact directory. Defaults to a fresh temp dir that is kept after "
            "the run."
        ),
    )
    parser.add_argument(
        "--psm-binary",
        type=Path,
        default=DEFAULT_PSM_BINARY,
        help=f"Path to the built psm CLI (default: {DEFAULT_PSM_BINARY}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face CLIP checkpoint (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"HDF5 group name to write/read (default: {DEFAULT_GROUP}).",
    )
    parser.add_argument(
        "--sample-fps",
        type=positive_float,
        default=1.0,
        help="Frame sampling rate passed to ffmpeg (default: 1.0).",
    )
    parser.add_argument(
        "--segment-sec",
        type=positive_float,
        default=2.0,
        help=(
            "Pseudo-cell duration. Frames in the same segment share one synthetic "
            "H3 cell so PSM returns narrow temporal intervals (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--time-window-sec",
        type=positive_float,
        help="PSM time window. Defaults to --segment-sec.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=16,
        help="CLIP image batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: auto, cpu, cuda, or mps (default: auto).",
    )
    parser.add_argument(
        "--top",
        type=positive_int,
        default=5,
        help="Maximum number of retrieved intervals to show (default: 5).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10,
        help="PSM H3 resolution (default: 10).",
    )
    parser.add_argument(
        "--precision",
        type=positive_int,
        default=10,
        help="PSM HLL precision bits (default: 10).",
    )
    parser.add_argument(
        "--exemplars",
        type=int,
        default=0,
        help=(
            "Per-tile exemplar reservoir for psm --similar-to. 0 auto-sizes to at "
            "least the sampled frames per segment."
        ),
    )
    parser.add_argument(
        "--grid-columns",
        type=positive_int,
        default=DEFAULT_GRID_COLUMNS,
        help="Width of the synthetic snake-grid track (default: 128).",
    )
    parser.add_argument(
        "--cell-step-deg",
        type=positive_float,
        default=DEFAULT_CELL_STEP_DEG,
        help="Spacing between synthetic pseudo-cells in degrees (default: 0.02).",
    )
    parser.add_argument(
        "--gps-source",
        type=Path,
        help=(
            "Path to a session HDF5 (e.g. features.h5) with a per-frame track "
            "in a 'dino', 'jepa', or 'gps' group. When set or auto-detected "
            "next to the video, real GPS is interpolated onto CLIP frame "
            "timestamps so retrieved cells are real H3 cells, not a synthetic "
            "snake-grid. Auto-detect: <video_dir>/features.h5."
        ),
    )
    parser.add_argument(
        "--no-gps",
        action="store_true",
        help="Force the synthetic snake-grid even if real GPS is available.",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-extract frames and rebuild the HDF5 even if artifacts already exist.",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep sampled JPEGs under output-dir/frames.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print ffmpeg and psm commands before running them.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, verbose: bool = False) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_tool(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Required executable '{name}' was not found on PATH")


def ffprobe_duration(video_path: Path, *, verbose: bool = False) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = run_command(cmd, verbose=verbose)
    except RuntimeError:
        return None
    payload = json.loads(result.stdout)
    raw = payload.get("format", {}).get("duration")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def extract_frames(
    video_path: Path,
    sample_fps: float,
    frames_dir: Path,
    *,
    verbose: bool = False,
) -> list[Path]:
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps}",
        "-start_number",
        "0",
        str(frames_dir / "frame_%06d.jpg"),
    ]
    run_command(cmd, verbose=verbose)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError("ffmpeg produced no sampled frames")
    return frame_paths


def load_rgb_images(paths: list[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def encode_images(
    model: CLIPModel,
    processor: AutoProcessor,
    device: torch.device,
    frame_paths: list[Path],
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            images = load_rgb_images(batch_paths)
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            feats = model.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            batches.append(feats.detach().cpu().to(torch.float32).numpy())
    return np.concatenate(batches, axis=0)


def encode_text(
    model: CLIPModel,
    processor: AutoProcessor,
    device: torch.device,
    query: str,
) -> np.ndarray:
    with torch.inference_mode():
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats[0].detach().cpu().to(torch.float32).numpy()


def load_session_track(
    features_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    """Load (relative_seconds, lat, lng, source_group) from a session HDF5.

    Tries 'dino' first (per-frame aligned, ~30 fps), then 'jepa', then the raw
    'gps' group (~1 fps). All paths are normalized to relative seconds since
    the first sample so the caller can match against CLIP frame timestamps
    without needing the video's absolute start time. Returns None if no
    usable group is found.
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
                # Drop any non-finite GPS samples (Aria captures occasionally
                # produce NaN lat/lng during fix loss).
                mask = np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lng)
                if not mask.any():
                    continue
                ts = ts[mask]
                lat = lat[mask]
                lng = lng[mask]
                if ts.size == 0:
                    continue
                rel = ts - ts[0]
                return rel, lat, lng, candidate
    except (OSError, KeyError):
        return None
    return None


def map_frames_to_gps_track(
    frame_timestamps: np.ndarray,
    track_rel_ts: np.ndarray,
    track_lat: np.ndarray,
    track_lng: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate a GPS track onto CLIP frame timestamps."""
    if track_rel_ts.size == 0:
        raise RuntimeError("GPS track is empty; cannot interpolate")
    clipped = np.clip(frame_timestamps, track_rel_ts[0], track_rel_ts[-1])
    lats = np.interp(clipped, track_rel_ts, track_lat)
    lngs = np.interp(clipped, track_rel_ts, track_lng)
    return lats, lngs


def synthetic_track(
    timestamps: np.ndarray,
    segment_sec: float,
    grid_columns: int,
    cell_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    lats = np.empty(timestamps.shape[0], dtype=np.float64)
    lngs = np.empty(timestamps.shape[0], dtype=np.float64)
    segment_ids = np.floor(timestamps / segment_sec).astype(np.int64)
    max_segment = int(segment_ids.max()) if segment_ids.size else 0

    for idx, segment in enumerate(segment_ids):
        row, col = divmod(int(segment), grid_columns)
        if row % 2 == 1:
            col = grid_columns - 1 - col
        lat = BASE_LAT + row * cell_step_deg
        lng = BASE_LNG + col * cell_step_deg
        if lat > 89.0 or lng < -179.0 or lng > 179.0:
            raise RuntimeError(
                "Synthetic track overflowed valid lat/lng bounds; increase "
                "--grid-columns or reduce --cell-step-deg"
            )
        lats[idx] = lat
        lngs[idx] = lng
    return lats, lngs, max_segment + 1


def write_hdf5(
    features_path: Path,
    group_name: str,
    timestamps: np.ndarray,
    lats: np.ndarray,
    lngs: np.ndarray,
    embeddings: np.ndarray,
    *,
    video_path: Path,
    model_name: str,
    sample_fps: float,
    segment_sec: float,
    time_window_sec: float,
    grid_columns: int,
    cell_step_deg: float,
    track_mode: str,
    gps_source: Path | None,
    gps_source_group: str | None,
) -> None:
    with h5py.File(features_path, "w") as handle:
        group = handle.create_group(group_name)
        group.create_dataset("timestamps", data=timestamps, dtype=np.float64)
        group.create_dataset("lat", data=lats, dtype=np.float64)
        group.create_dataset("lng", data=lngs, dtype=np.float64)
        group.create_dataset("embeddings", data=embeddings, dtype=np.float32)
        group.attrs["source_video"] = str(video_path.resolve())
        group.attrs["clip_model"] = model_name
        group.attrs["sample_fps"] = sample_fps
        group.attrs["segment_sec"] = segment_sec
        group.attrs["time_window_sec"] = time_window_sec
        group.attrs["track_mode"] = track_mode
        if track_mode == "synthetic_snake_grid":
            group.attrs["grid_columns"] = grid_columns
            group.attrs["cell_step_deg"] = cell_step_deg
        if gps_source is not None:
            group.attrs["gps_source"] = str(gps_source.resolve())
        if gps_source_group is not None:
            group.attrs["gps_source_group"] = gps_source_group


def read_feature_metadata(features_path: Path, group_name: str) -> dict[str, object]:
    with h5py.File(features_path, "r") as handle:
        if group_name not in handle:
            raise RuntimeError(f"HDF5 group '{group_name}' not found in {features_path}")
        group = handle[group_name]
        timestamps = np.asarray(group["timestamps"], dtype=np.float64)
        if timestamps.size == 0:
            raise RuntimeError(f"HDF5 group '{group_name}' has no timestamps")
        return {
            "timestamps": timestamps,
            "embedding_dim": int(group["embeddings"].shape[1]),
            "clip_model": group.attrs.get("clip_model"),
            "segment_sec": float(group.attrs.get("segment_sec", 0.0) or 0.0),
            "time_window_sec": float(group.attrs.get("time_window_sec", 0.0) or 0.0),
            "sample_fps": float(group.attrs.get("sample_fps", 0.0) or 0.0),
            "track_mode": attr_text(group.attrs.get("track_mode"))
            or "synthetic_snake_grid",
            "gps_source": attr_text(group.attrs.get("gps_source")),
            "gps_source_group": attr_text(group.attrs.get("gps_source_group")),
        }


def attr_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def capacity_for_timestamps(timestamps: np.ndarray, time_window_sec: float) -> int:
    first = float(timestamps[0])
    last = float(timestamps[-1])
    span = max(0.0, last - first)
    return max(1, int(math.ceil(span / time_window_sec)) + 1)


def exemplar_capacity(
    args: argparse.Namespace, sample_fps: float, segment_sec: float
) -> int:
    if args.exemplars > 0:
        return args.exemplars
    return max(8, int(math.ceil(sample_fps * segment_sec)))


def write_query_vector(query_vec: np.ndarray, query_path: Path) -> None:
    np.asarray(query_vec, dtype="<f4").tofile(query_path)


def run_psm_query(
    psm_binary: Path,
    features_path: Path,
    group_name: str,
    query_path: Path,
    *,
    time_window_sec: float,
    capacity: int,
    resolution: int,
    precision: int,
    top: int,
    exemplars: int,
    verbose: bool = False,
) -> dict[str, object]:
    cmd = [
        str(psm_binary),
        "-f",
        str(features_path),
        "-g",
        group_name,
        "-t",
        f"{time_window_sec:.6f}",
        "-r",
        str(resolution),
        "-C",
        str(capacity),
        "-p",
        str(precision),
        "--top",
        str(top),
        "--exemplars",
        str(exemplars),
        "--similar-to",
        str(query_path),
        "-j",
    ]
    result = run_command(cmd, verbose=verbose)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"psm returned non-JSON output:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        ) from exc


def format_seconds(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def print_summary(
    args: argparse.Namespace,
    work_dir: Path,
    features_path: Path,
    query_path: Path,
    results_path: Path,
    timestamps: np.ndarray,
    duration_sec: float,
    sample_fps: float,
    segment_sec: float,
    time_window_sec: float,
    capacity: int,
    embedding_dim: int,
    exemplars: int,
    psm_payload: dict[str, object],
    feature_meta: dict[str, object],
) -> None:
    results = psm_payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("psm JSON payload missing results list")

    sample_count = timestamps.shape[0]
    print(f"Video: {args.video}")
    print(f"Query: {args.query}")
    print(f"Artifacts: {work_dir}")
    print(f"Features: {features_path}")
    print(f"Query vector: {query_path}")
    print(f"PSM JSON: {results_path}")
    print(
        f"Samples: {sample_count} frames at {sample_fps:.3f} fps, "
        f"embedding_dim={embedding_dim}"
    )
    print(
        f"Duration: {format_seconds(duration_sec)}  "
        f"segment_sec={segment_sec:.3f}  "
        f"time_window_sec={time_window_sec:.3f}  "
        f"capacity={capacity}  exemplars={exemplars}"
    )
    track_mode = str(feature_meta.get("track_mode") or "synthetic_snake_grid")
    if track_mode == "real_gps":
        gps_source = feature_meta.get("gps_source")
        gps_group = feature_meta.get("gps_source_group")
        suffix = f" ({gps_group} group)" if gps_group else ""
        print(f"Track: real GPS from {gps_source}{suffix}")
    else:
        print("Track: synthetic snake-grid (no GPS available)")

    if not results:
        print("No similar intervals returned by psm.")
        return

    print("Retrieved intervals:")
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        t_min = float(item.get("t_min", 0.0))
        t_max = float(item.get("t_max", 0.0))
        similarity = float(item.get("similarity", 0.0))
        exemplar_t = float(item.get("exemplar_t", 0.0))
        sample_hits = int(np.count_nonzero((timestamps >= t_min) & (timestamps <= t_max)))
        print(
            f"{idx:2d}. sim={similarity:.4f}  "
            f"t=[{format_seconds(t_min)}, {format_seconds(t_max)}]  "
            f"exemplar={format_seconds(exemplar_t)}  "
            f"samples={sample_hits}  "
            f"cell={item.get('cell', '?')}"
        )


def main() -> int:
    args = parse_args()
    requested_time_window_sec = args.time_window_sec

    if args.video.is_dir():
        raise RuntimeError("video path must point to a single video file, not a directory")
    if not args.video.exists():
        raise RuntimeError(f"video not found: {args.video}")
    if not (0 <= args.resolution <= 15):
        raise RuntimeError("--resolution must be in [0, 15]")
    if args.exemplars < 0:
        raise RuntimeError("--exemplars must be >= 0")

    psm_binary = args.psm_binary
    if not psm_binary.exists():
        raise RuntimeError(f"psm binary not found: {psm_binary}")

    work_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="psm_e5_")).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"
    features_path = work_dir / "clip_features.h5"
    query_path = work_dir / "query.f32"
    results_path = work_dir / "psm_results.json"
    manifest_path = work_dir / "manifest.json"

    device = resolve_device(args.device)
    processor: AutoProcessor | None = None
    model: CLIPModel | None = None
    feature_meta: dict[str, object]
    duration_sec = ffprobe_duration(args.video, verbose=args.verbose)
    if features_path.exists() and not args.force_reembed:
        feature_meta = read_feature_metadata(features_path, args.group)
        existing_model = attr_text(feature_meta.get("clip_model"))
        if existing_model and existing_model != args.model:
            raise RuntimeError(
                f"{features_path} was built with clip_model={existing_model!r}; "
                f"rerun with --model {existing_model!s} or pass --force-reembed"
            )
        sample_fps = float(feature_meta.get("sample_fps", 0.0) or args.sample_fps)
        segment_sec = float(feature_meta.get("segment_sec", 0.0) or args.segment_sec)
        time_window_sec = (
            requested_time_window_sec
            or float(feature_meta.get("time_window_sec", 0.0) or 0.0)
            or segment_sec
        )
    else:
        ensure_tool("ffmpeg")
        sample_fps = args.sample_fps
        segment_sec = args.segment_sec
        time_window_sec = requested_time_window_sec or segment_sec
        processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
        model = CLIPModel.from_pretrained(args.model)
        model.eval().to(device)

        frame_paths = extract_frames(
            args.video, sample_fps, frames_dir, verbose=args.verbose
        )
        timestamps = np.arange(len(frame_paths), dtype=np.float64) / sample_fps
        embeddings = encode_images(
            model, processor, device, frame_paths, args.batch_size
        )
        if embeddings.shape[0] != timestamps.shape[0]:
            raise RuntimeError("frame count and embedding count diverged")

        gps_source: Path | None = None
        track: tuple[np.ndarray, np.ndarray, np.ndarray, str] | None = None
        if not args.no_gps:
            gps_source = args.gps_source
            if gps_source is None:
                candidate = args.video.parent / "features.h5"
                if candidate.exists():
                    gps_source = candidate
            if gps_source is not None:
                track = load_session_track(gps_source)
                if track is None and args.gps_source is not None:
                    raise RuntimeError(
                        f"--gps-source {gps_source} has no usable dino/jepa/gps "
                        f"group with timestamps + lat + lng"
                    )

        if track is not None:
            track_rel_ts, track_lat, track_lng, track_group = track
            lats, lngs = map_frames_to_gps_track(
                timestamps, track_rel_ts, track_lat, track_lng
            )
            track_mode = "real_gps"
            gps_source_group = track_group
            segment_count = 0
        else:
            lats, lngs, segment_count = synthetic_track(
                timestamps, segment_sec, args.grid_columns, args.cell_step_deg
            )
            track_mode = "synthetic_snake_grid"
            gps_source = None
            gps_source_group = None

        write_hdf5(
            features_path,
            args.group,
            timestamps,
            lats,
            lngs,
            embeddings,
            video_path=args.video,
            model_name=args.model,
            sample_fps=sample_fps,
            segment_sec=segment_sec,
            time_window_sec=time_window_sec,
            grid_columns=args.grid_columns,
            cell_step_deg=args.cell_step_deg,
            track_mode=track_mode,
            gps_source=gps_source,
            gps_source_group=gps_source_group,
        )
        feature_meta = {
            "timestamps": timestamps,
            "embedding_dim": int(embeddings.shape[1]),
            "clip_model": args.model,
            "segment_sec": segment_sec,
            "time_window_sec": time_window_sec,
            "sample_fps": sample_fps,
            "segment_count": segment_count,
            "track_mode": track_mode,
            "gps_source": str(gps_source) if gps_source else None,
            "gps_source_group": gps_source_group,
        }
        if not args.keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)

    if processor is None or model is None:
        processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
        model = CLIPModel.from_pretrained(args.model)
        model.eval().to(device)
    query_vec = encode_text(model, processor, device, args.query)
    write_query_vector(query_vec, query_path)

    timestamps = np.asarray(feature_meta["timestamps"], dtype=np.float64)
    if duration_sec is None:
        duration_sec = float(timestamps[-1]) if timestamps.size else 0.0
    capacity = capacity_for_timestamps(timestamps, time_window_sec)
    exemplars = exemplar_capacity(args, sample_fps, segment_sec)
    psm_payload = run_psm_query(
        psm_binary,
        features_path,
        args.group,
        query_path,
        time_window_sec=time_window_sec,
        capacity=capacity,
        resolution=args.resolution,
        precision=args.precision,
        top=args.top,
        exemplars=exemplars,
        verbose=args.verbose,
    )

    results_path.write_text(json.dumps(psm_payload, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "video": str(args.video.resolve()),
        "query": args.query,
        "artifacts_dir": str(work_dir),
        "features_path": str(features_path),
        "query_path": str(query_path),
        "psm_results_path": str(results_path),
        "psm_binary": str(psm_binary),
        "model": args.model,
        "sample_fps": sample_fps,
        "segment_sec": segment_sec,
        "time_window_sec": time_window_sec,
        "capacity": capacity,
        "top": args.top,
        "resolution": args.resolution,
        "precision": args.precision,
        "exemplars": exemplars,
        "device": str(device),
        "track_mode": feature_meta.get("track_mode"),
        "gps_source": feature_meta.get("gps_source"),
        "gps_source_group": feature_meta.get("gps_source_group"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print_summary(
        args,
        work_dir,
        features_path,
        query_path,
        results_path,
        timestamps,
        duration_sec,
        sample_fps,
        segment_sec,
        time_window_sec,
        capacity,
        int(feature_meta["embedding_dim"]),
        exemplars,
        psm_payload,
        feature_meta,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
