#!/usr/bin/env python3
"""Minimal E5 demo: CLIP text query -> psm --search -> retrieved intervals.

Thin shim over the in-tree extraction package. The first run builds (or
auto-detects from a sibling features.h5) a CLIP-embedded clip group and
caches it under --output-dir. Subsequent runs against the same dir reuse
the cache and only re-embed the new query text.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "extraction"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from psm_extraction.extract import ExtractOptions, extract
from psm_extraction.models import make_runner

DEFAULT_PSM_BINARY = REPO_ROOT / "targets" / "psm"
DEFAULT_GROUP = "clip"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"


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
            "Sample a video, embed frames + query in CLIP, write a v2 PSM "
            "HDF5 group via the in-tree psm_extraction package, run "
            "psm --search, and print the retrieved intervals."
        )
    )
    parser.add_argument("video", type=Path, help="Video to index.")
    parser.add_argument("query", help="Text query to retrieve from the video.")
    parser.add_argument("--output-dir", type=Path, help="Artifact directory.")
    parser.add_argument("--psm-binary", type=Path, default=DEFAULT_PSM_BINARY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--sample-fps", type=positive_float, default=1.0)
    parser.add_argument("--segment-sec", type=positive_float, default=2.0)
    parser.add_argument("--time-window-sec", type=positive_float)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--backend", default="auto", choices=["auto", "pytorch", "mlx", "cpu"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top", type=positive_int, default=5)
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--precision", type=positive_int, default=10)
    parser.add_argument("--exemplars", type=int, default=0)
    parser.add_argument("--grid-columns", type=positive_int, default=128)
    parser.add_argument("--cell-step-deg", type=positive_float, default=0.02)
    parser.add_argument("--gps-source", type=Path)
    parser.add_argument("--no-gps", action="store_true")
    parser.add_argument("--force-reembed", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def run_psm(
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
    verbose: bool,
) -> dict:
    cmd = [
        str(psm_binary),
        "-f", str(features_path),
        "-g", group_name,
        "-t", f"{time_window_sec:.6f}",
        "-r", str(resolution),
        "-C", str(capacity),
        "-p", str(precision),
        "--top", str(top),
        "--exemplars", str(exemplars),
        "--search", str(query_path),
        "-j",
    ]
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"psm failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def read_cached_metadata(features_path: Path, group_name: str) -> dict:
    with h5py.File(features_path, "r") as h:
        if group_name not in h:
            raise RuntimeError(f"group {group_name!r} not in {features_path}")
        group = h[group_name]
        timestamps = np.asarray(group["timestamps"], dtype=np.float64)
        if timestamps.size == 0:
            raise RuntimeError(f"{features_path} has empty timestamps")
        clip_model = group.attrs.get("clip_model")
        if clip_model is None:
            clip_model = group.attrs.get("checkpoint")
        if isinstance(clip_model, bytes):
            clip_model = clip_model.decode("utf-8", errors="replace")
        track_mode = group.attrs.get("track_mode")
        if isinstance(track_mode, bytes):
            track_mode = track_mode.decode("utf-8", errors="replace")
        return {
            "timestamps": timestamps,
            "embedding_dim": int(group["embeddings"].shape[1]),
            "checkpoint": str(clip_model) if clip_model else None,
            "sample_fps": float(group.attrs.get("sample_fps", 0.0) or 0.0),
            "track_mode": str(track_mode) if track_mode else "synthetic_snake_grid",
        }


def capacity_for_timestamps(timestamps: np.ndarray, window: float) -> int:
    span = float(timestamps[-1] - timestamps[0]) if timestamps.size else 0.0
    return max(1, int(math.ceil(span / window)) + 1)


def exemplar_count(args: argparse.Namespace) -> int:
    if args.exemplars > 0:
        return args.exemplars
    return max(8, int(math.ceil(args.sample_fps * args.segment_sec)))


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise RuntimeError(f"video not found: {args.video}")
    if not args.psm_binary.exists():
        raise RuntimeError(f"psm binary not found: {args.psm_binary}")

    work_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix="psm_e5_")).resolve()
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    features_path = work_dir / "clip_features.h5"
    query_path = work_dir / "query.f32"
    results_path = work_dir / "psm_results.json"
    manifest_path = work_dir / "manifest.json"

    runner = make_runner(
        "clip",
        checkpoint=args.model,
        backend=args.backend,
        device=args.device,
    )
    try:
        if features_path.exists() and not args.force_reembed:
            meta = read_cached_metadata(features_path, args.group)
            if meta["checkpoint"] and meta["checkpoint"] != args.model:
                raise RuntimeError(
                    f"{features_path} was built with checkpoint={meta['checkpoint']!r}; "
                    f"rerun with --model {meta['checkpoint']} or --force-reembed"
                )
            sample_fps = meta["sample_fps"] or args.sample_fps
            timestamps = meta["timestamps"]
            embedding_dim = meta["embedding_dim"]
            track_mode = meta["track_mode"]
        else:
            result = extract(
                ExtractOptions(
                    video=args.video,
                    output=features_path,
                    runners=[(args.group, runner)],
                    sample_fps=args.sample_fps,
                    segment_sec=args.segment_sec,
                    batch_size=args.batch_size,
                    use_gps=not args.no_gps,
                    gps_source=args.gps_source,
                    grid_columns=args.grid_columns,
                    cell_step_deg=args.cell_step_deg,
                    keep_frames=args.keep_frames,
                    frames_dir=work_dir / "frames",
                    verbose=args.verbose,
                )
            )
            sample_fps = result.sample_fps
            timestamps = result.timestamps
            embedding_dim = result.embedding_dims[args.group]
            track_mode = result.track_mode

        query_vec = runner.embed_text(args.query)
        np.asarray(query_vec, dtype="<f4").tofile(query_path)
    finally:
        runner.close()

    time_window = args.time_window_sec or args.segment_sec
    capacity = capacity_for_timestamps(timestamps, time_window)
    exemplars = exemplar_count(args)
    psm_payload = run_psm(
        args.psm_binary,
        features_path,
        args.group,
        query_path,
        time_window_sec=time_window,
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
        "psm_binary": str(args.psm_binary),
        "model": args.model,
        "backend": args.backend,
        "sample_fps": sample_fps,
        "segment_sec": args.segment_sec,
        "time_window_sec": time_window,
        "capacity": capacity,
        "top": args.top,
        "resolution": args.resolution,
        "precision": args.precision,
        "exemplars": exemplars,
        "track_mode": track_mode,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Video: {args.video}")
    print(f"Query: {args.query}")
    print(f"Artifacts: {work_dir}")
    print(f"Features: {features_path}")
    print(f"Query vector: {query_path}")
    print(f"PSM JSON: {results_path}")
    print(
        f"Samples: {timestamps.shape[0]} frames at {sample_fps:.3f} fps, "
        f"embedding_dim={embedding_dim}"
    )
    duration = float(timestamps[-1]) if timestamps.size else 0.0
    print(
        f"Duration: {format_seconds(duration)}  "
        f"segment_sec={args.segment_sec:.3f}  "
        f"time_window_sec={time_window:.3f}  "
        f"capacity={capacity}  exemplars={exemplars}"
    )
    if track_mode == "real_gps":
        print("Track: real GPS interpolated onto CLIP frame timestamps")
    else:
        print("Track: synthetic snake-grid (no GPS available or --no-gps)")

    results = psm_payload.get("results", []) or []
    if not results:
        print("No similar intervals returned by psm.")
        return 0

    print("Retrieved intervals:")
    for idx, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        t_min = float(item.get("t_min", 0.0))
        t_max = float(item.get("t_max", 0.0))
        sim = float(item.get("similarity", 0.0))
        ex_t = float(item.get("exemplar_t", 0.0))
        sample_hits = int(np.count_nonzero((timestamps >= t_min) & (timestamps <= t_max)))
        print(
            f"{idx:2d}. sim={sim:.4f}  "
            f"t=[{format_seconds(t_min)}, {format_seconds(t_max)}]  "
            f"exemplar={format_seconds(ex_t)}  "
            f"samples={sample_hits}  "
            f"cell={item.get('cell', '?')}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
