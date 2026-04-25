"""CLI entrypoint for psm-extraction.

Subcommands:
- migrate: in-place v1 → v2 attr back-fill (Phase 1)
- extract: video → CLIP embeddings → v2 features.h5 (Phase 2)
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__ as PACKAGE_VERSION
from .migrate import migrate_v1_to_v2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="psm-extraction",
        description="Produce / maintain features.h5 files for the PSM C engine.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"psm-extraction {PACKAGE_VERSION}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    migrate = sub.add_parser(
        "migrate",
        help="Add schema-v2 attrs in place to an existing v1 features.h5.",
    )
    migrate.add_argument("path", type=Path, help="features.h5 to upgrade in place.")
    migrate.add_argument(
        "--producer-version",
        default="0.1.0",
        help="Producer version string to record under the root attr.",
    )
    migrate.add_argument(
        "--source-video",
        help="Optional source-video path or hash to record at root.",
    )
    migrate.add_argument(
        "--session-id",
        help="Optional Aria session id to record at root.",
    )
    migrate.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the JSON report on stdout.",
    )

    extract = sub.add_parser(
        "extract",
        help="Run a model over a video and write a v2 features.h5.",
    )
    extract.add_argument("--video", type=Path, required=True, help="Source video.")
    extract.add_argument(
        "--output", type=Path, required=True, help="features.h5 to write."
    )
    extract.add_argument(
        "--models",
        default="clip",
        help="Comma-separated model family names. Supported: clip, dino, jepa.",
    )
    extract.add_argument(
        "--checkpoint",
        action="append",
        help=(
            "Per-family checkpoint override of the form FAMILY:CHECKPOINT, "
            "e.g. --checkpoint dino:facebook/dinov2-large. May be passed "
            "multiple times to override multiple families."
        ),
    )
    extract.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "pytorch", "mlx", "cpu"],
        help="Compute backend (auto picks CUDA > MLX-on-Apple > MPS > CPU).",
    )
    extract.add_argument(
        "--device",
        default="auto",
        help="PyTorch device override: auto | cuda | mps | cpu.",
    )
    extract.add_argument("--group", default="clip", help="HDF5 group name to write.")
    extract.add_argument("--sample-fps", type=float, default=1.0)
    extract.add_argument("--segment-sec", type=float, default=2.0)
    extract.add_argument("--batch-size", type=int, default=16)
    extract.add_argument(
        "--gps-source",
        type=Path,
        help="HDF5 with a dino/jepa/gps group; auto-detect <video_dir>/features.h5.",
    )
    extract.add_argument(
        "--gps-json",
        type=Path,
        help="Aria-style gps.json sidecar; auto-detect <video_dir>/gps.json.",
    )
    extract.add_argument(
        "--imu-json",
        type=Path,
        help="Aria-style imu.json sidecar; auto-detect <video_dir>/imu.json.",
    )
    extract.add_argument(
        "--metadata-json",
        type=Path,
        help="Aria metadata.json (used to derive the absolute timestamp epoch).",
    )
    extract.add_argument(
        "--no-gps",
        action="store_true",
        help="Force the synthetic snake-grid even if real GPS is available.",
    )
    extract.add_argument("--grid-columns", type=int, default=128)
    extract.add_argument("--cell-step-deg", type=float, default=0.02)
    extract.add_argument("--keep-frames", action="store_true")
    extract.add_argument("--frames-dir", type=Path)
    extract.add_argument("--source-video-attr", help="Override the root attr.")
    extract.add_argument("--session-id", help="Aria session id to record at root.")
    extract.add_argument("--verbose", action="store_true")
    extract.add_argument(
        "--force-reextract",
        action="store_true",
        help="Wipe and re-run ffmpeg even if a matching frames cache exists.",
    )
    extract.add_argument(
        "--force-reembed",
        action="store_true",
        help="Re-run model inference even when a cached embedding file exists.",
    )
    extract.add_argument(
        "--cache-dir",
        type=Path,
        help="Where per-model embedding caches go; defaults to <output>.parent.",
    )

    return parser


def _handle_migrate(args: argparse.Namespace) -> int:
    extras: dict[str, str] = {}
    if args.source_video:
        extras["source_video"] = args.source_video
    if args.session_id:
        extras["session_id"] = args.session_id
    report = migrate_v1_to_v2(
        args.path,
        producer_version=args.producer_version,
        extra_root_attrs=extras or None,
    )
    if not args.quiet:
        print(json.dumps(report, indent=2))
    return 0


def _handle_extract(args: argparse.Namespace) -> int:
    from .extract import ExtractOptions, extract
    from .models import SUPPORTED_FAMILIES, make_runner

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    if not requested:
        raise SystemExit("--models cannot be empty")
    unsupported = [m for m in requested if m not in SUPPORTED_FAMILIES]
    if unsupported:
        raise SystemExit(
            f"unsupported model families: {unsupported}; "
            f"supported: {list(SUPPORTED_FAMILIES)}"
        )

    checkpoints: dict[str, str] = {}
    for entry in args.checkpoint or []:
        if ":" not in entry:
            raise SystemExit(
                f"--checkpoint must be of the form FAMILY:PATH; got {entry!r}"
            )
        family, path = entry.split(":", 1)
        if family not in requested:
            raise SystemExit(
                f"--checkpoint {entry!r} references {family!r}, which is not in --models"
            )
        checkpoints[family] = path

    runners: list[tuple[str, object]] = []
    try:
        for family in requested:
            runner = make_runner(
                family,
                checkpoint=checkpoints.get(family),
                backend=args.backend,
                device=args.device,
            )
            runners.append((family, runner))

        result = extract(
            ExtractOptions(
                video=args.video,
                output=args.output,
                runners=runners,
                sample_fps=args.sample_fps,
                segment_sec=args.segment_sec,
                batch_size=args.batch_size,
                use_gps=not args.no_gps,
                gps_source=args.gps_source,
                gps_json=args.gps_json,
                imu_json=args.imu_json,
                metadata_json=args.metadata_json,
                grid_columns=args.grid_columns,
                cell_step_deg=args.cell_step_deg,
                keep_frames=args.keep_frames,
                frames_dir=args.frames_dir,
                source_video_attr=args.source_video_attr,
                session_id=args.session_id,
                verbose=args.verbose,
                force_reextract=args.force_reextract,
                force_reembed=args.force_reembed,
                cache_dir=args.cache_dir,
            )
        )
    finally:
        for _name, runner in runners:
            try:
                runner.close()
            except Exception:  # noqa: BLE001
                pass

    payload = {
        "features_path": str(result.features_path),
        "group_names": result.group_names,
        "frame_count": result.frame_count,
        "embedding_dims": result.embedding_dims,
        "sample_fps": result.sample_fps,
        "duration_sec": result.duration_sec,
        "track_mode": result.track_mode,
        "track_source": str(result.track_source) if result.track_source else None,
        "track_source_group": result.track_source_group,
        "sensor_groups_written": result.sensor_groups_written,
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "migrate":
        return _handle_migrate(args)
    if args.command == "extract":
        return _handle_extract(args)
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
