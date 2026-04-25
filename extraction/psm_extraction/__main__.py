"""CLI entrypoint for psm-extraction.

For Phase 1 only the `migrate` subcommand is wired; later phases add
`extract` and `add-group`. Stays small on purpose so phase 1 remains
self-contained and the extraction package can ship without runtime ML deps.
"""

from __future__ import annotations

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "migrate":
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
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
