#!/usr/bin/env python3
"""Convert Ego-Exo4D atomic_descriptions_val.json -> per-take questions.yaml.

Emits one file per Ego-Exo4D take_uid under
datasets/egoexo4d_atomic/<take_uid>/questions.yaml, in the schema
eval_lookback.py already consumes. Each ego-visible atomic description
becomes a question whose ground-truth interval is the small window
around the description's timestamp (default ±1.5s).

The take_uid maps directly to a directory under
/datasets/egoexo4d/v2/takes/, which is where the source MP4 + ego/exo
metadata live — so the per-extraction features.h5 path is already
implied by the manifest.

Usage:
  python scripts/egoexo4d_atomic_to_questions.py \\
    /datasets/egoexo4d/v2/annotations/atomic_descriptions_val.json \\
    --out-root datasets/egoexo4d_atomic \\
    [--half-window-sec 1.5] [--iou-threshold 0.3] [--limit N]

Re-running is idempotent — files get overwritten.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add the extraction package so we can run as a top-level script from
# the repo root without installing it (matches the rest of scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extraction"))

from psm_extraction.io import read_atomic_descriptions, summarize_atomic_split  # noqa: E402


def _yaml_escape(s: str) -> str:
    """Minimal YAML escape for double-quoted scalars."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def write_questions_yaml(out_path: Path, take, *, iou_threshold: float, half_window_sec: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Auto-generated from Ego-Exo4D atomic_descriptions.",
        "# Source: ego-exo4d-data.org v2 release; ego_visible+not-unsure filter.",
        f"# {len(take.descriptions)} questions for take_uid={take.take_uid}",
        f"# (intervals are timestamp +/- {half_window_sec}s).",
        "#",
        "# Re-generate via scripts/egoexo4d_atomic_to_questions.py; do not edit",
        "# by hand (edits will be lost on next conversion run).",
        "",
        f"session_id: {take.take_uid}",
        "session_start_unix: 0.0",
        f"iou_threshold: {iou_threshold}",
        "",
        "questions:",
    ]
    for i, d in enumerate(take.descriptions, start=1):
        lines.append(f"  - id: q{i}")
        lines.append(f'    query: "{_yaml_escape(d.text)}"')
        if d.subject:
            lines.append(f'    category: "subject={_yaml_escape(d.subject)}"')
        lines.append("    intervals:")
        lines.append(f"      - [{d.t_start_sec:.3f}, {d.t_end_sec:.3f}]")
        lines.append(
            f'    notes: "timestamp={d.timestamp_sec:.3f}s; '
            f'annotation_uid={d.annotation_uid}; annotator_id={d.annotator_id}"'
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Ego-Exo4D atomic_descriptions.json -> per-take questions.yaml.",
    )
    parser.add_argument(
        "atomic_json", type=Path,
        help="Path to atomic_descriptions_val.json or _train.json.",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("datasets/egoexo4d_atomic"),
        help="Output root; one subdirectory created per take_uid.",
    )
    parser.add_argument(
        "--half-window-sec", type=float, default=1.5,
        help="Half-width (sec) of the interval put around each point timestamp.",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help="iou_threshold field written into each questions.yaml.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of takes to emit (smoke-test mode).",
    )
    parser.add_argument(
        "--include-non-ego-visible", action="store_true",
        help="Disable the ego_visible filter (default: only ego-visible items "
             "are kept). Use with care — non-ego-visible items can't be "
             "retrieved from ego CLIP features and will hurt apparent accuracy.",
    )
    args = parser.parse_args()

    if not args.atomic_json.exists():
        print(f"ERR: {args.atomic_json} not found", file=sys.stderr)
        return 1

    takes = read_atomic_descriptions(
        args.atomic_json,
        half_window_sec=args.half_window_sec,
        require_ego_visible=not args.include_non_ego_visible,
    )
    stats = summarize_atomic_split(takes)
    print(
        f"[atomic] loaded {stats['n_takes']} takes, "
        f"{stats['n_descriptions']} descriptions "
        f"({stats['n_unique_subjects']} subject labels)",
        file=sys.stderr,
    )
    print(
        f"[atomic] descriptions/take: mean={stats['d_per_take_mean']:.1f}, "
        f"median={stats['d_per_take_median']:.1f}, max={stats['d_per_take_max']}",
        file=sys.stderr,
    )

    if args.limit is not None:
        takes = takes[: args.limit]
        print(f"[atomic] --limit applied: emitting {len(takes)} takes", file=sys.stderr)

    args.out_root.mkdir(parents=True, exist_ok=True)
    for take in takes:
        out_path = args.out_root / take.take_uid / "questions.yaml"
        write_questions_yaml(
            out_path, take,
            iou_threshold=args.iou_threshold,
            half_window_sec=args.half_window_sec,
        )

    print(
        f"[atomic] wrote {len(takes)} questions.yaml files under {args.out_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
