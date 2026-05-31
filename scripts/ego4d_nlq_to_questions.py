#!/usr/bin/env python3
"""Convert Ego4D nlq_val.json -> per-video datasets/ego4d_nlq/{uid}/questions.yaml.

The output YAML schema matches what eval_lookback.py already reads
for Aria sessions:

  session_id: <video_uid>
  session_start_unix: 0.0
  iou_threshold: 0.3
  questions:
    - id: q1
      query: "..."
      category: "<NLQ template>"
      intervals:
        - [t_start_sec, t_end_sec]
      notes: "annotation_uid=...; clip_uid=..."

One file per source video — matches the one-features.h5-per-video
extraction layout from the Ego4D download.

Usage:
  python scripts/ego4d_nlq_to_questions.py \\
    /checkpoint/.../ego4d/v2/annotations/nlq_val.json \\
    --out-root datasets/ego4d_nlq \\
    [--iou-threshold 0.3]

After running, every video_uid in nlq_val.json that has >=1 valid
query gets a directory at <out_root>/<video_uid>/ with a
questions.yaml inside. The directory is the natural place to drop
the per-video features.h5 later.

Re-running is idempotent — the file gets overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add the extraction package so we can run as a top-level script from
# the repo root without installing it (matches the rest of scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extraction"))

from psm_extraction.io import read_nlq_annotations, summarize_nlq_split  # noqa: E402


def _yaml_escape(s: str) -> str:
    """Minimal YAML string escape suitable for double-quoted scalars.

    PyYAML would handle this for us, but adding it as a dep for a
    one-shot writer is overkill — the only chars we have to worry
    about in NLQ queries are `"`, `\\`, and control chars (rare).
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def write_questions_yaml(out_path: Path, video, *, iou_threshold: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Auto-generated from Ego4D NLQ annotations.",
        "# Source: ego4d-data.org NLQ v2 release.",
        f"# {len(video.queries)} questions for video_uid={video.video_uid}.",
        "#",
        "# Re-generate via scripts/ego4d_nlq_to_questions.py; do not edit by hand",
        "# (edits will be lost on next conversion run).",
        "",
        f"session_id: {video.video_uid}",
        "session_start_unix: 0.0",
        f"iou_threshold: {iou_threshold}",
        "",
        "questions:",
    ]
    for i, q in enumerate(video.queries, start=1):
        lines.append(f"  - id: q{i}")
        lines.append(f'    query: "{_yaml_escape(q.query)}"')
        if q.template:
            lines.append(f'    category: "{_yaml_escape(q.template)}"')
        lines.append("    intervals:")
        lines.append(f"      - [{q.t_start_sec:.3f}, {q.t_end_sec:.3f}]")
        lines.append(f'    notes: "annotation_uid={q.annotation_uid}; clip_uid={q.clip_uid}"')
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Ego4D nlq_val.json -> per-video questions.yaml files."
    )
    parser.add_argument(
        "nlq_json", type=Path,
        help="Path to nlq_val.json or nlq_train.json (NLQ v2 schema).",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("datasets/ego4d_nlq"),
        help="Output root; one subdirectory created per video_uid.",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help="iou_threshold field written into each questions.yaml. "
             "Matches the Aria default; bumping it just changes the "
             "Hit@k cutoff in eval_lookback, not the data.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of videos to emit (smoke-test mode).",
    )
    args = parser.parse_args()

    if not args.nlq_json.exists():
        print(f"ERR: {args.nlq_json} not found", file=sys.stderr)
        return 1

    videos = read_nlq_annotations(args.nlq_json)
    stats = summarize_nlq_split(videos)
    print(f"[nlq] loaded {stats['n_videos']} videos, {stats['n_questions']} questions, "
          f"{stats['n_unique_templates']} templates", file=sys.stderr)
    print(f"[nlq] questions/video: mean={stats['q_per_video_mean']:.1f}, "
          f"median={stats['q_per_video_median']:.1f}, max={stats['q_per_video_max']}",
          file=sys.stderr)
    print(f"[nlq] interval duration sec: median={stats['duration_sec_median']:.2f}, "
          f"mean={stats['duration_sec_mean']:.2f}", file=sys.stderr)

    if args.limit is not None:
        videos = videos[: args.limit]
        print(f"[nlq] --limit applied: emitting {len(videos)} videos", file=sys.stderr)

    args.out_root.mkdir(parents=True, exist_ok=True)
    for video in videos:
        out_path = args.out_root / video.video_uid / "questions.yaml"
        write_questions_yaml(out_path, video, iou_threshold=args.iou_threshold)

    print(f"[nlq] wrote {len(videos)} questions.yaml files under {args.out_root}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
