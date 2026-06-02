#!/usr/bin/env python3
"""Convert Nymeria atomic_action.csv -> per-session questions.yaml.

Walks a Nymeria root (`nymeria_partial/` or `nymeria_dataset/`), reads
each session's `narration/atomic_action.csv`, and writes a
questions.yaml under `<out_root>/<session_id>/questions.yaml` in the
schema `eval_lookback.py` already consumes.

Per-narration mapping:
  CSV `start_time`, `end_time` (sec) -> intervals: [[t_start, t_end]]
  CSV last column (free text)        -> query: "..."
  CSV `annotator`                    -> notes: "annotator=..."

Unlike Ego-Exo4D atomic_descriptions, Nymeria narrations come with
real interval bounds so no half-window expansion is needed.

Usage:
  python scripts/nymeria_atomic_to_questions.py \\
    /checkpoint/.../nymeria_partial \\
    --out-root /checkpoint/.../nymeria_atomic

The session_id is the directory name (e.g.
20230607_s0_james_johnson_act0_e72nhq), matching the extractor's
output layout so eval scripts find features.h5 by glob.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extraction"))

from psm_extraction.io.nymeria_narration import (  # noqa: E402
    read_nymeria_root,
    summarize_nymeria_split,
)


def _yaml_escape(s: str) -> str:
    """Minimal YAML escape for double-quoted scalars."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def write_questions_yaml(out_path: Path, session, *, iou_threshold: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Auto-generated from Nymeria atomic_action.csv narrations.",
        "# Source: nymeria.fbresearch v0.0 release; atomic_action protocol.",
        f"# {len(session.narrations)} questions for session={session.session_id}.",
        "#",
        "# Re-generate via scripts/nymeria_atomic_to_questions.py; do not edit",
        "# by hand (edits will be lost on next conversion run).",
        "",
        f"session_id: {session.session_id}",
        "session_start_unix: 0.0",
        f"iou_threshold: {iou_threshold}",
        "",
        "questions:",
    ]
    for i, n in enumerate(session.narrations, start=1):
        lines.append(f"  - id: q{i}")
        lines.append(f'    query: "{_yaml_escape(n.text)}"')
        lines.append("    intervals:")
        lines.append(f"      - [{n.t_start_sec:.3f}, {n.t_end_sec:.3f}]")
        lines.append(
            f'    notes: "request_id={n.request_uid}; annotator={n.annotator_id}; '
            f'gaia_id={n.gaia_id}"'
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Nymeria atomic_action.csv -> per-session questions.yaml.",
    )
    parser.add_argument(
        "nymeria_root", type=Path,
        help="Nymeria session-dir root (nymeria_partial/ or nymeria_dataset/).",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("datasets/nymeria_atomic"),
        help="Output root; one subdir created per session_id.",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.3,
        help="iou_threshold written into each questions.yaml.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of sessions to emit (smoke-test).",
    )
    args = parser.parse_args()

    if not args.nymeria_root.is_dir():
        print(f"ERR: {args.nymeria_root} not a directory", file=sys.stderr)
        return 1

    sessions = read_nymeria_root(args.nymeria_root)
    stats = summarize_nymeria_split(sessions)
    print(
        f"[nymeria] loaded {stats['n_sessions']} sessions, "
        f"{stats['n_narrations']} narrations "
        f"({stats['n_unique_gaia_ids']} unique gaia_ids)",
        file=sys.stderr,
    )
    print(
        f"[nymeria] narrations/session: mean={stats['n_per_session_mean']:.1f}, "
        f"median={stats['n_per_session_median']:.1f}, "
        f"max={stats['n_per_session_max']}",
        file=sys.stderr,
    )
    print(
        f"[nymeria] interval duration sec: median={stats['duration_sec_median']:.2f}, "
        f"mean={stats['duration_sec_mean']:.2f}",
        file=sys.stderr,
    )

    if args.limit is not None:
        sessions = sessions[: args.limit]
        print(f"[nymeria] --limit applied: emitting {len(sessions)} sessions",
              file=sys.stderr)

    args.out_root.mkdir(parents=True, exist_ok=True)
    for s in sessions:
        out_path = args.out_root / s.session_id / "questions.yaml"
        write_questions_yaml(out_path, s, iou_threshold=args.iou_threshold)

    print(f"[nymeria] wrote {len(sessions)} questions.yaml files under {args.out_root}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
