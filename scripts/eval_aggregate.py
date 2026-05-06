#!/usr/bin/env python3
"""Aggregate eval_lookback.py JSON outputs across sessions.

Reads N JSON records produced by `scripts/eval_lookback.py --out <file>`
and prints a combined evaluation table:

  - Per-session row (mIoU @1, mIoU @k, Hit @k).
  - Combined-across-all-sessions row (mean over the union of questions).
  - Per-category breakdown across all sessions.

Use:

    python scripts/eval_aggregate.py captures/eval_*_clipL_e128.json
    python scripts/eval_aggregate.py --out captures/agg.json captures/eval_*.json
    python scripts/eval_aggregate.py --label-from-features  captures/eval_*.json

`--label-from-features` derives a short session label from the features
filename (e.g. `clip_l_features.h5` → the parent directory name) instead
of using the full path. Convenient when filenames are long.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _summary_field(summary: dict, name: str) -> float:
    """Pull a metric from the summary regardless of which `top` it was run with.

    eval_lookback writes keys like `bucket_miou_at_5`, `exemplar_hit_rate_at_5`
    that include the actual top-k integer. This helper finds the numeric
    suffix dynamically so aggregation is top-independent.
    """
    for key, value in summary.items():
        if key == name or key.startswith(f"{name}_at_"):
            return float(value)
    return 0.0


def _session_label(record: dict, label_from_features: bool) -> str:
    if label_from_features:
        path = Path(str(record.get("features") or ""))
        if path.parent.name:
            return path.parent.name
        return path.stem
    raw = record.get("questions_file") or record.get("features") or "?"
    return Path(str(raw)).stem


def _categorize(records: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        # Only score scored records (negative controls excluded).
        if not r.get("intervals_gt"):
            continue
        cat = r.get("category") or "(uncategorized)"
        grouped[cat].append(r)
    return grouped


def _agg(records: list[dict]) -> dict:
    n = max(len(records), 1)
    return {
        "n": len(records),
        "exemplar_miou_top1": sum(r["exemplar_iou_top1"] for r in records) / n,
        "exemplar_miou_at_k": sum(r["exemplar_iou_at_k"] for r in records) / n,
        "exemplar_hit_rate_at_k": sum(1 for r in records if r["exemplar_hit_at_k"]) / n,
        "bucket_miou_top1": sum(r["bucket_iou_top1"] for r in records) / n,
        "bucket_miou_at_k": sum(r["bucket_iou_at_k"] for r in records) / n,
    }


def _print_seed_summary(
    by_session: dict[str, list[dict]],
    common_top: int | None,
    common_iou_threshold: float | None,
    common_exemplars: int | None,
    common_encoder: str | None,
    out_path: Path | None,
) -> int:
    import statistics

    print()
    print("## Seed-sweep aggregate (mean ± std across seeds)")
    if common_top is not None:
        print(
            f"_top={common_top}, IoU threshold={common_iou_threshold}, "
            f"exemplars={common_exemplars}, encoder=`{common_encoder}`_"
        )
    print()
    print(
        "| session | n scored | seeds | exemplar mIoU @1 | exemplar mIoU @k | "
        "exemplar Hit @k | bucket mIoU @k |"
    )
    print("|---|---|---|---|---|---|---|")

    per_session_summaries: dict[str, dict] = {}
    for label in sorted(by_session):
        summary = _seed_aggregate(by_session[label])
        per_session_summaries[label] = summary
        seeds_str = ",".join(
            str(s["seed"]) for s in summary["per_seed"] if s["seed"] is not None
        )
        print(
            f"| `{label}` | {summary['n_scored']} | {seeds_str or '?'} | "
            f"{summary['exemplar_miou_top1_mean']:.3f} ± {summary['exemplar_miou_top1_std']:.3f} | "
            f"{summary['exemplar_miou_at_k_mean']:.3f} ± {summary['exemplar_miou_at_k_std']:.3f} | "
            f"{summary['exemplar_hit_rate_at_k_mean']:.1%} ± "
            f"{summary['exemplar_hit_rate_at_k_std']:.1%} | "
            f"{summary['bucket_miou_at_k_mean']:.3f} ± {summary['bucket_miou_at_k_std']:.3f} |"
        )

    # Combined: aggregate per-seed across sessions, then mean ± std.
    seeds_set: set = set()
    for runs in by_session.values():
        for r in runs:
            seeds_set.add(r.get("psm_seed"))
    combined_per_seed: list[dict] = []
    for seed in sorted(s for s in seeds_set if s is not None):
        scored: list[dict] = []
        for label, runs in by_session.items():
            for run in runs:
                if run.get("psm_seed") != seed:
                    continue
                scored.extend(
                    [r for r in run.get("records", []) if r.get("intervals_gt")]
                )
        n = max(len(scored), 1)
        combined_per_seed.append({
            "seed": seed,
            "n_scored": len(scored),
            "exemplar_miou_top1": sum(r["exemplar_iou_top1"] for r in scored) / n,
            "exemplar_miou_at_k": sum(r["exemplar_iou_at_k"] for r in scored) / n,
            "exemplar_hit_rate_at_k": sum(1 for r in scored if r["exemplar_hit_at_k"]) / n,
            "bucket_miou_at_k": sum(r["bucket_iou_at_k"] for r in scored) / n,
        })

    if combined_per_seed:
        n_total = combined_per_seed[0]["n_scored"]
        seeds_str = ",".join(str(s["seed"]) for s in combined_per_seed)
        m_top1 = [s["exemplar_miou_top1"] for s in combined_per_seed]
        m_at_k = [s["exemplar_miou_at_k"] for s in combined_per_seed]
        m_hit = [s["exemplar_hit_rate_at_k"] for s in combined_per_seed]
        m_bucket = [s["bucket_miou_at_k"] for s in combined_per_seed]
        n_seeds = len(combined_per_seed)

        def _mean_std(values):
            return statistics.fmean(values), (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )

        miou1_m, miou1_s = _mean_std(m_top1)
        miouk_m, miouk_s = _mean_std(m_at_k)
        hit_m, hit_s = _mean_std(m_hit)
        bucket_m, bucket_s = _mean_std(m_bucket)
        print(
            f"| **combined** | **{n_total}** | **{seeds_str}** | "
            f"**{miou1_m:.3f} ± {miou1_s:.3f}** | "
            f"**{miouk_m:.3f} ± {miouk_s:.3f}** | "
            f"**{hit_m:.1%} ± {hit_s:.1%}** | "
            f"**{bucket_m:.3f} ± {bucket_s:.3f}** |"
        )
        print()
        print(
            f"_combined: {n_total} scored question(s) per seed × "
            f"{n_seeds} seed(s) = {n_total * n_seeds} question-seed evaluations._"
        )

        # Per-seed sub-table for the appendix.
        print()
        print("### Per-seed combined breakdown")
        print()
        print(
            "| seed | n | exemplar mIoU @1 | exemplar mIoU @k | "
            "exemplar Hit @k |"
        )
        print("|---|---|---|---|---|")
        for s in combined_per_seed:
            print(
                f"| {s['seed']} | {s['n_scored']} | "
                f"{s['exemplar_miou_top1']:.3f} | "
                f"{s['exemplar_miou_at_k']:.3f} | "
                f"{s['exemplar_hit_rate_at_k']:.1%} |"
            )

    if out_path:
        out_data = {
            "mode": "by_seed",
            "shared_settings": {
                "top": common_top,
                "iou_threshold": common_iou_threshold,
                "exemplars": common_exemplars,
                "clip_checkpoint": common_encoder,
            },
            "sessions": {
                label: per_session_summaries[label] for label in sorted(per_session_summaries)
            },
            "combined": {
                "per_seed": combined_per_seed,
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_data, indent=2))
        print(f"\n[aggregate] wrote {out_path}", file=sys.stderr)
    return 0


def _seed_aggregate(runs_for_session: list[dict]) -> dict:
    """Mean ± std (sample, ddof=1) of per-seed scalar metrics for one session.

    Each run dict carries a `psm_seed` and the `summary` block. We compute
    mean and std for the four metrics on the per-seed table; std uses
    sample variance (n-1 denominator) so it matches reviewer expectations.
    """
    import statistics

    per_seed = []
    for run in sorted(runs_for_session, key=lambda d: d.get("psm_seed", 0) or 0):
        scored = [r for r in run.get("records", []) if r.get("intervals_gt")]
        n = max(len(scored), 1)
        per_seed.append({
            "seed": run.get("psm_seed"),
            "n_scored": len(scored),
            "exemplar_miou_top1": sum(r["exemplar_iou_top1"] for r in scored) / n,
            "exemplar_miou_at_k": sum(r["exemplar_iou_at_k"] for r in scored) / n,
            "exemplar_hit_rate_at_k": sum(1 for r in scored if r["exemplar_hit_at_k"]) / n,
            "bucket_miou_top1": sum(r["bucket_iou_top1"] for r in scored) / n,
            "bucket_miou_at_k": sum(r["bucket_iou_at_k"] for r in scored) / n,
        })
    metrics = (
        "exemplar_miou_top1",
        "exemplar_miou_at_k",
        "exemplar_hit_rate_at_k",
        "bucket_miou_top1",
        "bucket_miou_at_k",
    )
    summary: dict = {
        "per_seed": per_seed,
        "n_seeds": len(per_seed),
        "n_scored": per_seed[0]["n_scored"] if per_seed else 0,
    }
    for m in metrics:
        values = [s[m] for s in per_seed]
        if not values:
            summary[m + "_mean"] = 0.0
            summary[m + "_std"] = 0.0
            continue
        summary[m + "_mean"] = statistics.fmean(values)
        summary[m + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("inputs", nargs="+", type=Path, help="eval_lookback JSON files")
    ap.add_argument(
        "--label-from-features", action="store_true",
        help="label each session by its features.h5 parent directory",
    )
    ap.add_argument(
        "--by-seed", action="store_true",
        help=(
            "treat inputs as a multi-seed sweep (5 seeds × N sessions = 5N files). "
            "Group by session, compute mean ± std across seeds. "
            "Each input must carry `psm_seed` and the same scored question set "
            "per session."
        ),
    )
    ap.add_argument(
        "--out", type=Path,
        help="write combined JSON record (sessions + combined + by_category)",
    )
    args = ap.parse_args()

    sessions: list[dict] = []
    all_scored: list[dict] = []
    all_negative: list[dict] = []
    common_top: int | None = None
    common_iou_threshold: float | None = None
    common_exemplars: int | None = None
    common_encoder: str | None = None

    by_session_seed: dict[str, list[dict]] = defaultdict(list)

    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
        data = json.loads(path.read_text())
        records = data.get("records", [])
        scored = [r for r in records if r.get("intervals_gt")]
        negative = [r for r in records if not r.get("intervals_gt")]
        label = _session_label(data, args.label_from_features)
        sessions.append({
            "label": label,
            "source_path": str(path),
            "data": data,
            "scored": scored,
            "negative": negative,
        })
        all_scored.extend(scored)
        all_negative.extend(negative)
        by_session_seed[label].append(data)
        if common_top is None:
            common_top = data.get("top")
            common_iou_threshold = data.get("iou_threshold")
            common_exemplars = data.get("exemplars")
            common_encoder = data.get("clip_checkpoint")

    if args.by_seed:
        return _print_seed_summary(
            by_session_seed,
            common_top,
            common_iou_threshold,
            common_exemplars,
            common_encoder,
            args.out,
        )

    # ---- Per-session table ----
    print()
    print("## Aggregate eval — across sessions")
    if common_top is not None:
        print(
            f"_top={common_top}, IoU threshold={common_iou_threshold}, "
            f"exemplars={common_exemplars}, encoder=`{common_encoder}`_"
        )
    print()
    print(
        "| session | n scored | exemplar mIoU @1 | exemplar mIoU @k | "
        "exemplar Hit @k | bucket mIoU @1 |"
    )
    print("|---|---|---|---|---|---|")
    for s in sessions:
        agg = _agg(s["scored"])
        print(
            f"| `{s['label']}` | {agg['n']} | "
            f"{agg['exemplar_miou_top1']:.3f} | "
            f"{agg['exemplar_miou_at_k']:.3f} | "
            f"{agg['exemplar_hit_rate_at_k']:.1%} | "
            f"{agg['bucket_miou_top1']:.3f} |"
        )
    combined = _agg(all_scored)
    print(
        f"| **combined** | **{combined['n']}** | "
        f"**{combined['exemplar_miou_top1']:.3f}** | "
        f"**{combined['exemplar_miou_at_k']:.3f}** | "
        f"**{combined['exemplar_hit_rate_at_k']:.1%}** | "
        f"**{combined['bucket_miou_top1']:.3f}** |"
    )
    print()
    print(
        f"_combined: {combined['n']} scored question(s), "
        f"{len(all_negative)} negative control(s) across {len(sessions)} session(s)._"
    )

    # ---- Per-category breakdown across all sessions ----
    by_cat = _categorize(all_scored)
    if by_cat and not (len(by_cat) == 1 and "(uncategorized)" in by_cat):
        print()
        print("### Per-category breakdown (all sessions combined)")
        print()
        print(
            "| category | n | exemplar mIoU @1 | exemplar mIoU @k | "
            "exemplar Hit @k |"
        )
        print("|---|---|---|---|---|")
        for cat in sorted(by_cat):
            agg = _agg(by_cat[cat])
            print(
                f"| `{cat}` | {agg['n']} | "
                f"{agg['exemplar_miou_top1']:.3f} | "
                f"{agg['exemplar_miou_at_k']:.3f} | "
                f"{agg['exemplar_hit_rate_at_k']:.1%} |"
            )

    # ---- Optional JSON record ----
    if args.out:
        out_data = {
            "sessions": [
                {
                    "label": s["label"],
                    "source_path": s["source_path"],
                    "n_scored": len(s["scored"]),
                    "n_negative": len(s["negative"]),
                    "metrics": _agg(s["scored"]),
                }
                for s in sessions
            ],
            "combined": {
                "n_scored": combined["n"],
                "n_negative": len(all_negative),
                "metrics": combined,
            },
            "by_category": {
                cat: _agg(records) for cat, records in by_cat.items()
            },
            "shared_settings": {
                "top": common_top,
                "iou_threshold": common_iou_threshold,
                "exemplars": common_exemplars,
                "clip_checkpoint": common_encoder,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_data, indent=2))
        print(f"\n[aggregate] wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
