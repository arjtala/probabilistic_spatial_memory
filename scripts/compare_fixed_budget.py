#!/usr/bin/env python3
"""Paired session-level comparison for fixed-budget evaluation captures.

For each method, metrics are averaged over seeds within a session. The script
then reports the paired method-A minus method-B difference and a bootstrap
confidence interval obtained by resampling sessions, never questions or seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = (
    "hit_at_k",
    "miou_at_k",
    "oracle_retained",
    "rare_place_hit",
    "common_place_hit",
    "old_event_hit",
    "recent_event_hit",
    "revisited_place_hit",
)


@dataclass
class Capture:
    path: Path
    method: str
    session: str
    seed: int
    transform: str
    features_sha256: str | None
    questions_sha256: str | None
    config_fingerprint: str
    metrics: dict[str, float | None]


def transform_tag(data: dict[str, Any]) -> str:
    value = data.get("spatial_transform")
    if isinstance(value, dict):
        return str(value.get("tag") or "none")
    return str(value or "none")


def session_id(data: dict[str, Any], path: Path) -> str:
    explicit = data.get("session_id")
    if explicit:
        return str(explicit)
    features = data.get("features")
    if features:
        return Path(str(features)).parent.name
    return path.parent.name


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def record_flag(record: dict[str, Any], key: str) -> bool | None:
    value = record.get(key)
    return value if isinstance(value, bool) else None


def in_stratum(record: dict[str, Any], name: str) -> bool:
    strata = record.get("strata")
    return isinstance(strata, dict) and strata.get(name) is True


def capture_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    scored = [record for record in records if record.get("intervals_gt")]
    hits = [
        float(value)
        for record in scored
        if (value := record_flag(record, "exemplar_hit_at_k")) is not None
    ]
    mious = [
        float(record["exemplar_iou_at_k"])
        for record in scored
        if record.get("exemplar_iou_at_k") is not None
    ]
    oracle = [
        float(value)
        for record in scored
        if (value := record_flag(record, "oracle_retained_hit")) is not None
    ]
    strata_values: dict[str, list[float]] = {}
    for stratum in (
        "rare_place",
        "common_place",
        "old_event",
        "recent_event",
        "revisited_place",
    ):
        strata_values[stratum] = [
            float(value)
            for record in scored
            if in_stratum(record, stratum)
            and (value := record_flag(record, "exemplar_hit_at_k")) is not None
        ]
    return {
        "hit_at_k": mean_or_none(hits),
        "miou_at_k": mean_or_none(mious),
        "oracle_retained": mean_or_none(oracle),
        "rare_place_hit": mean_or_none(strata_values["rare_place"]),
        "common_place_hit": mean_or_none(strata_values["common_place"]),
        "old_event_hit": mean_or_none(strata_values["old_event"]),
        "recent_event_hit": mean_or_none(strata_values["recent_event"]),
        "revisited_place_hit": mean_or_none(strata_values["revisited_place"]),
    }


def load_capture(path: Path) -> Capture:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError("not an evaluation capture")
    method = data.get("retention_method") or data.get("baseline_method")
    if not method:
        raise ValueError("missing retention_method")
    seed = data.get("psm_seed")
    if not isinstance(seed, int):
        raise ValueError("missing integer psm_seed")
    return Capture(
        path=path,
        method=str(method),
        session=session_id(data, path),
        seed=seed,
        transform=transform_tag(data),
        features_sha256=data.get("features_sha256"),
        questions_sha256=data.get("questions_sha256"),
        config_fingerprint=json.dumps(
            {
                key: data.get(key)
                for key in (
                    "group",
                    "top",
                    "clip_checkpoint",
                    "clip_backend",
                    "iou_threshold",
                    "exemplar_tolerance_sec",
                    "visit_gap_sec",
                    "hll_capacity",
                    "hll_precision",
                    "evaluation_code_sha256",
                    "software_versions",
                    "model_revision",
                    "feature_checkpoint",
                )
            },
            sort_keys=True,
        ),
        metrics=capture_metrics(data["records"]),
    )


def expand_inputs(paths: list[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*.json") if recursive else path.glob("*.json")
            found.extend(
                candidate
                for candidate in sorted(iterator)
                if candidate.is_file() and not candidate.name.startswith("manifest_")
            )
        else:
            found.append(path)
    return list(dict.fromkeys(path.resolve() for path in found))


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = probability * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def paired_bootstrap(
    differences: list[float], resamples: int, ci: float, seed: int
) -> tuple[float, float, float]:
    mean = statistics.fmean(differences)
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(resamples)
    )
    alpha = (1.0 - ci) / 2.0
    return mean, percentile(draws, alpha), percentile(draws, 1.0 - alpha)


def fmt(value: float, percent: bool) -> str:
    return f"{value:.1%}" if percent else f"{value:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--method-a", default="spatial_priority")
    ap.add_argument("--method-b", default="global_reservoir")
    ap.add_argument("--budget", type=int, default=128)
    ap.add_argument("--h3-resolution", type=int, default=12)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--transform-a", default="base")
    ap.add_argument("--transform-b", default="base")
    ap.add_argument("--expected-seeds", default="0,1,2,3,4")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42,
                    help="bootstrap RNG seed")
    ap.add_argument("--min-sessions", type=int, default=2)
    ap.add_argument("--allow-unpaired", action="store_true")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.budget <= 0 or args.top <= 0 or not 0 <= args.h3_resolution <= 15:
        ap.error("budget/top must be positive and H3 resolution in [0, 15]")
    if args.method_a == args.method_b and args.transform_a == args.transform_b:
        ap.error("A and B must differ by method or transform")
    if args.resamples <= 0 or args.min_sessions <= 0 or not 0.0 < args.ci < 1.0:
        ap.error("resamples/min-sessions must be positive and CI in (0, 1)")
    try:
        expected_seeds = {
            int(value.strip())
            for value in args.expected_seeds.split(",")
            if value.strip()
        }
    except ValueError as exc:
        ap.error(f"bad --expected-seeds: {exc}")

    paths = expand_inputs(args.inputs, args.recursive)
    selected: list[Capture] = []
    skipped = 0
    for path in paths:
        try:
            data = json.loads(path.read_text())
            if data.get("exemplar_budget") != args.budget:
                continue
            if data.get("h3_resolution") != args.h3_resolution:
                continue
            if data.get("top") != args.top:
                continue
            capture = load_capture(path)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            skipped += 1
            continue
        wanted = (
            capture.method == args.method_a and capture.transform == args.transform_a
        ) or (
            capture.method == args.method_b and capture.transform == args.transform_b
        )
        if wanted:
            selected.append(capture)
    if not selected:
        raise SystemExit("no captures match the requested comparison")

    by_method_session: dict[tuple[str, str, str], list[Capture]] = defaultdict(list)
    for capture in selected:
        by_method_session[(capture.method, capture.transform, capture.session)].append(capture)

    session_metrics: dict[tuple[str, str, str], dict[str, float | None]] = {}
    session_hashes: dict[tuple[str, str, str], tuple[str | None, str | None]] = {}
    session_configs: dict[tuple[str, str, str], str] = {}
    for key, captures in by_method_session.items():
        seeds = {capture.seed for capture in captures}
        if seeds != expected_seeds:
            raise SystemExit(
                f"{key[0]}::{key[1]} has seeds {sorted(seeds)}, "
                f"expected {sorted(expected_seeds)}"
            )
        for seed in seeds:
            if sum(capture.seed == seed for capture in captures) != 1:
                raise SystemExit(f"duplicate capture for {key} seed={seed}")
        session_metrics[key] = {
            metric: mean_or_none(
                [
                    value
                    for capture in captures
                    if (value := capture.metrics[metric]) is not None
                ]
            )
            for metric in METRICS
        }
        hashes = {
            (capture.features_sha256, capture.questions_sha256)
            for capture in captures
        }
        if len(hashes) != 1:
            raise SystemExit(f"input hashes differ across seeds for {key}")
        only_hashes = next(iter(hashes))
        if any(value is None for value in only_hashes):
            raise SystemExit(f"input hashes are missing for {key}")
        session_hashes[key] = only_hashes
        configs = {capture.config_fingerprint for capture in captures}
        if len(configs) != 1:
            raise SystemExit(f"evaluation settings differ across seeds for {key}")
        session_configs[key] = next(iter(configs))

    sessions_a = {
        session
        for method, transform, session in session_metrics
        if method == args.method_a and transform == args.transform_a
    }
    sessions_b = {
        session
        for method, transform, session in session_metrics
        if method == args.method_b and transform == args.transform_b
    }
    common = sorted(sessions_a & sessions_b)
    if not common:
        raise SystemExit("methods have no paired sessions")
    if sessions_a != sessions_b and not args.allow_unpaired:
        raise SystemExit(
            "method session sets differ; rerun the missing captures or pass "
            "--allow-unpaired explicitly"
        )
    if len(common) < args.min_sessions:
        raise SystemExit(
            f"only {len(common)} paired session(s), fewer than "
            f"--min-sessions={args.min_sessions}"
        )
    for session in common:
        key_a = (args.method_a, args.transform_a, session)
        key_b = (args.method_b, args.transform_b, session)
        if session_hashes[key_a] != session_hashes[key_b]:
            raise SystemExit(f"paired input hashes differ for session {session}")
        if session_configs[key_a] != session_configs[key_b]:
            raise SystemExit(f"paired evaluation settings differ for session {session}")
    common_configs = {
        session_configs[(args.method_a, args.transform_a, session)]
        for session in common
    }
    if len(common_configs) != 1:
        raise SystemExit("evaluation settings differ across paired sessions")

    lines = [
        "# Paired Fixed-Budget Comparison",
        "",
        f"A: `{args.method_a}` (`{args.transform_a}`); "
        f"B: `{args.method_b}` (`{args.transform_b}`); "
        f"M={args.budget}, H3 r={args.h3_resolution}, K={args.top}. Seeds are averaged "
        "within session before paired session bootstrap.",
        "",
        f"| metric | sessions | A | B | A - B (paired {args.ci:.0%} CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        pairs = [
            (
                session_metrics[(args.method_a, args.transform_a, session)][metric],
                session_metrics[(args.method_b, args.transform_b, session)][metric],
            )
            for session in common
        ]
        valid = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not valid:
            continue
        a_values = [float(a) for a, _ in valid]
        b_values = [float(b) for _, b in valid]
        differences = [a - b for a, b in zip(a_values, b_values)]
        percent = metric != "miou_at_k"
        if len(valid) < args.min_sessions:
            lines.append(
                f"| {metric} | {len(valid)} | "
                f"{fmt(statistics.fmean(a_values), percent)} | "
                f"{fmt(statistics.fmean(b_values), percent)} | "
                "insufficient paired sessions |"
            )
            continue
        delta, low, high = paired_bootstrap(
            differences, args.resamples, args.ci, args.seed
        )
        lines.append(
            f"| {metric} | {len(valid)} | {fmt(statistics.fmean(a_values), percent)} "
            f"| {fmt(statistics.fmean(b_values), percent)} | "
            f"{fmt(delta, percent)} [{fmt(low, percent)}, {fmt(high, percent)}] |"
        )
    if sessions_a != sessions_b:
        lines.extend(
            [
                "",
                f"Warning: unpaired sessions were excluded. A-only={sorted(sessions_a - sessions_b)}, "
                f"B-only={sorted(sessions_b - sessions_a)}.",
            ]
        )
    if len(common) == 1:
        lines.extend(
            [
                "",
                "Warning: one paired session yields a degenerate bootstrap interval; "
                "do not interpret it as inferential uncertainty.",
            ]
        )
    if skipped:
        lines.extend(["", f"Skipped {skipped} non-evaluation JSON file(s)."])
    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(f"[fixed-budget] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
