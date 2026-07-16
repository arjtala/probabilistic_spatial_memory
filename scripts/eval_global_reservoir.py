#!/usr/bin/env python3
"""Evaluate a globally budgeted streaming reservoir baseline.

The baseline applies Vitter's Algorithm R to the full frame stream, then ranks
only the retained embeddings for each text query. It is deliberately
non-spatial: comparing it with fixed-budget PSM isolates whether allocating a
budget across H3 cells helps beyond uniform retention.

The logical state model charges ``4 * embedding_dim`` payload bytes plus one
float64 timestamp per retained exemplar. Container and allocator overhead are
excluded for both methods; measure RSS separately for implementation costs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from _eval_common import (
    embed_query_text,
    load_features,
    load_questions,
    make_clip_runner,
    score_predictions,
    summarize_question,
    write_eval_json,
)
from eval_brute_force_clip import topk_brute_force


TIMESTAMP_BYTES = 8


def reservoir_indices(n_items: int, capacity: int, seed: int) -> np.ndarray:
    """Return the items retained by streaming Algorithm R.

    The result is sorted only to make captures easier to inspect. Sorting does
    not affect retrieval because ranking is recomputed from cosine similarity.
    """
    if n_items < 0 or capacity < 0:
        raise ValueError("n_items and capacity must be non-negative")
    kept = min(n_items, capacity)
    if kept == 0:
        return np.empty(0, dtype=np.int64)
    if kept == n_items:
        return np.arange(n_items, dtype=np.int64)

    rng = np.random.default_rng(seed)
    reservoir = np.arange(kept, dtype=np.int64)
    for stream_idx in range(kept, n_items):
        slot = int(rng.integers(0, stream_idx + 1))
        if slot < kept:
            reservoir[slot] = stream_idx
    reservoir.sort()
    return reservoir


def capacity_for_budget(budget_bytes: int, embedding_dim: int) -> int:
    if budget_bytes < 0 or embedding_dim <= 0:
        raise ValueError("budget must be non-negative and embedding_dim positive")
    return budget_bytes // (embedding_dim * np.dtype(np.float32).itemsize + TIMESTAMP_BYTES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path)
    ap.add_argument("questions", type=Path)
    budget = ap.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget-mib", type=float,
                        help="logical state budget in MiB")
    budget.add_argument("--capacity", type=int,
                        help="number of frame embeddings retained")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    )
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument("--allow-checkpoint-mismatch", action="store_true")
    ap.add_argument("--iou-threshold", type=float, default=None)
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.budget_mib is not None and args.budget_mib <= 0:
        raise SystemExit("--budget-mib must be positive")
    if args.capacity is not None and args.capacity <= 0:
        raise SystemExit("--capacity must be positive")
    if args.top <= 0:
        raise SystemExit("--top must be positive")

    spec = load_questions(args.questions)
    all_questions = spec.get("questions") or []
    questions = [
        question
        for question in all_questions
        if question.get("query_mode", "similarity_search") == "similarity_search"
        and question.get("query")
        and question.get("intervals")
    ]
    if not questions:
        raise SystemExit(f"no scored similarity-search questions in {args.questions}")
    question_ids = [
        str(question.get("id") or f"q{position + 1}")
        for position, question in enumerate(questions)
    ]
    duplicate_ids = sorted(
        qid for qid in set(question_ids) if question_ids.count(qid) > 1
    )
    if duplicate_ids:
        raise SystemExit(f"duplicate question IDs: {duplicate_ids}")
    if len(questions) != len(all_questions):
        print(
            f"[global_reservoir] excluding {len(all_questions) - len(questions)} "
            "non-text or unscored questions",
            file=sys.stderr,
        )
    iou_threshold = (
        float(args.iou_threshold)
        if args.iou_threshold is not None
        else float(spec.get("iou_threshold", 0.3))
    )

    emb_unit, ts_rel, session_start = load_features(args.features, args.group)
    n_frames, dim = emb_unit.shape
    with h5py.File(args.features, "r") as handle:
        checkpoint_raw = handle[args.group].attrs.get("checkpoint")
    if isinstance(checkpoint_raw, bytes):
        feature_checkpoint = checkpoint_raw.decode("utf-8")
    elif checkpoint_raw is None:
        feature_checkpoint = None
    else:
        feature_checkpoint = str(checkpoint_raw)
    if (
        feature_checkpoint
        and feature_checkpoint != args.clip_checkpoint
        and not args.allow_checkpoint_mismatch
    ):
        raise SystemExit(
            f"query checkpoint {args.clip_checkpoint!r} does not match feature "
            f"checkpoint {feature_checkpoint!r}"
        )
    item_bytes = dim * np.dtype(np.float32).itemsize + TIMESTAMP_BYTES
    if args.budget_mib is not None:
        target_budget_bytes = int(args.budget_mib * 1024 * 1024)
        capacity = capacity_for_budget(target_budget_bytes, dim)
    else:
        capacity = int(args.capacity)
        target_budget_bytes = capacity * item_bytes
    if capacity < args.top:
        raise SystemExit(
            f"budget retains only {capacity} exemplars, fewer than --top={args.top}"
        )

    indices = reservoir_indices(n_frames, capacity, args.seed)
    retained_emb = emb_unit[indices]
    retained_ts = ts_rel[indices]
    retained_count = int(indices.size)
    logical_state_bytes = retained_count * item_bytes
    max_state_bytes = capacity * item_bytes
    if max_state_bytes > target_budget_bytes:
        raise RuntimeError("internal error: reservoir exceeds target budget")

    print(
        f"[global_reservoir] frames={n_frames} dim={dim} capacity={capacity} "
        f"retained={retained_count} state={logical_state_bytes / 2**20:.3f} MiB",
        file=sys.stderr,
    )

    runner = make_clip_runner(args.clip_checkpoint, args.clip_device)
    records: list[dict] = []
    for q in questions:
        qid = q.get("id") or f"q{len(records) + 1}"
        text = str(q["query"])
        gts_rel = [
            (float(interval[0]), float(interval[1]))
            for interval in q.get("intervals", [])
        ]
        qvec = embed_query_text(runner, text)
        intervals = topk_brute_force(
            qvec,
            retained_emb,
            retained_ts,
            top=args.top,
            exemplar_tolerance=args.exemplar_tolerance,
        )
        preds = score_predictions(
            intervals,
            gts_rel,
            exemplar_tolerance=args.exemplar_tolerance,
        )
        records.append(
            summarize_question(
                qid,
                text,
                q.get("category") or "(uncategorized)",
                q.get("notes", ""),
                gts_rel,
                preds,
                iou_threshold=iou_threshold,
            )
        )
        records[-1]["oracle_retained_hit"] = any(
            bool(np.any((retained_ts >= start) & (retained_ts <= end)))
            for start, end in gts_rel
        )

    backend = getattr(runner, "backend", "unknown")
    close = getattr(runner, "close", None)
    if close is not None:
        close()

    write_eval_json(
        args.out,
        features_h5=args.features,
        questions_file=args.questions,
        group=args.group,
        top=args.top,
        records=records,
        session_start=session_start,
        clip_checkpoint=args.clip_checkpoint,
        clip_backend=backend,
        iou_threshold=iou_threshold,
        exemplar_tolerance=args.exemplar_tolerance,
        baseline_method="global_reservoir",
        seed=args.seed,
        extra_settings={
            "reservoir_capacity": capacity,
            "retained_exemplars": retained_count,
            "retained_fraction": (
                float(retained_count / n_frames) if n_frames else 0.0
            ),
            "record_count": n_frames,
            "embedding_dim": dim,
            "feature_checkpoint": feature_checkpoint,
            "fixed_budget": {
                "target_bytes": target_budget_bytes,
                "max_logical_state_bytes": max_state_bytes,
                "logical_state_bytes": logical_state_bytes,
                "item_bytes": item_bytes,
                "timestamp_bytes_per_item": TIMESTAMP_BYTES,
                "accounting": "raw float32 payload + float64 timestamp; container overhead excluded",
            },
        },
    )

    summary = [r for r in records if r["intervals_gt"]]
    hits = sum(bool(r["exemplar_hit_at_k"]) for r in summary)
    print(
        f"global reservoir: Hit@{args.top}={hits / max(len(summary), 1):.1%} "
        f"({hits}/{len(summary)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
