#!/usr/bin/env python3
"""Brute-force CLIP retrieval baseline (E11).

For each question, embed the text via CLIP, rank every frame in the
session's `features.h5::<group>/embeddings` by cosine, return the top-k
frames as `[frame_t ± exemplar_tolerance]` intervals. Score with the
same IoU + Hit@k harness as `eval_lookback.py`.

This is the "no spatial structure, no time bucketing, no reservoir"
baseline. If PSM doesn't beat this, the H3+ring-buffer machinery isn't
doing accuracy work and the paper has to lead with bounded-memory
instead.

Usage:

    python scripts/eval_brute_force_clip.py \\
        datasets/1501677363692556/clip_bigg_features.h5 \\
        datasets/1501677363692556/questions.yaml \\
        --top 5 --out captures/brute_force_bigG.json

Output JSON is schema-compatible with `eval_lookback.py --out` so
`scripts/eval_aggregate.py` can pool brute-force runs alongside PSM
runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _eval_common import (
    REPO_ROOT,
    auto_session_start,
    embed_query_text,
    load_features,
    load_questions,
    make_clip_runner,
    question_text_and_skip_reason,
    score_predictions,
    summarize_question,
    write_eval_json,
)


def topk_brute_force(
    qvec: np.ndarray,
    emb_unit: np.ndarray,
    ts_rel: np.ndarray,
    *,
    top: int,
    exemplar_tolerance: float,
) -> list[tuple[float, float, float]]:
    """Return top-k `(t_min, t_max, similarity)` intervals from full-bank ranking.

    `qvec` is the unit-normalized query vector. `emb_unit` is the (N, D)
    unit-normalized frame embedding bank. The cosine is a single dot
    product; argpartition pulls the top-k indices in O(N) instead of
    sorting all N. Each predicted interval is `[frame_t ± tol]` — same
    semantics as PSM's `exemplar_t ± tol` so the scorer agrees.
    """
    sims = emb_unit @ qvec  # (N,)
    k = min(top, sims.shape[0])
    if k <= 0:
        return []
    # argpartition gives unsorted top-k; sort them by similarity desc.
    idx_unsorted = np.argpartition(-sims, k - 1)[:k]
    idx = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    out = []
    for i in idx:
        t = float(ts_rel[i])
        sim = float(sims[i])
        out.append((t - exemplar_tolerance, t + exemplar_tolerance, sim))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path,
                    help="HDF5 features file (must contain --group)")
    ap.add_argument("questions", type=Path,
                    help="YAML/JSON question file")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        help="image-text encoder for the question (default: bigG to match v2)",
    )
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument(
        "--iou-threshold", type=float, default=None,
        help="override Hit @k threshold (default: from question file or 0.3)",
    )
    ap.add_argument(
        "--exemplar-tolerance", type=float, default=1.5,
        help="seconds ± frame_t treated as the predicted interval",
    )
    ap.add_argument("--out", type=Path,
                    help="write detailed JSON record here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    spec = load_questions(args.questions)
    questions = spec.get("questions") or []
    if not questions:
        raise SystemExit(f"no questions in {args.questions}")

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = float(spec.get("iou_threshold", 0.3))

    emb_unit, ts_rel, session_start = load_features(args.features, args.group)
    print(
        f"[brute_force] loaded {emb_unit.shape[0]} frames × {emb_unit.shape[1]} "
        f"dim from {args.features.name}::{args.group}",
        file=sys.stderr,
    )

    runner = make_clip_runner(args.clip_checkpoint, args.clip_device)

    records: list[dict] = []
    for q in questions:
        qid = q.get("id") or f"q{len(records) + 1}"
        text, skip = question_text_and_skip_reason(q)
        gts_rel: list[tuple[float, float]] = [
            (float(iv[0]), float(iv[1])) for iv in q.get("intervals", [])
        ]
        category = q.get("category") or "(uncategorized)"
        notes = q.get("notes", "")

        if text is None:
            if args.verbose:
                print(f"[brute_force] skip {qid}: {skip}", file=sys.stderr)
            preds = []
        else:
            qvec = embed_query_text(runner, text)
            top_intervals = topk_brute_force(
                qvec, emb_unit, ts_rel,
                top=args.top,
                exemplar_tolerance=args.exemplar_tolerance,
            )
            preds = score_predictions(
                top_intervals, gts_rel,
                exemplar_tolerance=args.exemplar_tolerance,
            )

        records.append(summarize_question(
            qid, text or "(skipped)", category, notes, gts_rel, preds,
            iou_threshold=iou_threshold,
        ))

    runner.close()

    # ---- Print a compact summary to stdout (mirrors eval_lookback.py) ----
    scored = [r for r in records if r["intervals_gt"]]
    n = max(len(scored), 1)
    miou1 = sum(r["exemplar_iou_top1"] for r in scored) / n
    mioukk = sum(r["exemplar_iou_at_k"] for r in scored) / n
    hit = sum(1 for r in scored if r["exemplar_hit_at_k"]) / n
    bucket1 = sum(r["bucket_iou_top1"] for r in scored) / n
    bucketk = sum(r["bucket_iou_at_k"] for r in scored) / n

    print()
    print(f"## brute_force_clip on {args.features.name}::{args.group}")
    print(f"_top={args.top}, encoder=`{args.clip_checkpoint}`, "
          f"IoU threshold={iou_threshold}, "
          f"exemplar tol=±{args.exemplar_tolerance:.1f}s_")
    print()
    print("| metric | value |")
    print("|---|---|")
    print(f"| exemplar mIoU @1 | {miou1:.3f} |")
    print(f"| exemplar mIoU @{args.top} | {mioukk:.3f} |")
    print(f"| exemplar Hit @{args.top} | {hit:.1%} ({sum(1 for r in scored if r['exemplar_hit_at_k'])}/{len(scored)}) |")
    print(f"| bucket mIoU @1 | {bucket1:.3f} |")
    print(f"| bucket mIoU @{args.top} | {bucketk:.3f} |")

    if args.out:
        write_eval_json(
            args.out,
            features_h5=args.features,
            questions_file=args.questions,
            group=args.group,
            top=args.top,
            records=records,
            session_start=session_start,
            clip_checkpoint=args.clip_checkpoint,
            clip_backend=runner.backend if hasattr(runner, "backend") else "unknown",
            iou_threshold=iou_threshold,
            exemplar_tolerance=args.exemplar_tolerance,
            baseline_method="brute_force_clip",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
