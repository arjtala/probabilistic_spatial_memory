#!/usr/bin/env python3
"""Sliding-window CLIP retrieval baseline (E11).

Slides W-second windows across the session, mean-pools the per-frame CLIP
embeddings inside each window, ranks windows by cosine against the query
embedding, returns the top-k windows as `[t_start, t_start + W]`
intervals. Scores with the same IoU + Hit@k harness as `eval_lookback.py`.

Supports sweeping the window size via `--window-sizes "3,5,10"`, writing
one JSON file per size (the output path gains a `_w<size>s` suffix).

This is the "no spatial structure, but with temporal smoothing" baseline.
If PSM only marginally beats a sliding-window CLIP baseline at matched
window size, the H3+ring-buffer structure's accuracy contribution is
smaller than its bounded-memory contribution — useful framing either way.

Usage:

    python scripts/eval_sliding_window.py \\
        datasets/1501677363692556/clip_bigg_features.h5 \\
        datasets/1501677363692556/questions.yaml \\
        --top 5 --window-sizes "3,5,10" --stride-frac 0.5 \\
        --out captures/sliding_bigG.json

`--stride-frac 0.5` is 50% overlap (every W/2 seconds).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _eval_common import (
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


def build_windows(
    ts_rel: np.ndarray, emb_unit: np.ndarray, window_sec: float, stride_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool unit embeddings inside each [t, t+window_sec) window.

    Returns (window_starts, window_emb_unit) where window_emb_unit is
    re-unit-normalized so a dot product against a unit query equals
    cosine. Windows containing zero frames are dropped.
    """
    if ts_rel.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0, emb_unit.shape[1]), dtype=np.float32)

    t_end = float(ts_rel[-1])
    starts = np.arange(0.0, t_end - 1e-6, stride_sec, dtype=np.float64)
    if starts.size == 0:
        starts = np.array([0.0], dtype=np.float64)

    pooled_list: list[np.ndarray] = []
    kept_starts: list[float] = []
    for s in starts:
        e = s + window_sec
        # Inclusive on start, exclusive on end to avoid double-counting on
        # boundary frames when windows are non-overlapping.
        mask = (ts_rel >= s) & (ts_rel < e)
        n = int(mask.sum())
        if n == 0:
            continue
        mean = emb_unit[mask].mean(axis=0)
        kept_starts.append(float(s))
        pooled_list.append(mean.astype(np.float32))

    if not pooled_list:
        return np.empty((0,), dtype=np.float64), np.empty((0, emb_unit.shape[1]), dtype=np.float32)

    pooled = np.stack(pooled_list, axis=0)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    pooled_unit = (pooled / safe).astype(np.float32)
    return np.array(kept_starts, dtype=np.float64), pooled_unit


def topk_windows(
    qvec: np.ndarray,
    starts: np.ndarray,
    win_emb_unit: np.ndarray,
    window_sec: float,
    *,
    top: int,
) -> list[tuple[float, float, float]]:
    """Rank windows by cosine and return top-k as `[start, start+window]`."""
    if win_emb_unit.shape[0] == 0:
        return []
    sims = win_emb_unit @ qvec
    k = min(top, sims.shape[0])
    idx_unsorted = np.argpartition(-sims, k - 1)[:k]
    idx = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    out = []
    for i in idx:
        s = float(starts[i])
        out.append((s, s + window_sec, float(sims[i])))
    return out


def run_one_window_size(
    *,
    window_sec: float,
    stride_sec: float,
    emb_unit: np.ndarray,
    ts_rel: np.ndarray,
    questions: list[dict],
    runner,
    top: int,
    iou_threshold: float,
    exemplar_tolerance: float,
    verbose: bool,
) -> list[dict]:
    starts, win_emb_unit = build_windows(
        ts_rel, emb_unit, window_sec, stride_sec
    )
    print(
        f"[sliding_{window_sec:g}s] built {win_emb_unit.shape[0]} windows "
        f"(stride={stride_sec:g}s)",
        file=sys.stderr,
    )

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
            if verbose:
                print(f"[sliding_{window_sec:g}s] skip {qid}: {skip}",
                      file=sys.stderr)
            preds = []
        else:
            qvec = embed_query_text(runner, text)
            tops = topk_windows(
                qvec, starts, win_emb_unit, window_sec, top=top
            )
            preds = score_predictions(
                tops, gts_rel,
                exemplar_tolerance=exemplar_tolerance,
            )

        records.append(summarize_question(
            qid, text or "(skipped)", category, notes, gts_rel, preds,
            iou_threshold=iou_threshold,
        ))
    return records


def out_path_with_window(base: Path | None, window_sec: float) -> Path | None:
    if base is None:
        return None
    suffix = f"_w{int(window_sec) if window_sec == int(window_sec) else window_sec}s"
    return base.with_name(f"{base.stem}{suffix}{base.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path,
                    help="HDF5 features file (must contain --group)")
    ap.add_argument("questions", type=Path,
                    help="YAML/JSON question file")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--window-sizes", default="3,5,10",
        help="comma-separated window sizes in seconds; one run per size",
    )
    ap.add_argument(
        "--stride-frac", type=float, default=0.5,
        help="stride as a fraction of window size (0.5 = 50%% overlap)",
    )
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    )
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument("--iou-threshold", type=float, default=None)
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--out", type=Path,
                    help="base path; '_w<size>s' is appended per window size")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    spec = load_questions(args.questions)
    questions = spec.get("questions") or []
    if not questions:
        raise SystemExit(f"no questions in {args.questions}")

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = float(spec.get("iou_threshold", 0.3))

    try:
        window_sizes = [float(w.strip()) for w in args.window_sizes.split(",") if w.strip()]
    except ValueError as exc:
        raise SystemExit(f"bad --window-sizes {args.window_sizes!r}: {exc}")
    if not window_sizes:
        raise SystemExit("need at least one --window-sizes entry")
    for w in window_sizes:
        if w <= 0:
            raise SystemExit(f"window size must be > 0, got {w}")

    emb_unit, ts_rel, session_start = load_features(args.features, args.group)
    print(
        f"[sliding] loaded {emb_unit.shape[0]} frames × {emb_unit.shape[1]} "
        f"dim from {args.features.name}::{args.group}",
        file=sys.stderr,
    )

    runner = make_clip_runner(args.clip_checkpoint, args.clip_device)
    runner_backend = getattr(runner, "backend", "unknown")

    print()
    print(f"## sliding_window_clip on {args.features.name}::{args.group}")
    print(f"_top={args.top}, encoder=`{args.clip_checkpoint}`, "
          f"stride={args.stride_frac:g}× window, "
          f"IoU threshold={iou_threshold}, "
          f"exemplar tol=±{args.exemplar_tolerance:.1f}s_")
    print()
    print("| window | exemplar mIoU @1 | exemplar mIoU @k | exemplar Hit @k | bucket mIoU @k |")
    print("|---|---|---|---|---|")

    for window_sec in window_sizes:
        stride_sec = window_sec * args.stride_frac
        records = run_one_window_size(
            window_sec=window_sec,
            stride_sec=stride_sec,
            emb_unit=emb_unit,
            ts_rel=ts_rel,
            questions=questions,
            runner=runner,
            top=args.top,
            iou_threshold=iou_threshold,
            exemplar_tolerance=args.exemplar_tolerance,
            verbose=args.verbose,
        )

        scored = [r for r in records if r["intervals_gt"]]
        n = max(len(scored), 1)
        miou1 = sum(r["exemplar_iou_top1"] for r in scored) / n
        mioukk = sum(r["exemplar_iou_at_k"] for r in scored) / n
        hit = sum(1 for r in scored if r["exemplar_hit_at_k"]) / n
        bucketk = sum(r["bucket_iou_at_k"] for r in scored) / n
        print(f"| {window_sec:g}s | {miou1:.3f} | {mioukk:.3f} | "
              f"{hit:.1%} ({sum(1 for r in scored if r['exemplar_hit_at_k'])}/{len(scored)}) | "
              f"{bucketk:.3f} |")

        out = out_path_with_window(args.out, window_sec)
        if out is not None:
            write_eval_json(
                out,
                features_h5=args.features,
                questions_file=args.questions,
                group=args.group,
                top=args.top,
                records=records,
                session_start=session_start,
                clip_checkpoint=args.clip_checkpoint,
                clip_backend=runner_backend,
                iou_threshold=iou_threshold,
                exemplar_tolerance=args.exemplar_tolerance,
                baseline_method=f"sliding_window_{window_sec:g}s",
                extra_settings={
                    "window_sec": window_sec,
                    "stride_sec": stride_sec,
                },
            )

    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
