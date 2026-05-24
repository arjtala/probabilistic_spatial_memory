#!/usr/bin/env python3
"""Uniform-sample CLIP retrieval baseline (E11).

For each rate R (seconds-per-sample), pick exactly one frame per R-second
bin (the frame closest to bin center), embed the question, rank the
sampled frames by cosine, return the top-k as `[frame_t ± tol]`
intervals. Scores with the same harness as `eval_lookback.py`.

This is the trivial-floor baseline: no temporal smoothing, no spatial
structure, just throw away most of the data and rank what's left. If
PSM doesn't beat this, the whole project is in trouble. Supports
sweeping the sample rate via `--rates "30,75"`.

Usage:

    python scripts/eval_uniform_sample.py \\
        datasets/1501677363692556/clip_bigg_features.h5 \\
        datasets/1501677363692556/questions.yaml \\
        --top 5 --rates "30,75" \\
        --out captures/uniform_bigG.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _eval_common import (
    embed_query_text,
    load_features,
    load_questions,
    make_clip_runner,
    question_text_and_skip_reason,
    score_predictions,
    summarize_question,
    write_eval_json,
)


def pick_one_per_bin(
    ts_rel: np.ndarray, emb_unit: np.ndarray, rate_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the frame closest to each bin's center.

    Bins are `[0, rate_sec)`, `[rate_sec, 2*rate_sec)`, etc. Empty bins
    are dropped. Returns (sampled_ts, sampled_emb_unit).
    """
    if ts_rel.size == 0:
        return ts_rel.copy(), emb_unit.copy()
    t_end = float(ts_rel[-1])
    n_bins = int(np.ceil((t_end + 1e-9) / rate_sec))
    sampled_ts: list[float] = []
    sampled_idx: list[int] = []
    for b in range(n_bins):
        lo = b * rate_sec
        hi = lo + rate_sec
        center = lo + rate_sec / 2.0
        mask = (ts_rel >= lo) & (ts_rel < hi)
        if not mask.any():
            continue
        candidate_idx = np.where(mask)[0]
        dists = np.abs(ts_rel[candidate_idx] - center)
        best = candidate_idx[int(np.argmin(dists))]
        sampled_ts.append(float(ts_rel[best]))
        sampled_idx.append(int(best))
    if not sampled_idx:
        return np.array([], dtype=np.float64), np.empty((0, emb_unit.shape[1]), dtype=np.float32)
    idx_arr = np.array(sampled_idx, dtype=np.int64)
    return np.array(sampled_ts, dtype=np.float64), emb_unit[idx_arr].copy()


def topk_uniform(
    qvec: np.ndarray,
    ts: np.ndarray,
    emb_unit: np.ndarray,
    *,
    top: int,
    exemplar_tolerance: float,
) -> list[tuple[float, float, float]]:
    if emb_unit.shape[0] == 0:
        return []
    sims = emb_unit @ qvec
    k = min(top, sims.shape[0])
    idx_unsorted = np.argpartition(-sims, k - 1)[:k]
    idx = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    out = []
    for i in idx:
        t = float(ts[i])
        out.append((t - exemplar_tolerance, t + exemplar_tolerance, float(sims[i])))
    return out


def run_one_rate(
    *,
    rate_sec: float,
    emb_unit: np.ndarray,
    ts_rel: np.ndarray,
    questions: list[dict],
    runner,
    top: int,
    iou_threshold: float,
    exemplar_tolerance: float,
    verbose: bool,
) -> list[dict]:
    sampled_ts, sampled_emb = pick_one_per_bin(ts_rel, emb_unit, rate_sec)
    print(
        f"[uniform_{rate_sec:g}s] sampled {sampled_emb.shape[0]} frames "
        f"(one per {rate_sec:g}s)",
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
                print(f"[uniform_{rate_sec:g}s] skip {qid}: {skip}",
                      file=sys.stderr)
            preds = []
        else:
            qvec = embed_query_text(runner, text)
            tops = topk_uniform(
                qvec, sampled_ts, sampled_emb,
                top=top, exemplar_tolerance=exemplar_tolerance,
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


def out_path_with_rate(base: Path | None, rate_sec: float) -> Path | None:
    if base is None:
        return None
    suffix = f"_r{int(rate_sec) if rate_sec == int(rate_sec) else rate_sec}s"
    return base.with_name(f"{base.stem}{suffix}{base.suffix}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--rates", default="30,75",
        help="comma-separated seconds-per-sample rates; one run per rate",
    )
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    )
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument("--iou-threshold", type=float, default=None)
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--out", type=Path,
                    help="base path; '_r<rate>s' is appended per rate")
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
        rates = [float(r.strip()) for r in args.rates.split(",") if r.strip()]
    except ValueError as exc:
        raise SystemExit(f"bad --rates {args.rates!r}: {exc}")
    if not rates:
        raise SystemExit("need at least one --rates entry")
    for r in rates:
        if r <= 0:
            raise SystemExit(f"rate must be > 0, got {r}")

    emb_unit, ts_rel, session_start = load_features(args.features, args.group)
    print(
        f"[uniform] loaded {emb_unit.shape[0]} frames × {emb_unit.shape[1]} "
        f"dim from {args.features.name}::{args.group}",
        file=sys.stderr,
    )

    runner = make_clip_runner(args.clip_checkpoint, args.clip_device)
    runner_backend = getattr(runner, "backend", "unknown")

    print()
    print(f"## uniform_sample_clip on {args.features.name}::{args.group}")
    print(f"_top={args.top}, encoder=`{args.clip_checkpoint}`, "
          f"IoU threshold={iou_threshold}, "
          f"exemplar tol=±{args.exemplar_tolerance:.1f}s_")
    print()
    print("| rate | n_sampled | exemplar mIoU @1 | exemplar mIoU @k | exemplar Hit @k |")
    print("|---|---|---|---|---|")

    for rate_sec in rates:
        records = run_one_rate(
            rate_sec=rate_sec,
            emb_unit=emb_unit,
            ts_rel=ts_rel,
            questions=questions,
            runner=runner,
            top=args.top,
            iou_threshold=iou_threshold,
            exemplar_tolerance=args.exemplar_tolerance,
            verbose=args.verbose,
        )

        # Recompute n_sampled (a function of rate, not of question loop)
        n_sampled, _ = pick_one_per_bin(ts_rel, emb_unit, rate_sec)
        scored = [r for r in records if r["intervals_gt"]]
        n = max(len(scored), 1)
        miou1 = sum(r["exemplar_iou_top1"] for r in scored) / n
        mioukk = sum(r["exemplar_iou_at_k"] for r in scored) / n
        hit = sum(1 for r in scored if r["exemplar_hit_at_k"]) / n
        print(f"| {rate_sec:g}s | {n_sampled.shape[0]} | {miou1:.3f} | "
              f"{mioukk:.3f} | {hit:.1%} ({sum(1 for r in scored if r['exemplar_hit_at_k'])}/{len(scored)}) |")

        out = out_path_with_rate(args.out, rate_sec)
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
                baseline_method=f"uniform_sample_{rate_sec:g}s",
                extra_settings={"sample_rate_sec": rate_sec},
            )

    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
