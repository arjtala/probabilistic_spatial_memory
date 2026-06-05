#!/usr/bin/env python3
"""PSM + temporal-pooling rerank.

Hypothesis: sliding-window CLIP @ 10s beats per-frame brute-force CLIP
on Hit@5 because window-mean embeddings smooth out per-frame variance.
If true, PSM's substrate can claim the same smoothing benefit by
re-scoring its top-K candidates against a mean-pooled embedding
window centered on each candidate's exemplar_t -- without changing
the substrate's storage layout or bounded-memory operating point.

Pipeline per query:
  1. PSM --search returns top-K candidates (exemplar_t per candidate).
  2. For each candidate, pull the frame embeddings from features.h5
     whose timestamps are within [exemplar_t - W/2, exemplar_t + W/2]
     (W = --pool-window seconds).
  3. Mean-pool those embeddings -> one pooled vector per candidate.
  4. Re-score (query, pooled_vector) cosines and re-order top-K.
  5. Emit a per-question record in the same shape as eval_lookback.py
     so the aggregator pools the result alongside everything else.

The pooling is read from the existing features.h5 embedding bank,
so this script does NOT need a PSM C-side change and does NOT
need re-extraction. The bank is present anyway because PSM ingest
read from it -- the bounded-memory deployment claim doesn't need
the bank at all once PSM is built. We rely on the bank here
purely as the rerank-time evidence; treat it as the same
infrastructure brute-force CLIP uses.

W=0 is the no-pool baseline and reproduces PSM-only Hit@5.
W=10 should approach sliding-window @ 10s's 15.0% Hit@5 ceiling.

Usage:
  python scripts/eval_psm_temporal_pool.py \\
    /checkpoint/.../<sid>/clip_l_features.h5 \\
    /checkpoint/.../<sid>/questions.yaml \\
    --pool-window 10 --per-cell-cap 5 \\
    --out captures/eval_<sid>_pool10.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "extraction"))

from scripts.eval_lookback import (  # noqa: E402
    auto_session_start,
    best_iou,
    point_in_any,
    run_psm_search,
)


def _load_bank(features_h5: Path, group: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return L2-normalized embeddings + session-relative timestamps + session_start."""
    with h5py.File(features_h5, "r") as f:
        emb = f[f"{group}/embeddings"][:].astype(np.float32)
        ts = f[f"{group}/timestamps"][:].astype(np.float64)
    session_start = float(ts[0])
    ts_rel = (ts - session_start).astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_unit = (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32)
    return emb_unit, ts_rel, session_start


def _pool_around(
    emb_bank: np.ndarray,
    ts_bank: np.ndarray,
    center_t: float,
    half_window_s: float,
) -> np.ndarray | None:
    """Mean of embeddings whose timestamp falls in [center - W/2, center + W/2].

    Returns None if no frames fall in the window (caller should fall
    back to the single nearest frame).
    """
    lo = center_t - half_window_s
    hi = center_t + half_window_s
    mask = (ts_bank >= lo) & (ts_bank <= hi)
    if not mask.any():
        return None
    pooled = emb_bank[mask].mean(axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--per-cell-cap", type=int, default=5)
    ap.add_argument("--time-window", type=float, default=30.0)
    ap.add_argument("--capacity", type=int, default=60)
    ap.add_argument("--h3-resolution", type=int, default=12)
    ap.add_argument("--precision", type=int, default=14)
    ap.add_argument("--exemplars", type=int, default=1024)
    ap.add_argument("--exemplar-codec", default="raw")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument(
        "--pool-window", type=float, default=10.0,
        help="Temporal window (seconds) around each PSM top-K exemplar_t "
             "to mean-pool embeddings over before re-scoring. W=0 disables "
             "pooling (returns PSM-only ordering).",
    )
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--clip-checkpoint",
                    default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    ap.add_argument("--clip-device", default="cpu")
    ap.add_argument("--psm-binary", type=Path,
                    default=REPO_ROOT / "targets" / "psm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # CLIP text encoder for the query embedding.
    from psm_extraction.models import make_runner
    runner = make_runner("clip", checkpoint=args.clip_checkpoint,
                         backend="auto", device=args.clip_device)

    emb_bank, ts_bank, session_start_h5 = _load_bank(args.features, args.group)

    with args.questions.open() as f:
        spec = yaml.safe_load(f)
    questions = spec.get("questions") or []
    session_start = float(spec.get("session_start_unix", 0.0))
    if "session_start_unix" not in spec:
        session_start = auto_session_start(args.features, args.group)
    iou_threshold = float(spec.get("iou_threshold", 0.3))

    half_W = args.pool_window / 2.0
    records: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="psm-pool-eval-"))
    try:
        for q in questions:
            qid = q.get("id") or f"q{len(records) + 1}"
            text = q.get("query")
            if not text:
                raise SystemExit(f"question {qid!r} missing 'query'")
            gts_rel = [
                (float(iv[0]), float(iv[1])) for iv in q.get("intervals", [])
            ]

            qvec = runner.embed_text(text).astype(np.float32)
            qpath = tmp_dir / f"{qid}.f32"
            qpath.write_bytes(qvec.tobytes())
            payload = run_psm_search(
                args.psm_binary, args.features, args.group, qpath,
                top=args.top,
                time_window=args.time_window,
                capacity=args.capacity,
                h3_resolution=args.h3_resolution,
                precision=args.precision,
                exemplars=args.exemplars,
                seed=(None if args.seed < 0 else args.seed),
                exemplar_codec=args.exemplar_codec,
                verbose=args.verbose,
                per_cell_cap=args.per_cell_cap,
            )
            results = payload.get("results", [])
            if not results:
                # PSM returned nothing -- emit empty record so question count
                # matches across runs.
                records.append({
                    "id": qid, "query": text,
                    "intervals_gt": gts_rel,
                    "preds": [],
                    "bucket_iou_at_k": 0.0,
                    "exemplar_iou_at_k": 0.0,
                    "exemplar_hit_at_k": False,
                    "bucket_hit_at_k": False,
                })
                continue

            # Pool + re-score each candidate.
            # Normalize query.
            qn = qvec / max(np.linalg.norm(qvec), 1e-12)
            rescored = []
            for r in results:
                t_min = float(r["t_min"]) - session_start
                t_max = float(r["t_max"]) - session_start
                exemplar_t = float(r.get("exemplar_t",
                                         (t_min + t_max) / 2.0)) - session_start
                if args.pool_window > 0:
                    pooled = _pool_around(emb_bank, ts_bank, exemplar_t, half_W)
                else:
                    pooled = None
                if pooled is not None:
                    new_sim = float(np.dot(pooled, qn))
                else:
                    new_sim = float(r.get("similarity", 0.0))
                rescored.append({
                    "cell": r.get("cell"),
                    "lat": r.get("lat"),
                    "lng": r.get("lng"),
                    "similarity": new_sim,
                    "exemplar_t": exemplar_t,
                    "t_min": t_min,
                    "t_max": t_max,
                    "count": r.get("count", 0),
                })

            # Re-sort by new similarity, descending.
            rescored.sort(key=lambda p: -p["similarity"])

            # Score against GT.
            preds = []
            for c in rescored:
                bucket_iou, gt_idx_b = best_iou((c["t_min"], c["t_max"]), gts_rel)
                exemplar_iou, gt_idx_e = best_iou(
                    (c["exemplar_t"] - args.exemplar_tolerance,
                     c["exemplar_t"] + args.exemplar_tolerance), gts_rel)
                exemplar_hit_idx = point_in_any(c["exemplar_t"], gts_rel)
                preds.append({
                    **c,
                    "bucket_iou": bucket_iou,
                    "bucket_matched_gt": gt_idx_b,
                    "exemplar_iou": exemplar_iou,
                    "exemplar_matched_gt": gt_idx_e,
                    "exemplar_hits_gt": exemplar_hit_idx >= 0,
                })

            bucket_at_k = max((p["bucket_iou"] for p in preds), default=0.0)
            exemplar_at_k = max((p["exemplar_iou"] for p in preds), default=0.0)
            any_exemplar_hit = any(p["exemplar_hits_gt"] for p in preds)
            records.append({
                "id": qid, "query": text,
                "intervals_gt": gts_rel,
                "preds": preds,
                "bucket_iou_at_k": bucket_at_k,
                "exemplar_iou_at_k": exemplar_at_k,
                "exemplar_hit_at_k": any_exemplar_hit,
                "bucket_hit_at_k": bucket_at_k >= iou_threshold,
            })
    finally:
        if not args.verbose:
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()
        runner.close()

    scored = [r for r in records if r["intervals_gt"]]
    bucket_at_k = float(np.mean([r["bucket_iou_at_k"] for r in scored])) if scored else 0.0
    exemplar_at_k = float(np.mean([r["exemplar_iou_at_k"] for r in scored])) if scored else 0.0
    bucket_hit_at_k = float(np.mean([r["bucket_hit_at_k"] for r in scored])) if scored else 0.0
    exemplar_hit_at_k = float(np.mean([r["exemplar_hit_at_k"] for r in scored])) if scored else 0.0

    out_doc = {
        "features": str(args.features),
        "questions": str(args.questions),
        "group": args.group,
        "method": "psm_temporal_pool",
        "pool_window_sec": args.pool_window,
        "per_cell_cap": args.per_cell_cap,
        "n_questions": len(records),
        "n_scored": len(scored),
        "summary": {
            "bucket_miou_at_5": bucket_at_k,
            "exemplar_miou_at_5": exemplar_at_k,
            "bucket_hit_rate_at_5": bucket_hit_at_k,
            "exemplar_hit_rate_at_5": exemplar_hit_at_k,
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"[psm-pool] W={args.pool_window:>5.1f}s  pcc={args.per_cell_cap}  "
          f"hit@5={exemplar_hit_at_k*100:5.1f}%  "
          f"exemplar_mIoU@5={exemplar_at_k:.3f}  "
          f"bucket_mIoU@5={bucket_at_k:.3f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
