#!/usr/bin/env python3
"""Drop-in replacement of CLIP text encoder with Long-CLIP, image side unchanged.

KEY INSIGHT: Long-CLIP-L's image tower is byte-identical to OpenAI
CLIP-L. The two models share the same vision projection; only the
text encoder's positional embedding is extended (77 -> 248 tokens).
That means we can swap the text encoder *at query time only*,
reading the existing clip_l_features.h5 bank for the image side.
No re-extraction needed.

Pipeline per question:
  1. Embed text query with Long-CLIP (248-token context).
  2. Score against the existing clip_l_features.h5 image embeddings
     by cosine similarity.
  3. Brute-force top-5 retrieval (no PSM substrate; this isolates
     the text-encoder effect).

This replaces eval_brute_force_clip.py's text-encoding step.

Usage:
  python scripts/eval_longclip_text.py \\
    /checkpoint/.../<sid>/clip_l_features.h5 \\
    /checkpoint/.../<sid>/questions.yaml \\
    --out captures/eval_<sid>_longclip_text.json
"""
from __future__ import annotations

import argparse
import json
import sys
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
)


def _load_bank(features_h5: Path, group: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(features_h5, "r") as f:
        emb = f[f"{group}/embeddings"][:].astype(np.float32)
        ts = f[f"{group}/timestamps"][:].astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / np.where(norms > 0, norms, 1.0)).astype(np.float32), ts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--context-length", type=int, default=248)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from psm_extraction.models import make_runner
    runner = make_runner("longclip", device=args.device,
                         context_length=args.context_length)
    if runner.embedding_dim != _load_bank(args.features, args.group)[0].shape[1]:
        # Reload to print the actual mismatch.
        bank_dim = _load_bank(args.features, args.group)[0].shape[1]
        raise SystemExit(
            f"text-encoder dim ({runner.embedding_dim}) does not match "
            f"image bank dim ({bank_dim}); Long-CLIP-L should be 768-d "
            f"and the bank should be the CLIP-L bank.")

    emb_bank, ts_bank = _load_bank(args.features, args.group)
    session_start = float(ts_bank[0])
    ts_rel = ts_bank - session_start

    with args.questions.open() as f:
        spec = yaml.safe_load(f)
    questions = spec.get("questions") or []
    iou_threshold = float(spec.get("iou_threshold", 0.3))

    records: list[dict] = []
    for q in questions:
        qid = q.get("id") or f"q{len(records)+1}"
        text = q.get("query")
        if not text:
            raise SystemExit(f"question {qid!r} missing 'query'")
        gts_rel = [(float(iv[0]), float(iv[1])) for iv in q.get("intervals", [])]

        qvec = runner.embed_text(text).astype(np.float32)
        qn = qvec / max(np.linalg.norm(qvec), 1e-12)
        sims = emb_bank @ qn  # (N,)
        top_idx = np.argpartition(-sims, args.top)[: args.top]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        preds = []
        for i in top_idx:
            t = float(ts_rel[i])
            bucket_iou, gt_idx_b = best_iou(
                (t - args.exemplar_tolerance, t + args.exemplar_tolerance), gts_rel)
            exemplar_iou, gt_idx_e = best_iou(
                (t - args.exemplar_tolerance, t + args.exemplar_tolerance), gts_rel)
            exemplar_hit_idx = point_in_any(t, gts_rel)
            preds.append({
                "exemplar_t": t,
                "similarity": float(sims[i]),
                "bucket_iou": bucket_iou,
                "exemplar_iou": exemplar_iou,
                "exemplar_hits_gt": exemplar_hit_idx >= 0,
            })

        bucket_at_k = max((p["bucket_iou"] for p in preds), default=0.0)
        exemplar_at_k = max((p["exemplar_iou"] for p in preds), default=0.0)
        any_hit = any(p["exemplar_hits_gt"] for p in preds)
        records.append({
            "id": qid, "query": text, "intervals_gt": gts_rel, "preds": preds,
            "bucket_iou_at_k": bucket_at_k,
            "exemplar_iou_at_k": exemplar_at_k,
            "exemplar_hit_at_k": any_hit,
            "bucket_hit_at_k": bucket_at_k >= iou_threshold,
        })

    runner.close()

    scored = [r for r in records if r["intervals_gt"]]
    n = len(scored) or 1
    summary = {
        "bucket_miou_at_5": float(np.mean([r["bucket_iou_at_k"] for r in scored])),
        "exemplar_miou_at_5": float(np.mean([r["exemplar_iou_at_k"] for r in scored])),
        "bucket_hit_rate_at_5": float(np.mean([r["bucket_hit_at_k"] for r in scored])),
        "exemplar_hit_rate_at_5": float(np.mean([r["exemplar_hit_at_k"] for r in scored])),
    }
    out_doc = {
        "method": "longclip_text_brute_force",
        "context_length": args.context_length,
        "features": str(args.features),
        "n_questions": len(records),
        "n_scored": len(scored),
        "summary": summary,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"[longclip] hit@5={summary['exemplar_hit_rate_at_5']*100:5.1f}%  "
          f"exemplar_mIoU@5={summary['exemplar_miou_at_5']:.3f}  "
          f"({args.out})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
