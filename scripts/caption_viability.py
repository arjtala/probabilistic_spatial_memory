#!/usr/bin/env python3
"""Policy-free caption viability gate (PREREGISTRATION.md Amendment B).

Answers one question per bank: *can the encoder find the captioned frame at
all?* Retrieval runs over the FULL unbudgeted frame bank — no budget, no
retention policy. If a caption cannot retrieve its own ground-truth interval
when every frame is available, no policy can rescue it under a budget, so the
fixed-budget comparison would have no headroom to measure.

Two frozen criteria (Amendment B):

  * **decisive** — >= --min-top5-rate of captions retrieve a GT interval within
    top-5 on the full bank (default 0.50);
  * **degeneracy guard** — median pairwise cosine between caption *text*
    embeddings <= --max-median-cosine (default 0.80), catching a bank where
    every caption says the same thing ("a two-story building on the right").

Both must pass. Exit code 0 = gate passed, 1 = failed, so it can gate a run.

Usage:
    python scripts/caption_viability.py \\
        --bank    .../questions.yaml --features .../clip_l_features.h5 \\
        [--bank ... --features ... ]... \\
        --out captures/hdd/caption_viability_v3.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
from eval_fixed_budget import make_runner  # noqa: E402

_DEFAULT_CHECKPOINT = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"


def _unit(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def score_bank(bank_path: Path, features: Path, runner, *, top_k: int) -> dict:
    doc = yaml.safe_load(bank_path.read_text()) or {}
    questions = [q for q in (doc.get("questions") or []) if q.get("query")]
    with h5py.File(features, "r") as f:
        emb = _unit(f["clip/embeddings"][:].astype(np.float32))
        ts = f["clip/timestamps"][:].astype(np.float64)

    ranks: list[int | None] = []
    qvecs: list[np.ndarray] = []
    for q in questions:
        v = _unit(runner.embed_text(q["query"]).astype(np.float32).ravel())
        qvecs.append(v)
        order = np.argsort(-(emb @ v))
        intervals = [(float(a), float(b)) for a, b in q.get("intervals", [])]
        hit = next(
            (
                rank
                for rank, idx in enumerate(order, 1)
                if any(lo <= ts[idx] <= hi for lo, hi in intervals)
            ),
            None,
        )
        ranks.append(hit)

    n = len(questions)
    top5 = sum(1 for r in ranks if r is not None and r <= top_k)
    # Median pairwise cosine over caption text embeddings: a bank of
    # near-identical captions can score well on nothing and must not pass.
    med_cos = None
    if n >= 2:
        q = np.stack(qvecs)
        sims = q @ q.T
        iu = np.triu_indices(n, k=1)
        med_cos = float(np.median(sims[iu]))
    return {
        "bank": str(bank_path),
        "features": str(features),
        "session_id": doc.get("session_id"),
        "prompt_sha256": (doc.get("generator") or {}).get("prompt_sha256"),
        "model": (doc.get("generator") or {}).get("model"),
        "n_questions": n,
        "n_frames": int(emb.shape[0]),
        f"top{top_k}_hits": top5,
        f"top{top_k}_rate": (top5 / n) if n else None,
        "median_pairwise_caption_cosine": med_cos,
        "ranks": ranks,
        "queries": [q["query"] for q in questions],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bank", type=Path, action="append", required=True)
    ap.add_argument("--features", type=Path, action="append", required=True)
    ap.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-top5-rate", type=float, default=0.50,
                    help="frozen decisive criterion (Amendment B)")
    ap.add_argument("--max-median-cosine", type=float, default=0.80,
                    help="frozen degeneracy guard (Amendment B)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if len(args.bank) != len(args.features):
        ap.error("--bank and --features must be given the same number of times")

    runner = make_runner(args.checkpoint, args.device)
    rows = [
        score_bank(b, f, runner, top_k=args.top_k)
        for b, f in zip(args.bank, args.features)
    ]

    key = f"top{args.top_k}_rate"
    n_tot = sum(r["n_questions"] for r in rows)
    hits = sum(r[f"top{args.top_k}_hits"] for r in rows)
    pooled = (hits / n_tot) if n_tot else 0.0
    cosines = [r["median_pairwise_caption_cosine"] for r in rows
               if r["median_pairwise_caption_cosine"] is not None]
    med_cos = statistics.median(cosines) if cosines else None
    per_drive_pass = [
        r["session_id"] for r in rows
        if (r[key] or 0.0) >= args.min_top5_rate
        and (r["median_pairwise_caption_cosine"] or 0.0) <= args.max_median_cosine
    ]
    passed = pooled >= args.min_top5_rate and (
        med_cos is not None and med_cos <= args.max_median_cosine
    )

    print(f"\n{'session':16} {'n':>4} {'top'+str(args.top_k):>6} {'rate':>7} {'med_cos':>8}")
    for r in rows:
        rate = r[key]
        print(f"{str(r['session_id'])[:16]:16} {r['n_questions']:4d} "
              f"{r[f'top{args.top_k}_hits']:6d} "
              f"{(f'{rate:.2f}' if rate is not None else '  n/a'):>7} "
              f"{(f'{r['median_pairwise_caption_cosine']:.3f}' if r['median_pairwise_caption_cosine'] is not None else 'n/a'):>8}")
    print(f"\npooled: {hits}/{n_tot} = {pooled:.1%} "
          f"(bar {args.min_top5_rate:.0%}) | median caption cosine "
          f"{med_cos if med_cos is None else round(med_cos, 3)} "
          f"(ceiling {args.max_median_cosine})")
    print(f"drives clearing the gate individually: {len(per_drive_pass)}/{len(rows)}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "criteria": {
                "min_top5_rate": args.min_top5_rate,
                "max_median_cosine": args.max_median_cosine,
                "top_k": args.top_k,
                "retrieval": "full unbudgeted bank, CLIP text->frame cosine",
            },
            "checkpoint": args.checkpoint,
            "pooled_rate": pooled,
            "pooled_hits": hits,
            "n_questions": n_tot,
            "median_caption_cosine": med_cos,
            "drives_clearing_individually": per_drive_pass,
            "gate_passed": passed,
            "banks": rows,
        }, indent=2))
        print(f"[viability] -> {args.out}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
