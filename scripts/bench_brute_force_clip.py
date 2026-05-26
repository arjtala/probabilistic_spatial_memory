#!/usr/bin/env python3
"""Brute-force CLIP retrieval latency + memory benchmark (paper item 7).

Times the dot-product + top-k retrieval step that `eval_brute_force_clip.py`
does on a real session, isolated from the per-query CLIP text-encoding
(which both PSM and brute-force pay equally). Reports median µs per
query so it pairs directly with the PSM `query_similar` benchmark
emitted by `targets/benchmark_spatial_memory`.

Also reports the memory footprint: bytes held in RAM to make brute-force
work (the full N-frame embedding bank). Pairs with PSM's bounded per-
tile state for the paper's memory-vs-session-length plot.

This is *not* the eval harness — it doesn't score against ground truth.
It's the systems-level cost of the retrieval primitive, the missing
half of the E11 picture.

Usage:

    python scripts/bench_brute_force_clip.py \\
        datasets/1501677363692556/clip_bigg_features.h5 \\
        --warmup 5 --trials 200

For a paired PSM benchmark:

    ./targets/benchmark_spatial_memory

Both report median µs per query; combine for the F6 figure.

Output: a JSON record at --out (default
benchmarks/bench_brute_force_<features-basename>.json) plus a markdown
summary table to stdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def run_one_trial(emb_unit: np.ndarray, qvec: np.ndarray, top: int) -> float:
    """Time a single brute-force query: dot product + argpartition + sort.

    Mirrors `eval_brute_force_clip.topk_brute_force` byte-for-byte so the
    µs/query number is honest. CLIP text encoding is excluded — PSM pays
    that cost too; the per-query latency that differs between methods
    is the dot+rank step.
    """
    t0 = time.perf_counter()
    sims = emb_unit @ qvec
    k = min(top, sims.shape[0])
    idx_unsorted = np.argpartition(-sims, k - 1)[:k]
    _ = idx_unsorted[np.argsort(-sims[idx_unsorted])]
    return (time.perf_counter() - t0) * 1e6  # µs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path, help="HDF5 features file")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--warmup", type=int, default=5,
        help="discard first N trials (cache/branch-predictor warmup)",
    )
    ap.add_argument(
        "--trials", type=int, default=200,
        help="timed trials per query (default 200; >=100 for stable median)",
    )
    ap.add_argument(
        "--n-queries", type=int, default=20,
        help="synthetic random queries, drawn from a unit-norm normal in "
             "the embedding's dimensionality; representative of CLIP text "
             "embeddings since both are L2-unit",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="numpy seed for the synthetic query vectors (reproducibility)",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.features, "r") as f:
        if args.group not in f:
            raise SystemExit(f"group {args.group!r} not in {args.features}")
        emb = f[f"{args.group}/embeddings"][:].astype(np.float32)
    n_frames, dim = emb.shape

    # L2-normalize per frame so the dot product equals cosine. This is
    # the in-RAM representation brute-force actually consults.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    emb_unit = (emb / safe).astype(np.float32)
    bytes_in_ram = emb_unit.nbytes  # the "bounded memory" baseline cost

    # Synthetic queries shaped like CLIP text embeddings: unit normal,
    # then L2-normalized to match.
    queries = rng.standard_normal(size=(args.n_queries, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Warm the cache / branch predictor with `--warmup` untimed reps,
    # then collect `--trials` timed reps per query. Returning the median
    # smooths over kernel-context-switch jitter.
    all_us: list[float] = []
    per_query_medians: list[float] = []
    for qi in range(args.n_queries):
        qvec = queries[qi]
        for _ in range(args.warmup):
            _ = run_one_trial(emb_unit, qvec, args.top)
        trials = [run_one_trial(emb_unit, qvec, args.top)
                  for _ in range(args.trials)]
        all_us.extend(trials)
        per_query_medians.append(statistics.median(trials))

    overall_median = statistics.median(all_us)
    overall_p95 = float(np.percentile(all_us, 95))
    overall_p99 = float(np.percentile(all_us, 99))
    per_query_median = statistics.fmean(per_query_medians)

    # Markdown summary to stdout.
    print()
    print(f"## brute_force_clip latency on {args.features.name}::{args.group}")
    print(
        f"_n_frames={n_frames}, dim={dim}, top={args.top}, "
        f"trials={args.trials} × queries={args.n_queries} = "
        f"{args.trials * args.n_queries} timed ops_"
    )
    print()
    print("| metric | value |")
    print("|---|---|")
    print(f"| median µs / query | **{overall_median:.2f}** |")
    print(f"| p95 µs / query | {overall_p95:.2f} |")
    print(f"| p99 µs / query | {overall_p99:.2f} |")
    print(f"| mean-of-per-query-medians µs | {per_query_median:.2f} |")
    print(f"| bytes in RAM (embedding bank) | **{bytes_in_ram:,}** "
          f"({bytes_in_ram / 1024 / 1024:.1f} MiB) |")
    print(f"| bytes / frame | {bytes_in_ram / n_frames:.0f} "
          f"(= dim × 4) |")
    print()
    print(
        "_Compare against PSM's `targets/benchmark_spatial_memory` "
        "`query_similar` row for the paired latency number. Memory is "
        "the linearly-scaling cost brute-force pays that PSM does not._"
    )

    out_path = args.out
    if out_path is None:
        out_path = (Path(__file__).resolve().parents[1] / "benchmarks"
                    / f"bench_brute_force_{args.features.stem}.json")
    record = {
        "features": str(args.features),
        "group": args.group,
        "n_frames": int(n_frames),
        "dim": int(dim),
        "top": args.top,
        "warmup": args.warmup,
        "trials_per_query": args.trials,
        "n_queries": args.n_queries,
        "median_us": overall_median,
        "p95_us": overall_p95,
        "p99_us": overall_p99,
        "mean_per_query_median_us": per_query_median,
        "bytes_in_ram": int(bytes_in_ram),
        "bytes_per_frame": int(bytes_in_ram // n_frames),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(f"\n[bench] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
