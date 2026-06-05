#!/usr/bin/env python3
"""Sliding-window mean-pool the embeddings in a features.h5 file.

Approach 2 in the PSM-vs-sliding-window investigation. The W=30
query-time rerank (approach 1) capped at 15.0% Hit@5 because PSM's
top-50 candidate set is already exhaustive on a 5-10min Nymeria
session. To break the candidate-set ceiling, replace each frame's
embedding with the mean of frames within ±W/2 seconds *before* PSM
sees the data. PSM's H3+reservoir then samples from smoothed
embeddings, so the cell-level candidate generation -- not just the
final ranking -- benefits from temporal pooling.

Same number of frames in / out; only the embeddings change. All
other groups + attrs are passed through unchanged.

Usage:
  python scripts/smooth_features.py \\
    /checkpoint/.../<sid>/clip_l_features.h5 \\
    /checkpoint/.../<sid>/clip_l_features_pool30s.h5 \\
    --window 30 --group clip
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def smooth_embeddings(emb: np.ndarray, ts: np.ndarray, window_s: float) -> np.ndarray:
    """Replace each row with the mean of rows whose timestamps fall in
    [t_i - W/2, t_i + W/2]. The output is NOT re-normalized -- PSM's
    cosine path normalizes on the fly. If callers want unit-norm
    embeddings, normalize before passing in.
    """
    half = window_s / 2.0
    # Timestamps are typically monotonic (extraction writes them in order),
    # but use sort indices defensively.
    order = np.argsort(ts)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ts_sorted = ts[order]
    emb_sorted = emb[order]

    out = np.empty_like(emb_sorted)
    # Two-pointer sliding window. ts_sorted is monotonic.
    n = len(ts_sorted)
    lo = 0
    hi = 0
    for i in range(n):
        t = ts_sorted[i]
        while lo < n and ts_sorted[lo] < t - half:
            lo += 1
        while hi < n and ts_sorted[hi] <= t + half:
            hi += 1
        # window is [lo, hi)
        out[i] = emb_sorted[lo:hi].mean(axis=0)
    # Restore original order.
    return out[inv]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--window", type=float, required=True,
                    help="Pool window in seconds (each frame's embedding "
                         "becomes the mean of frames in ±window/2).")
    ap.add_argument("--group", default="clip",
                    help="HDF5 group containing /embeddings and /timestamps. "
                         "Other groups are copied unchanged.")
    args = ap.parse_args()

    if args.dst.exists():
        raise SystemExit(f"refusing to overwrite existing {args.dst}")
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    # Copy then mutate the target group's embeddings in place. h5py
    # doesn't let us overwrite a dataset's contents if shape matches,
    # but we can delete and recreate. Cleanest: copy file, then edit.
    shutil.copy2(args.src, args.dst)

    with h5py.File(args.dst, "r+") as f:
        emb = f[f"{args.group}/embeddings"][:].astype(np.float32)
        ts = f[f"{args.group}/timestamps"][:].astype(np.float64)
        if emb.shape[0] != ts.shape[0]:
            raise SystemExit(
                f"embedding/timestamp count mismatch: "
                f"{emb.shape[0]} vs {ts.shape[0]}")
        smoothed = smooth_embeddings(emb, ts, args.window).astype(np.float32)
        # Preserve dataset attributes (e.g., model id).
        attrs = dict(f[f"{args.group}/embeddings"].attrs)
        del f[f"{args.group}/embeddings"]
        ds = f.create_dataset(f"{args.group}/embeddings", data=smoothed,
                              chunks=True, compression="gzip",
                              compression_opts=4)
        for k, v in attrs.items():
            ds.attrs[k] = v
        # Mark provenance so we don't confuse smoothed vs raw banks later.
        f.attrs["smoothed_pool_window_s"] = float(args.window)
        f.attrs["smoothed_source"] = str(args.src.name)

    print(f"[smooth] window={args.window:.1f}s  "
          f"{emb.shape[0]} frames -> {args.dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
