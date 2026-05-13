#!/usr/bin/env python3
"""Compute codec-induced retrieval drift from eval_lookback JSON outputs.

Pairs runs that share (session, seed, exemplars, top, encoder, exemplar
reservoir parameters) but differ in `exemplar_codec`. For each non-raw codec
reports, per question and aggregated:

  - top-k cell overlap (Jaccard) at k = 1, 5, 10
  - rank correlation on cells common to both rankings (Spearman if scipy is
    available, else a small in-tree implementation; falls back to "n/a" when
    fewer than 2 cells are shared)
  - winning-exemplar cosine error vs raw (similarity values from the JSON)

This isolates codec impact from question-IoU noise (which depends on whether
the shifted top-1 cell happens to land inside a GT interval).

Use:

    python scripts/eval_codec_drift.py captures/eval_*_clipBigG_e128_s*.json \\
        captures/eval_*_clipBigG_turboquant_4b_e128_s*.json \\
        captures/eval_*_clipBigG_turboquant_2b_e128_s*.json

Outputs a markdown table to stdout. With --out PATH writes the per-question
breakdown as JSON for downstream plotting.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def jaccard(a: list[str], b: list[str], k: int) -> float:
    sa = set(a[:k])
    sb = set(b[:k])
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def spearman_on_common(rank_a: dict[str, int],
                       rank_b: dict[str, int]) -> float | None:
    """Spearman ρ over cells common to both rankings.

    Both inputs map cell -> 0-based rank in their respective top-k. Returns
    None when fewer than 2 cells overlap (correlation is undefined).
    """
    common = sorted(set(rank_a) & set(rank_b))
    n = len(common)
    if n < 2:
        return None
    ra = [rank_a[c] for c in common]
    rb = [rank_b[c] for c in common]
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = sum((ra[i] - mean_a) ** 2 for i in range(n))
    den_b = sum((rb[i] - mean_b) ** 2 for i in range(n))
    if den_a == 0.0 or den_b == 0.0:
        # Constant ranking on at least one side — correlation undefined.
        return None
    return num / (den_a ** 0.5 * den_b ** 0.5)


def load_runs(paths: list[Path]) -> list[dict]:
    runs = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
        data = json.loads(p.read_text())
        data["__source"] = str(p)
        runs.append(data)
    return runs


def pair_key(run: dict) -> tuple:
    """The tuple that must match between raw and codec runs to compare them.

    Codec is intentionally excluded — that's the axis we vary.
    """
    return (
        Path(str(run.get("features") or "")).parent.name,
        Path(str(run.get("questions_file") or "")).name,
        run.get("psm_seed"),
        run.get("exemplars"),
        run.get("top"),
        run.get("clip_checkpoint"),
        run.get("h3_resolution"),
        run.get("capacity"),
        run.get("precision"),
        run.get("time_window_sec"),
    )


def question_preds(record: dict) -> list[dict]:
    return record.get("preds") or []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument(
        "--ks", default="1,5,10",
        help="comma-separated top-k values for Jaccard (default 1,5,10)",
    )
    ap.add_argument(
        "--out", type=Path,
        help="write per-question drift records as JSON",
    )
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    runs = load_runs(args.inputs)
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for run in runs:
        codec = run.get("exemplar_codec") or "raw"
        by_key[pair_key(run)][codec] = run

    # Pair every non-raw codec with the matching raw run.
    drift_records: list[dict] = []
    for key, codecs in by_key.items():
        raw_run = codecs.get("raw")
        if raw_run is None:
            print(
                f"[drift] no raw companion for key {key}; skipping codecs "
                f"{sorted(c for c in codecs if c != 'raw')}",
                file=sys.stderr,
            )
            continue
        raw_records = {r["id"]: r for r in raw_run.get("records", [])}
        for codec, other in codecs.items():
            if codec == "raw":
                continue
            for r_other in other.get("records", []):
                qid = r_other["id"]
                r_raw = raw_records.get(qid)
                if r_raw is None:
                    continue
                raw_preds = question_preds(r_raw)
                oth_preds = question_preds(r_other)
                cells_raw = [p["cell"] for p in raw_preds]
                cells_oth = [p["cell"] for p in oth_preds]
                rank_raw = {c: i for i, c in enumerate(cells_raw)}
                rank_oth = {c: i for i, c in enumerate(cells_oth)}
                top1_match = bool(cells_raw and cells_oth and
                                  cells_raw[0] == cells_oth[0])
                # Winning-exemplar cosine error: raw top-1 similarity minus
                # codec top-1 similarity. Negative means the codec scored its
                # top-1 *higher* than raw (rare; happens when the new winner
                # has a different exemplar with a closer rotated projection).
                sim_raw = (raw_preds[0].get("similarity")
                           if raw_preds else None)
                sim_oth = (oth_preds[0].get("similarity")
                           if oth_preds else None)
                if sim_raw is not None and sim_oth is not None:
                    sim_delta = sim_raw - sim_oth
                else:
                    sim_delta = None

                drift_records.append({
                    "session": Path(str(raw_run.get("features"))).parent.name,
                    "questions_file": Path(
                        str(raw_run.get("questions_file"))).name,
                    "seed": raw_run.get("psm_seed"),
                    "exemplars": raw_run.get("exemplars"),
                    "top": raw_run.get("top"),
                    "encoder": raw_run.get("clip_checkpoint"),
                    "codec": codec,
                    "qid": qid,
                    "category": r_raw.get("category"),
                    "top1_match": top1_match,
                    "jaccard": {
                        f"top{k}": jaccard(cells_raw, cells_oth, k)
                        for k in ks
                    },
                    "rank_correlation":
                        spearman_on_common(rank_raw, rank_oth),
                    "n_common_cells": len(set(rank_raw) & set(rank_oth)),
                    "top1_similarity_raw": sim_raw,
                    "top1_similarity_codec": sim_oth,
                    "top1_similarity_delta": sim_delta,
                })

    if not drift_records:
        print("[drift] no paired codec/raw runs found; nothing to report",
              file=sys.stderr)
        return 1

    # ---- Aggregate by codec ----
    by_codec: dict[str, list[dict]] = defaultdict(list)
    for d in drift_records:
        by_codec[d["codec"]].append(d)

    print()
    print("## Codec drift vs raw")
    sample_run = next(iter(runs))
    print(
        f"_paired by (session, seed, exemplars, top, encoder, retention); "
        f"top={sample_run.get('top')}, exemplars={sample_run.get('exemplars')}, "
        f"encoder=`{sample_run.get('clip_checkpoint')}`_"
    )
    print()
    headers = (
        "| codec | n questions | top-1 match "
        + "".join(f"| Jaccard top-{k} " for k in ks)
        + "| rank ρ (mean) | top-1 cos delta (mean ± std) |"
    )
    print(headers)
    print("|" + "---|" * (3 + len(ks) + 2))
    for codec in sorted(by_codec):
        rows = by_codec[codec]
        n = len(rows)
        top1_rate = sum(1 for r in rows if r["top1_match"]) / n
        jacc_means = {
            f"top{k}":
                sum(r["jaccard"][f"top{k}"] for r in rows) / n
            for k in ks
        }
        rho_vals = [r["rank_correlation"] for r in rows
                    if r["rank_correlation"] is not None]
        rho_mean = statistics.fmean(rho_vals) if rho_vals else None
        sim_deltas = [r["top1_similarity_delta"] for r in rows
                      if r["top1_similarity_delta"] is not None]
        if sim_deltas:
            sd_mean = statistics.fmean(sim_deltas)
            sd_std = (statistics.stdev(sim_deltas)
                      if len(sim_deltas) > 1 else 0.0)
            sd_str = f"{sd_mean:+.4f} ± {sd_std:.4f}"
        else:
            sd_str = "n/a"
        rho_str = f"{rho_mean:.3f}" if rho_mean is not None else "n/a"
        jacc_cells = "".join(
            f"| {jacc_means[f'top{k}']:.3f} " for k in ks
        )
        print(
            f"| `{codec}` | {n} | {top1_rate:.1%} "
            f"{jacc_cells}| {rho_str} | {sd_str} |"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "ks": ks,
            "n_records": len(drift_records),
            "codecs": sorted(by_codec),
            "records": drift_records,
        }, indent=2))
        print(f"\n[drift] wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
