#!/usr/bin/env python3
"""Bootstrap confidence intervals on per-question metrics from eval JSONs.

Reads one or more JSON files produced by eval_lookback.py (or any script
that writes the same ``records`` array with ``exemplar_iou_at_k``,
``bucket_iou_at_k``, and ``exemplar_hit_at_k`` fields) and prints a table
with 95 % bootstrap CIs (percentile method, 10 000 resamples) for:

  - exemplar mIoU @k
  - bucket mIoU @k
  - Hit @k rate

Use:

    python scripts/bootstrap_ci.py captures/eval_*_clipBigG_e128_s0.json
    python scripts/bootstrap_ci.py --aggregate captures/eval_*_clipBigG_e128_s*.json
    python scripts/bootstrap_ci.py --json captures/eval_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via the percentile bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(values)
    mean = float(np.mean(values))
    boot = np.array(
        [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_resamples)]
    )
    alpha = (1 - ci) / 2
    return mean, float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


def bootstrap_paired_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the per-question mean of (a - b).

    Inputs must be paired (same length, same question order). Each
    resample draws one set of indices and reuses it for both arrays,
    preserving the per-question pairing.
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal-length arrays ({len(a)} vs {len(b)})")
    rng = np.random.RandomState(seed)
    diffs = a - b
    n = len(diffs)
    mean = float(np.mean(diffs))
    idx = rng.randint(0, n, size=(n_resamples, n))
    boot = diffs[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return mean, float(np.percentile(boot, 100 * alpha)), float(np.percentile(boot, 100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _scored_records(data: dict) -> list[dict]:
    """Return only scored records (those with ground-truth intervals).

    Accepts both eval_lookback.py output (``records``) and
    eval_psm_mllm.py output (``questions_out``).
    """
    raw = data.get("records") or data.get("questions_out") or []
    return [r for r in raw if r.get("intervals_gt")]


def _extract_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    """Pull the three metric arrays from scored records."""
    return {
        "exemplar_miou": np.array([r["exemplar_iou_at_k"] for r in records], dtype=float),
        "bucket_miou": np.array([r["bucket_iou_at_k"] for r in records], dtype=float),
        "hit_at_k": np.array(
            [1.0 if r["exemplar_hit_at_k"] else 0.0 for r in records], dtype=float,
        ),
    }


def _compute_cis(
    arrays: dict[str, np.ndarray],
    n_resamples: int,
    ci: float,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap CIs for each metric array."""
    out: dict[str, tuple[float, float, float]] = {}
    for name, values in arrays.items():
        if len(values) == 0:
            out[name] = (0.0, 0.0, 0.0)
        else:
            out[name] = bootstrap_ci(values, n_resamples=n_resamples, ci=ci, seed=seed)
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(mean: float, lo: float, hi: float, is_pct: bool = False) -> str:
    if is_pct:
        return f"{mean:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"


def _label(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Bootstrap CIs on per-question metrics from eval JSONs.",
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="eval JSON file(s)")
    ap.add_argument(
        "--aggregate", action="store_true",
        help="pool all questions across input files and compute a single CI row",
    )
    ap.add_argument(
        "--json", action="store_true", dest="json_out",
        help="print machine-readable JSON instead of a table",
    )
    ap.add_argument(
        "--resamples", type=int, default=10_000,
        help="number of bootstrap resamples (default: 10000)",
    )
    ap.add_argument(
        "--ci", type=float, default=0.95,
        help="confidence level (default: 0.95)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    ap.add_argument(
        "--diff", action="store_true",
        help=(
            "compute a paired bootstrap CI on the per-question difference "
            "(first input minus second input). Requires exactly two input "
            "files whose scored records share matching `id` fields."
        ),
    )
    args = ap.parse_args()

    # Load all files.
    loaded: list[tuple[Path, dict]] = []
    for path in args.inputs:
        if not path.exists():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 1
        loaded.append((path, json.loads(path.read_text())))

    if args.diff:
        if len(loaded) != 2:
            print("error: --diff requires exactly two input files", file=sys.stderr)
            return 1
        return _print_diff(loaded[0], loaded[1], args)

    # Per-file results.
    results: list[dict] = []
    all_records: list[dict] = []
    for path, data in loaded:
        scored = _scored_records(data)
        all_records.extend(scored)
        arrays = _extract_arrays(scored)
        cis = _compute_cis(arrays, args.resamples, args.ci, args.seed)
        results.append({
            "file": str(path),
            "label": _label(path),
            "n_scored": len(scored),
            "exemplar_miou": cis["exemplar_miou"],
            "bucket_miou": cis["bucket_miou"],
            "hit_at_k": cis["hit_at_k"],
        })

    # Aggregate (pooled across all files).
    agg_result: dict | None = None
    if args.aggregate:
        arrays = _extract_arrays(all_records)
        cis = _compute_cis(arrays, args.resamples, args.ci, args.seed)
        agg_result = {
            "label": "pooled",
            "n_scored": len(all_records),
            "exemplar_miou": cis["exemplar_miou"],
            "bucket_miou": cis["bucket_miou"],
            "hit_at_k": cis["hit_at_k"],
        }

    # Output.
    if args.json_out:
        return _print_json(results, agg_result, args)
    return _print_table(results, agg_result, args)


def _print_diff(
    a_pair: tuple[Path, dict],
    b_pair: tuple[Path, dict],
    args: argparse.Namespace,
) -> int:
    """Print a paired-difference bootstrap CI between two eval files."""
    a_path, a_data = a_pair
    b_path, b_data = b_pair
    a_scored = _scored_records(a_data)
    b_scored = _scored_records(b_data)
    a_by_id = {r["id"]: r for r in a_scored}
    b_by_id = {r["id"]: r for r in b_scored}
    common = sorted(set(a_by_id) & set(b_by_id))
    if not common:
        print("error: no shared scored record ids between the two files", file=sys.stderr)
        return 1
    a_only = sorted(set(a_by_id) - set(b_by_id))
    b_only = sorted(set(b_by_id) - set(a_by_id))

    metrics = ["exemplar_iou_at_k", "bucket_iou_at_k", "exemplar_hit_at_k"]
    rows: list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
    for m in metrics:
        a_vals = np.array(
            [(1.0 if a_by_id[i].get(m) else 0.0) if "hit" in m else float(a_by_id[i].get(m, 0.0)) for i in common],
            dtype=float,
        )
        b_vals = np.array(
            [(1.0 if b_by_id[i].get(m) else 0.0) if "hit" in m else float(b_by_id[i].get(m, 0.0)) for i in common],
            dtype=float,
        )
        a_ci = bootstrap_ci(a_vals, args.resamples, args.ci, args.seed)
        b_ci = bootstrap_ci(b_vals, args.resamples, args.ci, args.seed)
        d_ci = bootstrap_paired_diff_ci(a_vals, b_vals, args.resamples, args.ci, args.seed)
        rows.append((m, a_ci, b_ci, d_ci))

    if args.json_out:
        out = {
            "ci": args.ci,
            "resamples": args.resamples,
            "seed": args.seed,
            "a": {"file": str(a_path), "label": _label(a_path), "n_scored": len(a_scored)},
            "b": {"file": str(b_path), "label": _label(b_path), "n_scored": len(b_scored)},
            "n_paired": len(common),
            "n_a_only": len(a_only),
            "n_b_only": len(b_only),
            "metrics": {
                m: {
                    "a": _ci_to_dict(a_ci),
                    "b": _ci_to_dict(b_ci),
                    "diff_a_minus_b": _ci_to_dict(d_ci),
                }
                for (m, a_ci, b_ci, d_ci) in rows
            },
        }
        print(json.dumps(out, indent=2))
        return 0

    ci_pct = args.ci * 100
    print()
    print(
        f"## Paired bootstrap {ci_pct:.0f}% CI on diff "
        f"({args.resamples} resamples, seed={args.seed})"
    )
    print(f"  A: {a_path}  (n_scored={len(a_scored)})")
    print(f"  B: {b_path}  (n_scored={len(b_scored)})")
    print(f"  paired n = {len(common)}  (A-only: {len(a_only)}, B-only: {len(b_only)})")
    print()
    print("| metric | A | B | A - B |")
    print("|---|---|---|---|")
    for m, a_ci, b_ci, d_ci in rows:
        is_pct = "hit" in m
        print(
            f"| {m} | {_fmt(*a_ci, is_pct=is_pct)} | "
            f"{_fmt(*b_ci, is_pct=is_pct)} | {_fmt(*d_ci, is_pct=is_pct)} |"
        )
    print()
    return 0


def _ci_to_dict(ci: tuple[float, float, float]) -> dict:
    return {"mean": ci[0], "ci_low": ci[1], "ci_high": ci[2]}


def _print_json(
    results: list[dict],
    agg_result: dict | None,
    args: argparse.Namespace,
) -> int:
    out: dict = {
        "ci": args.ci,
        "resamples": args.resamples,
        "seed": args.seed,
        "per_file": [
            {
                "file": r["file"],
                "label": r["label"],
                "n_scored": r["n_scored"],
                "exemplar_miou": _ci_to_dict(r["exemplar_miou"]),
                "bucket_miou": _ci_to_dict(r["bucket_miou"]),
                "hit_at_k": _ci_to_dict(r["hit_at_k"]),
            }
            for r in results
        ],
    }
    if agg_result is not None:
        out["pooled"] = {
            "n_scored": agg_result["n_scored"],
            "exemplar_miou": _ci_to_dict(agg_result["exemplar_miou"]),
            "bucket_miou": _ci_to_dict(agg_result["bucket_miou"]),
            "hit_at_k": _ci_to_dict(agg_result["hit_at_k"]),
        }
    print(json.dumps(out, indent=2))
    return 0


def _print_table(
    results: list[dict],
    agg_result: dict | None,
    args: argparse.Namespace,
) -> int:
    ci_pct = args.ci * 100
    print()
    print(f"## Bootstrap {ci_pct:.0f}% CIs ({args.resamples} resamples, seed={args.seed})")
    print()
    print(
        "| file | n | exemplar mIoU @k | bucket mIoU @k | Hit @k |"
    )
    print("|---|---|---|---|---|")

    for r in results:
        print(
            f"| `{r['label']}` | {r['n_scored']} | "
            f"{_fmt(*r['exemplar_miou'])} | "
            f"{_fmt(*r['bucket_miou'])} | "
            f"{_fmt(*r['hit_at_k'], is_pct=True)} |"
        )

    if agg_result is not None:
        print(
            f"| **pooled** | **{agg_result['n_scored']}** | "
            f"**{_fmt(*agg_result['exemplar_miou'])}** | "
            f"**{_fmt(*agg_result['bucket_miou'])}** | "
            f"**{_fmt(*agg_result['hit_at_k'], is_pct=True)}** |"
        )

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
