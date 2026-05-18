#!/usr/bin/env python3
"""Generate the §4 compression-vs-Hit@5 figure for the v2 writeup.

Reads per-codec JSON aggregates from captures/eval_*_clipBigG_*_e128_s*.json
(produced by scripts/eval_bigg_all.sh CODEC=...) and plots:
  - x-axis: bytes per exemplar (log scale)
  - y-axis: exemplar Hit@5 (mean across 3 sessions × 5 seeds × 20 questions)
  - error bars: across-seed std
  - annotations: codec name + compression ratio vs raw

Single-axis design — the headline finding is the trade-off curve being flat,
so a single plot reads cleaner than a multi-panel layout.

Output: journal/figures/codec_tradeoff.png at 300 dpi.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# bytes/exemplar at OpenCLIP-bigG (1280-d) per the v2 writeup §4 table.
BYTES_PER_EXEMPLAR = {
    "raw":           5120,
    "turboquant_4b": 1041,
    "turboquant_3b":  785,
    "turboquant_2b":  529,
}

CODEC_PRETTY = {
    "raw":           "raw f32",
    "turboquant_4b": "TQ 4-bit",
    "turboquant_3b": "TQ 3-bit",
    "turboquant_2b": "TQ 2-bit",
}


def load_codec_runs(captures_dir: Path) -> dict[str, list[dict]]:
    """Group eval JSONs by codec across (session, seed)."""
    by_codec: dict[str, list[dict]] = {c: [] for c in BYTES_PER_EXEMPLAR}
    for path in glob.glob(str(captures_dir / "eval_*_clipBigG*_e128_s*.json")):
        data = json.loads(Path(path).read_text())
        codec = data.get("exemplar_codec") or "raw"
        if codec in by_codec:
            by_codec[codec].append(data)
    return by_codec


def per_seed_hit5(runs: list[dict]) -> list[float]:
    """For a single codec, return one Hit@5 per seed averaged across sessions.

    Mirrors eval_aggregate.py's combined-per-seed math: pool scored records
    across sessions for each seed, then take the rate.
    """
    seeds: dict[int, list[dict]] = {}
    for run in runs:
        seed = run.get("psm_seed")
        if seed is None:
            continue
        scored = [r for r in run.get("records", []) if r.get("intervals_gt")]
        seeds.setdefault(seed, []).extend(scored)
    rates = []
    for seed in sorted(seeds):
        recs = seeds[seed]
        if not recs:
            continue
        rates.append(sum(1 for r in recs if r.get("exemplar_hit_at_k")) / len(recs))
    return rates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--captures", type=Path, default=REPO_ROOT / "captures")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "journal" / "figures" / "codec_tradeoff.png")
    args = ap.parse_args()

    by_codec = load_codec_runs(args.captures)

    points = []  # (bytes, mean_hit5, std_hit5, codec)
    for codec in by_codec:
        rates = per_seed_hit5(by_codec[codec])
        if not rates:
            print(f"[skip] no runs for codec={codec}")
            continue
        mean = statistics.fmean(rates)
        std = statistics.stdev(rates) if len(rates) > 1 else 0.0
        points.append((BYTES_PER_EXEMPLAR[codec], mean, std, codec))
    points.sort(key=lambda p: p[0])

    if not points:
        raise SystemExit("no codec runs found; nothing to plot")

    raw_bytes = BYTES_PER_EXEMPLAR["raw"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    xs = [p[0] for p in points]
    ys = [p[1] * 100 for p in points]
    es = [p[2] * 100 for p in points]
    labels = [p[3] for p in points]

    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="#1f4e79",
                ecolor="#1f4e79", elinewidth=1.4, capsize=4, markersize=8,
                linewidth=1.6, alpha=0.9)

    for x, y, label in zip(xs, ys, labels):
        ratio = raw_bytes / x
        pretty = CODEC_PRETTY[label]
        annotation = f"{pretty}\n({ratio:.1f}× smaller)" if label != "raw" else f"{pretty}\n(baseline)"
        ax.annotate(
            annotation, xy=(x, y),
            xytext=(0, -34), textcoords="offset points",
            ha="center", va="top",
            fontsize=9, color="#333333",
        )

    # Reference: raw Hit@5 with shaded across-seed band.
    raw_idx = next((i for i, p in enumerate(points) if p[3] == "raw"), None)
    if raw_idx is not None:
        raw_y = points[raw_idx][1] * 100
        raw_e = points[raw_idx][2] * 100
        ax.axhspan(raw_y - raw_e, raw_y + raw_e,
                   facecolor="#999999", alpha=0.15, zorder=0,
                   label=f"raw Hit@5 ± std")

    ax.set_xscale("log")
    ax.set_xlabel("Bytes per exemplar (log scale)")
    ax.set_ylabel("exemplar Hit @5 (%)")
    ax.set_title("Compression vs answer quality\n"
                 "(OpenCLIP-bigG, 3 sessions × 5 seeds × 20 questions)",
                 fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_ylim(70, 90)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    print("\npoints:")
    for b, m, s, c in points:
        print(f"  {c:14s} bytes={b:5d}  hit5={m*100:5.2f}% ± {s*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
