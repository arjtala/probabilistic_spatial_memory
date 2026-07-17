"""Fixed-budget result figure (2 panels), generated from captured JSON.

Panel A: Hit@5 vs. logical memory (M / logical MB) for the four streaming/
         diagnostic policies the paper foregrounds (global_reservoir,
         uniform_time, semantic_kcenter, spatial_priority). Seeds are averaged
         within a session, then macro-averaged over sessions. The preregistered
         operating point M=128 / r12 is marked explicitly. This is NOT labelled
         a Pareto plot (we do not draw a nondominated frontier).

Panel B: rare-place and common-place Hit@5 deltas of the spatial probe at the
         preregistered M=128 / r12, relative to the budget-matched
         global_reservoir baseline, with paired session-bootstrap 95% CIs
         (sessions resampled, never questions/seeds).

Reuses the loaders + paired bootstrap from compare_fixed_budget.py so numbers
match the paper's paired tables exactly. Semantic k-center is kept visually
prominent (it wins aggregate Hit@5).

Run:
  /home/arjangt/.conda/envs/psm/bin/python scripts/plot_fixed_budget.py \
      captures/wearables_fixed_budget --recursive
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_fixed_budget import (  # noqa: E402
    expand_inputs, load_capture, paired_bootstrap,
)

OUT_DIR = Path("journal/figures")
# Logical selector state per exemplar: float32 d=768 embedding + float64 ts.
LOGICAL_BYTES_PER_EXEMPLAR = 768 * 4 + 8

# Policies shown in Panel A (order = legend/z-order); k-center emphasised.
PANEL_A = [
    ("semantic_kcenter", "semantic k-center (offline)", "#dd8452", "o", 2.4),
    ("uniform_time",     "uniform time",                "#55a868", "s", 1.6),
    ("global_reservoir", "global reservoir",            "#4c72b0", "^", 1.6),
    ("spatial_priority", "spatial priority (probe)",    "#c44e52", "D", 2.0),
]
PRIMARY_BUDGET = 128
PRIMARY_RES = 12


def collect(paths, resolution, transform="base", top=5):
    """method -> budget -> session -> {metric: seed-averaged value}."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    import json
    for p in paths:
        try:
            data = json.loads(p.read_text())
            if data.get("h3_resolution") != resolution or data.get("top") != top:
                continue
            cap = load_capture(p)
        except Exception:
            continue
        if cap.transform != transform:
            continue
        budget = data.get("exemplar_budget")
        by[cap.method][budget][cap.session].append(cap.metrics)
    # average seeds within session
    out = defaultdict(lambda: defaultdict(dict))
    for method, budgets in by.items():
        for budget, sessions in budgets.items():
            for session, metric_dicts in sessions.items():
                avg = {}
                for key in metric_dicts[0]:
                    vals = [m[key] for m in metric_dicts if m.get(key) is not None]
                    avg[key] = statistics.fmean(vals) if vals else None
                out[method][budget][session] = avg
    return out


def macro(sessions_map, key):
    vals = [m[key] for m in sessions_map.values() if m.get(key) is not None]
    return statistics.fmean(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--resolution", type=int, default=PRIMARY_RES)
    ap.add_argument("--baseline", default="global_reservoir")
    ap.add_argument("--probe", default="spatial_priority")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stem", default="fixed_budget")
    args = ap.parse_args()

    paths = expand_inputs(args.inputs, args.recursive)
    data = collect(paths, args.resolution)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ---- Panel A: Hit@5 vs logical MB ---------------------------------
    for method, label, color, marker, lw in PANEL_A:
        budgets = sorted(b for b in data.get(method, {}) if b is not None)
        xs, ys = [], []
        for b in budgets:
            h = macro(data[method][b], "hit_at_k")
            if h is None:
                continue
            xs.append(b * LOGICAL_BYTES_PER_EXEMPLAR / 1e6)  # decimal MB
            ys.append(h * 100)
        if not xs:
            continue
        axA.plot(xs, ys, marker=marker, color=color, lw=lw, ms=6, label=label,
                 zorder=3 if "priority" in method or "kcenter" in method else 2)
    # Mark the preregistered operating point M=128 / r12.
    x128 = PRIMARY_BUDGET * LOGICAL_BYTES_PER_EXEMPLAR / 1e6
    axA.axvline(x128, ls="--", color="0.5", lw=1.0, zorder=1)
    axA.annotate("preregistered\n$M{=}128$, $r{=}12$", xy=(x128, axA.get_ylim()[0]),
                 xytext=(x128 * 1.04, 10.5), fontsize=8, color="0.35")
    axA.set_xlabel("logical exemplar memory (MB)  [$M\\times$ (768$\\cdot$4$+$8) B]")
    axA.set_ylabel("Hit@5 (\\%)")
    axA.set_title("(A) Retrieval vs. memory budget")
    axA.grid(ls=":", alpha=0.4)
    axA.legend(fontsize=8, loc="lower right")
    # secondary M ticks
    secax = axA.secondary_xaxis(
        "top",
        functions=(lambda mib: mib * 1e6 / LOGICAL_BYTES_PER_EXEMPLAR,
                   lambda m: m * LOGICAL_BYTES_PER_EXEMPLAR / 1e6))
    secax.set_xlabel("exemplar budget $M$", fontsize=9)

    # ---- Panel B: rare/common deltas at M=128, paired bootstrap -------
    probe = data.get(args.probe, {}).get(PRIMARY_BUDGET, {})
    base = data.get(args.baseline, {}).get(PRIMARY_BUDGET, {})
    common_sessions = sorted(set(probe) & set(base))
    strata = [("rare_place_hit", "rare-place"), ("common_place_hit", "common-place")]
    labels, means, los, his = [], [], [], []
    for key, nice in strata:
        diffs = [probe[s][key] - base[s][key] for s in common_sessions
                 if probe[s].get(key) is not None and base[s].get(key) is not None]
        if not diffs:
            continue
        m, lo, hi = paired_bootstrap(diffs, args.resamples, args.ci, args.seed)
        labels.append(f"{nice}\n(n={len(diffs)})")
        means.append(m * 100); los.append((m - lo) * 100); his.append((hi - m) * 100)
    ypos = np.arange(len(labels))
    colors = ["#c44e52" if m > 0 else "#4c72b0" for m in means]
    axB.barh(ypos, means, xerr=[los, his], color=colors, alpha=0.85,
             error_kw=dict(ecolor="0.2", capsize=4, lw=1.2))
    axB.axvline(0, color="0.3", lw=1.0)
    axB.set_yticks(ypos)
    axB.set_yticklabels(labels)
    axB.set_xlabel("Hit@5 $\\Delta$ (pp):  spatial priority $-$ global reservoir")
    axB.set_title(f"(B) Allocation shifts strata ($M{{=}}128$, $r{{=}}12$)")
    axB.grid(axis="x", ls=":", alpha=0.4)
    for i, m in enumerate(means):
        axB.text(m + (0.3 if m >= 0 else -0.3), ypos[i],
                 f"{m:+.1f}", va="center",
                 ha="left" if m >= 0 else "right", fontsize=8)

    fig.suptitle("Fixed global exemplar budget: spatial allocation shifts which "
                 "places are remembered, not aggregate retrieval", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{args.stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {OUT_DIR}/{args.stem}.{{svg,pdf}}")
    # echo numbers
    print("# Panel A Hit@5 (%) by method x M:")
    for method, *_ in PANEL_A:
        row = []
        for b in sorted(x for x in data.get(method, {}) if x is not None):
            h = macro(data[method][b], "hit_at_k")
            if h is not None:
                row.append(f"M{b}={h*100:.1f}")
        print(f"#   {method:20s} {'  '.join(row)}")
    print(f"# Panel B ({args.probe}-{args.baseline}, M=128, n={len(common_sessions)}):")
    for (key, nice) in strata:
        diffs = [probe[s][key] - base[s][key] for s in common_sessions
                 if probe[s].get(key) is not None and base[s].get(key) is not None]
        if diffs:
            m, lo, hi = paired_bootstrap(diffs, args.resamples, args.ci, args.seed)
            print(f"#   {nice:12s} {m*100:+.1f}pp [{lo*100:+.1f}, {hi*100:+.1f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
