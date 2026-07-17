r"""Fig. 5 (HDD): analytical modeled-memory panel.

Replaces the earlier 3-panel HDD figure (whose cardinality/AUC panels are
withdrawn as PSM-engine results, see journal/review_fixes_2026-07-16.md). This
single panel plots \emph{modeled logical state} from the RTK trajectory (no
embeddings), reading captures/hdd/memory_vs_area.json:

  - Dense embedding bank   : all frames x d x 4 B.
  - PSM exemplar state     : sum_cells min(frames, R) x d x 4 B.
  - PSM full state         : exemplar + a C-slot HLL ring (2^p B/slot) per cell.

The HLL ring is charged to EVERY visited cell (most single-visit), so full PSM
can EXCEED the dense bank at low fps -- the panel is drawn to make that crossover
explicit. This is an analytical model of logical state, NOT measured RSS.

The cached JSON may store per-cell exemplar/bank counts computed under an older
HLL constant; we recompute the HLL term here from --hll-capacity/--hll-precision
so the figure is correct regardless of when the JSON was produced.

Run (needs numpy, matplotlib):
  /home/arjangt/.conda/envs/psm/bin/python scripts/plot_f5_hdd_memory.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_JSON = Path("captures/hdd/memory_vs_area.json")
OUT_DIR = Path("journal/figures")
FPS_ORDER = (1.0, 5.0, 15.0, 30.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--hll-capacity", type=int, default=60,
                    help="C: HLL sketches per cell (paper design point 60)")
    ap.add_argument("--hll-precision", type=int, default=10,
                    help="p: 2^p bytes per sketch (default 10 -> 1 KiB)")
    ap.add_argument("--stem", default="f5_hdd_memory")
    args = ap.parse_args()

    doc = json.loads(args.json.read_text())
    model = doc.get("model", {})
    per_ex_bytes = int(model.get("per_exemplar_bytes", 768 * 4))
    n_cells = int(doc["total_cells_r%d" % 10] if "total_cells_r10" in doc
                  else next(v for k, v in doc.items() if k.startswith("total_cells")))
    hll_bytes = args.hll_capacity * (2 ** args.hll_precision)
    hll_gb = n_cells * hll_bytes / 1e9
    total_hours = doc.get("total_hours", 0.0)

    fps_list, bank, psm_ex, psm_full = [], [], [], []
    for fps in FPS_ORDER:
        rec = doc["fps"].get(str(fps))
        if rec is None:
            continue
        fps_list.append(fps)
        ex_gb = float(rec["psm_exemplars"]) * per_ex_bytes / 1e9
        bank.append(float(rec["bank_exemplars"]) * per_ex_bytes / 1e9)
        psm_ex.append(ex_gb)
        psm_full.append(ex_gb + hll_gb)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(fps_list))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    b1 = ax.bar(x - w, bank, w, label="dense embedding bank",
                color="#c44e52")
    b2 = ax.bar(x, psm_ex, w, label="PSM exemplar state ($R{=}128$)",
                color="#4c72b0")
    b3 = ax.bar(x + w, psm_full, w,
                label=f"PSM full (exemplars $+$ HLL ring, $C{{=}}{args.hll_capacity}$)",
                color="#2a4a7a", hatch="//", edgecolor="white", linewidth=0.4)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in fps_list])
    ax.set_xlabel("ingest rate (fps)")
    ax.set_ylabel("modeled embedding memory (GB, log scale)")
    ax.set_title("HDD modeled memory (analytical logical state, not measured RSS)")

    # Make the 1-fps crossover explicit: PSM full > dense bank.
    if fps_list and fps_list[0] == 1.0 and psm_full[0] > bank[0]:
        ax.annotate(
            f"full PSM {psm_full[0]:.2f} GB $>$ bank {bank[0]:.2f} GB at 1 fps",
            xy=(x[0] + w, psm_full[0]), xytext=(x[0] - 0.38, 11.4),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    cap = (f"HRI Driving Dataset, {total_hours:.1f} h, {n_cells:,} distinct r10 cells; "
           f"HLL ring {args.hll_capacity}$\\times2^{{{args.hll_precision}}}$B/cell "
           f"= {hll_gb:.2f} GB total.")
    fig.text(0.01, -0.02, cap, fontsize=7, ha="left")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(OUT_DIR / f"{args.stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"# wrote {OUT_DIR}/{args.stem}.{{svg,pdf}}")
    print(f"# fps      bank_GB   PSM_ex_GB  PSM_full_GB  bank/full")
    for i, f in enumerate(fps_list):
        print(f"#  {int(f):>3d}   {bank[i]:>8.2f}  {psm_ex[i]:>9.2f}  "
              f"{psm_full[i]:>10.2f}  {bank[i]/psm_full[i]:>6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
