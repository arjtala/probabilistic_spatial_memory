"""Combined 3-panel HDD figure for the paper's §5 (page-budget trim).

Reads the three HDD capture JSONs and renders one figure with a panel per
result, so the §5 HDD subsection can reference a single figure instead of three
full-width ones (saves ~0.6-0.7 LNCS pp). Pure JSON -> no cluster/GPS/GPU
dependency to redraw; the standalone figures are left untouched (this is
additive until the §5 layout swap is committed).

Panels:
  A  memory-vs-area: PSM bounded per-cell reservoir vs dense bank over hours
     (F-HDD-1)  -- "state is O(area), not O(time)"
  B  HLL cardinality on the top revisited cell: cumulative accrual (left axis)
     + ring-buffer windowed estimate decaying between visits (right axis)
     (F-HDD-2)  -- "accrues + decays across revisits in 1 KiB"
  C  cross-session retrieval AUC: shuffled null / cross-session / same-drive UB
     (F-HDD-3)  -- "the persistent memory is retrievable across sessions"

Run:
  /opt/conda/bin/python scripts/plot_hdd_combined.py --layout vertical --png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CAP = Path("captures/hdd")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--cap-dir", type=Path, default=CAP)
    ap.add_argument("--layout", choices=["vertical", "horizontal"],
                    default="vertical",
                    help="vertical (3 rows, single-column LNCS) or horizontal strip")
    ap.add_argument("--fps", type=str, default="30.0",
                    help="which memory-curve fps to draw in panel A")
    ap.add_argument("--out", type=Path,
                    default=Path("journal/figures/hdd_combined"))
    ap.add_argument("--png", action="store_true", help="also emit .png (for inspection)")
    args = ap.parse_args()

    mem = json.loads((args.cap_dir / "memory_vs_area.json").read_text())
    hll = json.loads((args.cap_dir / "hll_cardinality.json").read_text())
    xs = json.loads((args.cap_dir / "cross_session_retrieval.json").read_text())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERR: matplotlib required", file=sys.stderr)
        return 1

    if args.layout == "vertical":
        # Wider-and-shorter so it displays ~4.8x2.5in at \linewidth -- small
        # enough (<50% text height) that a [t]/[tb] float places in-section
        # rather than drifting to the end of the document.
        fig, (a, b, c) = plt.subplots(3, 1, figsize=(5.4, 4.0))
    else:
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.3))

    # --- Panel A: memory-vs-area -------------------------------------------
    cur = mem["curve"]
    hours = cur["hours"]
    fps = args.fps if args.fps in cur["psm_gb"] else next(iter(cur["psm_gb"]))
    a.plot(hours, cur["bank_gb"][fps], color="C3", lw=1.8,
           label=f"dense bank ({int(float(fps))} fps)")
    a.plot(hours, cur["psm_gb"][fps], color="C0", lw=2.0,
           label="PSM (area-bounded)")
    ratio = mem["fps"][fps]["bank_over_psm"]
    a.set_xlabel("cumulative driving (hours)")
    a.set_ylabel("embedding memory (GB)")
    a.set_title(f"(A) State scales with area, not time ({ratio:.1f}$\\times$)", fontsize=9)
    a.legend(fontsize=7, loc="upper left")

    # --- Panel B: HLL accrual (left) + windowed decay (right) --------------
    cell = hll["cells"][0]
    pts = cell["points"]
    t0 = pts[0]["ts"]
    days = [(p["ts"] - t0) / 86400.0 for p in pts]
    b.plot(days, [p["accrual"] for p in pts], color="C0", lw=2.0,
           label="cumulative")
    b.set_xlabel("days since first visit")
    b.set_ylabel("HLL cardinality (accrual)", color="C0")
    b.tick_params(axis="y", labelcolor="C0")
    b2 = b.twinx()
    b2.plot(days, [p["windowed"] for p in pts], color="C2", lw=1.0, alpha=0.8,
            label="windowed")
    b2.set_ylabel("windowed (decay)", color="C2")
    b2.tick_params(axis="y", labelcolor="C2")
    b.set_title(f"(B) Cardinality accrues + decays ({hll['bytes_per_cell']//1024} KiB/cell)",
                fontsize=9)

    # --- Panel C: cross-session retrieval AUC ------------------------------
    names = ["shuffled\n(null)", "cross-\nsession", "same-drive\n(UB)"]
    vals = [xs["shuffled_cell_auc_control"]["mean"],
            xs["cross_session_auc"]["mean"],
            xs["same_drive_auc_upper_bound"]["mean"]]
    c.bar(names, vals, color=["0.6", "C0", "C2"])
    c.axhline(0.5, ls="--", color="k", lw=0.8)
    c.set_ylim(0.4, 1.0)
    c.set_ylabel("retrieval AUC")
    c.set_title("(C) Retrievable across sessions", fontsize=9)
    for i, v in enumerate(vals):
        c.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out}.svg", bbox_inches="tight", pad_inches=0.02)
    if args.png:
        fig.savefig(f"{out}.png", dpi=130, bbox_inches="tight", pad_inches=0.02)
    print(f"# wrote {out}.pdf/.svg" + (" /.png" if args.png else "")
          + f"  (layout={args.layout})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
