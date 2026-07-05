"""F-HDD-1: memory-vs-area -- PSM bounded state vs dense embedding bank.

The OpenSUN3D reframe's headline claim is "state scales by area explored, not
by sequence length." HDD is the corpus that can show it: 104 h of driving that
re-covers the same arterials (47% of traversal events land in >=2-day cells,
see hdd_revisit_density.py).

The honest memory model. Both systems ingest the SAME frame stream at
``sample_fps``; the frame count in a cell is proportional to dwell time there.
  - Dense bank keeps EVERY frame's embedding:
      exemplars = total_frames = total_seconds * fps
  - PSM keeps a bounded per-cell reservoir: at most R exemplars per cell, so
      exemplars = sum_cells min(frames_in_cell, R)
    Singly-driven cells contribute their few frames; revisited/dwelled cells
    saturate at R. PSM is therefore <= the bank ALWAYS, and the gap is exactly
    the frames the reservoir cap discards on re-seen ground. (An earlier draft
    modeled PSM as n_cells * R -- assuming every cell full -- which wrongly
    inflated PSM above the bank at low fps. The min() is the real reservoir.)

bytes = exemplars * dim * bytes_per_dim (+ a small per-cell HLL ring for PSM).
The bank grows linearly with fps x time; PSM's growth bends down as area
re-covers. This script reports the ratio and crossover for several fps.

Reuses the drive enumeration from hdd_revisit_density.py, but bins ALL RTK
points (no speed gate: the bank stores frames during stops too), decimated for
speed, at r10.

Run (needs h3, numpy; matplotlib for --plot):
  /opt/conda/bin/python scripts/hdd_memory_vs_area.py --plot
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3  # noqa: E402
from hdd_revisit_density import DEFAULT_ROOT, find_drives  # noqa: E402

DECISION_RES = 10
DEFAULT_OUT = Path("captures/hdd/memory_vs_area.json")

# Memory-model defaults (CLIP-L operating point from the paper).
DIM = 768                 # CLIP-L embedding dim
BYTES_PER_DIM = 4         # float32 raw exemplars
RESERVOIR = 128           # per-cell reservoir cap (bounded-memory default)
HLL_BYTES = 2048          # per-cell HLL ring state (small beside a full reservoir)
FPS_GRID = (1.0, 5.0, 15.0, 30.0)
_DECIM_HZ = 5.0           # bin points at ~5 Hz for speed (dwell fraction preserved)


def _load_all_points(csv_dir: Path):
    """All finite RTK points (ts, lat, lng) for a drive, no speed gate.

    Range-guarded to SF Bay (drops swap/bad-fix files). Decimated to ~_DECIM_HZ
    -- we only need per-cell dwell *fractions*, which decimation preserves.
    """
    raw = np.genfromtxt(csv_dir / "rtk_pos.csv", delimiter=",", comments="#",
                        usecols=(0, 2, 3), dtype=np.float64)
    raw = np.atleast_2d(raw)
    if raw.size == 0:
        return None
    ts, lat, lng = raw[:, 0], raw[:, 1], raw[:, 2]
    ok = (np.isfinite(ts) & np.isfinite(lat) & np.isfinite(lng) &
          (lat >= 36.5) & (lat <= 38.5) & (lng >= -123.5) & (lng <= -120.5))
    ts, lat, lng = ts[ok], lat[ok], lng[ok]
    if ts.size < 2:
        return None
    order = np.argsort(ts)
    ts, lat, lng = ts[order], lat[order], lng[order]
    span = float(ts[-1] - ts[0])
    if span <= 0:
        return None
    stride = max(1, int(round((ts.size / span) / _DECIM_HZ)))
    return ts[::stride], lat[::stride], lng[::stride], span


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--reservoir", type=int, default=RESERVOIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    drives = find_drives(args.root)
    if not drives:
        print(f"ERR: no drives under {args.root}", file=sys.stderr)
        return 1

    # Per drive (chronological): duration + per-cell dwell seconds at r10.
    recs = []  # (start_ts, duration_s, {cell: dwell_s})
    for _, _, csv_dir in drives:
        loaded = _load_all_points(csv_dir)
        if loaded is None:
            continue
        ts, lat, lng, span = loaded
        n = ts.size
        per_pt = span / n  # seconds each decimated point represents
        dwell: dict[str, float] = defaultdict(float)
        for a, o in zip(lat, lng):
            dwell[h3.latlng_to_cell(float(a), float(o), DECISION_RES)] += per_pt
        recs.append((float(ts[0]), span, dwell))
    recs.sort(key=lambda r: r[0])

    R = args.reservoir
    per_ex_bytes = args.dim * BYTES_PER_DIM

    # Cumulative curves over chronological drives.
    cell_dwell: dict[str, float] = defaultdict(float)
    cum_secs = 0.0
    cum_hours, cum_cells_l = [], []
    psm_ex_l, bank_ex_l = [], []
    for _, dur, dwell in recs:
        for c, d in dwell.items():
            cell_dwell[c] += d
        cum_secs += dur
        cum_hours.append(cum_secs / 3600.0)
        cum_cells_l.append(len(cell_dwell))

    # PSM/bank exemplar counts are fps-dependent, so compute per fps below;
    # here record the final per-cell dwell for the exemplar sums.
    total_secs = cum_secs
    total_cells = len(cell_dwell)
    dwell_arr = np.fromiter(cell_dwell.values(), dtype=np.float64)

    fps_report = {}
    for fps in FPS_GRID:
        cell_frames = dwell_arr * fps
        psm_ex = float(np.minimum(cell_frames, R).sum())
        bank_ex = float(cell_frames.sum())            # == total_secs * fps
        psm_gb = (psm_ex * per_ex_bytes + total_cells * HLL_BYTES) / 1e9
        bank_gb = bank_ex * per_ex_bytes / 1e9
        fps_report[str(fps)] = {
            "psm_exemplars": psm_ex,
            "bank_exemplars": bank_ex,
            "psm_gb": psm_gb,
            "bank_gb": bank_gb,
            "bank_over_psm": bank_gb / psm_gb if psm_gb else 0.0,
            "psm_saving_pct": 100 * (1 - psm_ex / bank_ex) if bank_ex else 0.0,
        }

    # ---- report --------------------------------------------------------
    print(f"# F-HDD-1 memory-vs-area  ({len(recs)} drives, {total_secs/3600:.1f} h, "
          f"{total_cells} distinct r{DECISION_RES} cells)")
    print(f"# model: PSM = sum_cells min(frames_in_cell, R={R}); "
          f"bank = all frames; dim={args.dim}x{BYTES_PER_DIM}B "
          f"({per_ex_bytes} B/exemplar)")
    print()
    hdr = f"{'fps':>5s} {'PSM_GB':>8s} {'bank_GB':>8s} {'bank/PSM':>9s} {'PSM saving':>11s}"
    print(hdr); print("-" * len(hdr))
    for fps in FPS_GRID:
        r = fps_report[str(fps)]
        print(f"{fps:>5.0f} {r['psm_gb']:>8.2f} {r['bank_gb']:>8.2f} "
              f"{r['bank_over_psm']:>8.1f}x {r['psm_saving_pct']:>10.1f}%")
    half_cells = int(np.interp(0.5 * total_secs / 3600, cum_hours, cum_cells_l))
    print(f"#")
    print(f"# area growth: {half_cells}/{total_cells} distinct cells "
          f"({100*half_cells/total_cells:.0f}%) seen by the halfway mark. NOTE: "
          f"on this corpus area keeps growing (the fleet explores new routes "
          f"over 8 months), so the saving is NOT an area plateau -- it comes "
          f"from the per-cell reservoir capping redundant frames (dwell + "
          f"revisits) at R, i.e. PSM memory grows SUBLINEARLY in the frame "
          f"count while the bank is linear.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_drives": len(recs),
        "total_hours": total_secs / 3600.0,
        "total_cells_r%d" % DECISION_RES: total_cells,
        "model": {"dim": args.dim, "bytes_per_dim": BYTES_PER_DIM,
                  "reservoir": R, "per_exemplar_bytes": per_ex_bytes,
                  "hll_bytes_per_cell": HLL_BYTES,
                  "psm_rule": "sum_cells min(frames_in_cell, R)"},
        "fps": fps_report,
        "cum_hours": cum_hours,
        "cum_cells": cum_cells_l,
    }, indent=2))
    print(f"\n# wrote {args.out}")

    if args.plot:
        _plot(cum_hours, recs, dwell_arr, total_secs, R, per_ex_bytes,
              total_cells)
    return 0


def _plot(cum_hours, recs, dwell_arr, total_secs, R, per_ex_bytes,
          total_cells) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib unavailable; skipping --plot", file=sys.stderr)
        return
    out_dir = Path("journal/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cumulative exemplar curves vs hours, at a realistic streaming fps.
    # Recompute per-cell cumulative dwell across drives for the PSM bend.
    from collections import defaultdict as _dd
    cell_dwell = _dd(float)
    hours, psm15, psm30, bank15, bank30, bank1 = [], [], [], [], [], []
    cum = 0.0
    for _, dur, dwell in recs:
        for c, d in dwell.items():
            cell_dwell[c] += d
        cum += dur
        arr = np.fromiter(cell_dwell.values(), dtype=np.float64)
        hours.append(cum / 3600.0)
        for fps, psm_l, bank_l in ((15.0, psm15, bank15), (30.0, psm30, bank30)):
            cf = arr * fps
            psm_l.append((np.minimum(cf, R).sum() * per_ex_bytes
                          + len(arr) * 2048) / 1e9)
            bank_l.append(cf.sum() * per_ex_bytes / 1e9)
        bank1.append(cum * 1.0 * per_ex_bytes / 1e9)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hours, bank30, "-", color="C3", label="dense bank @ 30 fps")
    ax.plot(hours, psm30, "-", color="C0", lw=2.2, label="PSM @ 30 fps (bounded)")
    ax.plot(hours, bank15, "--", color="C3", alpha=0.7, label="dense bank @ 15 fps")
    ax.plot(hours, psm15, "--", color="C0", lw=2.0, alpha=0.7, label="PSM @ 15 fps")
    ax.set_xlabel("cumulative driving (hours)")
    ax.set_ylabel("embedding memory (GB)")
    ax.set_title("HDD: PSM area-bounded reservoir vs. dense embedding bank")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hdd_memory_vs_area.svg")
    plt.close(fig)
    print("# wrote journal/figures/hdd_memory_vs_area.svg")


if __name__ == "__main__":
    sys.exit(main())



if __name__ == "__main__":
    sys.exit(main())
