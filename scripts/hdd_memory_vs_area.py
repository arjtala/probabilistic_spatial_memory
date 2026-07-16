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

bytes = exemplars * dim * bytes_per_dim + a per-cell HLL ring for PSM. The ring
is C sketches x 2^p bytes, charged to EVERY visited cell -- at the paper's C=60,
p=10 design point that is 60 KiB/cell, which is NOT negligible on a corpus with
tens of thousands of mostly single-visit cells. This script reports the exemplar
term and the ring term SEPARATELY (the exemplar term is the part the reservoir
cap makes sublinear; the ring is a fixed per-cell tax that a global memory budget
would remove), plus the ratio and crossover for several fps.

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
# Per-cell HLL ring state = C sketches x 2^p bytes each. The paper's temporal
# envelope design point is C=60 windows (30 s cadence -> 30 min) at p=10 -> 1 KiB
# per sketch, so the ring is 60 KiB/cell, NOT a small constant. (An earlier draft
# hard-coded 2 KiB, which made the ring look negligible beside the reservoir and
# understated PSM's real footprint by ~58 KiB/cell; on this corpus's 23,298 cells
# that is ~1.4 GB. The ring is charged to EVERY visited cell, including the many
# single-visit ones, which is exactly why per-cell allocation is wasteful.)
HLL_CAPACITY = 60         # C: number of HLL sketches in the ring buffer
HLL_PRECISION = 10        # p: 2^p registers/bytes per sketch (1 KiB at p=10)
HLL_BYTES = HLL_CAPACITY * (2 ** HLL_PRECISION)   # 60 KiB/cell at C=60, p=10
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
    # Per-drive teleport-outlier reject (great-circle > 80 km from the median),
    # matching hdd_revisit_density; otherwise a bad fix spawns a spurious cell.
    med_lat, med_lng = float(np.median(lat)), float(np.median(lng))
    dlat = np.radians(lat - med_lat)
    dlng = np.radians(lng - med_lng)
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(np.radians(lat)) * np.cos(np.radians(med_lat)) * np.sin(dlng / 2) ** 2)
    near = 2 * 6371.0 * np.arcsin(np.sqrt(a)) <= 80.0
    ts, lat, lng = ts[near], lat[near], lng[near]
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
    ap.add_argument("--hll-capacity", type=int, default=HLL_CAPACITY,
                    help="C: HLL sketches in the ring (default %(default)s)")
    ap.add_argument("--hll-precision", type=int, default=HLL_PRECISION,
                    help="p: 2^p bytes per sketch (default %(default)s)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    hll_bytes = args.hll_capacity * (2 ** args.hll_precision)

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

    # Cumulative curves over chronological drives (per-tick PSM/bank GB so the
    # combined figure can be drawn from the JSON alone, no raw-GPS pass).
    cell_dwell: dict[str, float] = defaultdict(float)
    cum_secs = 0.0
    cum_hours, cum_cells_l = [], []
    _CURVE_FPS = (1.0, 15.0, 30.0)
    curve_psm = {f: [] for f in _CURVE_FPS}
    curve_bank = {f: [] for f in _CURVE_FPS}
    for _, dur, dwell in recs:
        for c, d in dwell.items():
            cell_dwell[c] += d
        cum_secs += dur
        cum_hours.append(cum_secs / 3600.0)
        cum_cells_l.append(len(cell_dwell))
        arr_t = np.fromiter(cell_dwell.values(), dtype=np.float64)
        for f in _CURVE_FPS:
            cf = arr_t * f
            curve_psm[f].append(float((np.minimum(cf, R).sum() * per_ex_bytes
                                       + arr_t.size * hll_bytes) / 1e9))
            curve_bank[f].append(float(cf.sum() * per_ex_bytes / 1e9))

    # PSM/bank exemplar counts are fps-dependent, so compute per fps below;
    # here record the final per-cell dwell for the exemplar sums.
    total_secs = cum_secs
    total_cells = len(cell_dwell)
    dwell_arr = np.fromiter(cell_dwell.values(), dtype=np.float64)

    hll_gb = total_cells * hll_bytes / 1e9   # ring state, charged to every cell
    fps_report = {}
    for fps in FPS_GRID:
        cell_frames = dwell_arr * fps
        psm_ex = float(np.minimum(cell_frames, R).sum())
        bank_ex = float(cell_frames.sum())            # == total_secs * fps
        psm_exemplar_gb = psm_ex * per_ex_bytes / 1e9  # exemplars only (HLL as
                                                       # optional metadata)
        psm_gb = psm_exemplar_gb + hll_gb              # full per-cell footprint
        bank_gb = bank_ex * per_ex_bytes / 1e9
        fps_report[str(fps)] = {
            "psm_exemplars": psm_ex,
            "bank_exemplars": bank_ex,
            "psm_exemplar_gb": psm_exemplar_gb,
            "psm_hll_gb": hll_gb,
            "psm_gb": psm_gb,
            "bank_gb": bank_gb,
            "bank_over_psm": bank_gb / psm_gb if psm_gb else 0.0,
            "bank_over_psm_exemplar_only": bank_gb / psm_exemplar_gb
            if psm_exemplar_gb else 0.0,
            "psm_saving_pct": 100 * (1 - psm_gb / bank_gb) if bank_gb else 0.0,
        }

    # ---- report --------------------------------------------------------
    print(f"# F-HDD-1 memory-vs-area  ({len(recs)} drives, {total_secs/3600:.1f} h, "
          f"{total_cells} distinct r{DECISION_RES} cells)")
    print(f"# model: PSM = sum_cells min(frames_in_cell, R={R}); "
          f"bank = all frames; dim={args.dim}x{BYTES_PER_DIM}B "
          f"({per_ex_bytes} B/exemplar)")
    print(f"# HLL ring: C={args.hll_capacity} x 2^{args.hll_precision} B = "
          f"{hll_bytes} B/cell x {total_cells} cells = {hll_gb:.2f} GB "
          f"(charged to EVERY visited cell, most of them single-visit)")
    print()
    hdr = (f"{'fps':>5s} {'PSM_ex_GB':>10s} {'+HLL_GB':>8s} {'PSM_GB':>8s} "
           f"{'bank_GB':>8s} {'bank/PSM':>9s} {'bank/ex-only':>13s}")
    print(hdr); print("-" * len(hdr))
    for fps in FPS_GRID:
        r = fps_report[str(fps)]
        print(f"{fps:>5.0f} {r['psm_exemplar_gb']:>10.2f} {r['psm_hll_gb']:>8.2f} "
              f"{r['psm_gb']:>8.2f} {r['bank_gb']:>8.2f} "
              f"{r['bank_over_psm']:>8.1f}x {r['bank_over_psm_exemplar_only']:>12.1f}x")
    half_cells = int(np.interp(0.5 * total_secs / 3600, cum_hours, cum_cells_l))
    print(f"#")
    print(f"# area growth: {half_cells}/{total_cells} distinct cells "
          f"({100*half_cells/total_cells:.0f}%) seen by the halfway mark. HONEST "
          f"READING: area keeps growing over the 8 months, so PSM's total state "
          f"is NOT bounded -- it grows with the cell table. Counting the full "
          f"C={args.hll_capacity} HLL ring at every cell, PSM can EXCEED the dense "
          f"bank at low fps (the ring is charged even to single-visit cells). The "
          f"reservoir cap only makes the EXEMPLAR term sublinear in frame count "
          f"(bank/ex-only column); the per-cell ring is the cost a global memory "
          f"budget would remove. Report exemplar and ring bytes separately.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_drives": len(recs),
        "total_hours": total_secs / 3600.0,
        "total_cells_r%d" % DECISION_RES: total_cells,
        "model": {"dim": args.dim, "bytes_per_dim": BYTES_PER_DIM,
                  "reservoir": R, "per_exemplar_bytes": per_ex_bytes,
                  "hll_capacity": args.hll_capacity,
                  "hll_precision": args.hll_precision,
                  "hll_bytes_per_cell": hll_bytes,
                  "psm_rule": "sum_cells min(frames_in_cell, R)"},
        "fps": fps_report,
        "cum_hours": cum_hours,
        "cum_cells": cum_cells_l,
        "curve": {"hours": cum_hours,
                  "psm_gb": {str(f): curve_psm[f] for f in _CURVE_FPS},
                  "bank_gb": {str(f): curve_bank[f] for f in _CURVE_FPS}},
    }, indent=2))
    print(f"\n# wrote {args.out}")

    if args.plot:
        _plot(cum_hours, recs, dwell_arr, total_secs, R, per_ex_bytes,
              total_cells, hll_bytes)
    return 0


def _plot(cum_hours, recs, dwell_arr, total_secs, R, per_ex_bytes,
          total_cells, hll_bytes) -> None:
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
                          + len(arr) * hll_bytes) / 1e9)
            bank_l.append(cf.sum() * per_ex_bytes / 1e9)
        bank1.append(cum * 1.0 * per_ex_bytes / 1e9)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hours, bank30, "-", color="C3", label="dense bank @ 30 fps")
    ax.plot(hours, psm30, "-", color="C0", lw=2.2,
            label=f"PSM @ 30 fps (R={R}+HLL ring)")
    ax.plot(hours, bank15, "--", color="C3", alpha=0.7, label="dense bank @ 15 fps")
    ax.plot(hours, psm15, "--", color="C0", lw=2.0, alpha=0.7, label="PSM @ 15 fps")
    ax.set_xlabel("cumulative driving (hours)")
    ax.set_ylabel("embedding memory (GB)")
    ax.set_title("HDD: PSM per-cell reservoir+HLL ring vs. dense embedding bank")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hdd_memory_vs_area.svg")
    fig.savefig(out_dir / "hdd_memory_vs_area.pdf")
    plt.close(fig)
    print("# wrote journal/figures/hdd_memory_vs_area.svg")


if __name__ == "__main__":
    sys.exit(main())
