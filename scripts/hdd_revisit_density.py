"""Revisit-density probe for the Honda HDD driving corpus (OpenSUN3D reframe).

The OpenSUN3D / SpatialMem reframe leans on two claims the frozen 14-session
corpus does not demonstrate: multi-session persistent memory and unique-event
cardinality across revisits (see docs/HDD.md). HDD is 104 h of SF-Bay driving
that almost certainly re-drives arterials -- but its revisit structure is
undocumented, so we measure it here (catch #1). This is the Option-A de-risk:
GPS-only, no encoder, no annotation.

The pre-registered decision rule (docs/HDD.md, committed before this ran):
  primary metric = coverage-weighted fraction of driving over >=2-distinct-DAY
  cells at r10; go/no-go bar = 30%. Day-count (not drive-count) is the real
  multi-session signal; r10 is RTK-noise-robust; coverage is weighted by
  drive-visits (not raw GPS point count, which stationary idling inflates).

Outputs:
  - a per-resolution table to stdout (--summary adds histograms + gap dist);
  - captures/hdd/revisit_density.json: per-res stats, the r10 inter-visit
    temporal-gap distribution, and the top-N most-revisited cells with their
    drive-ID + day lists (seeds step 3 HLL accrual + Option-C annotation);
  - --plot: r10 cell map colored by distinct-day count + a histogram SVG.

Run (needs h3, e.g. the extraction viz extra or /opt/conda):
  /opt/conda/bin/python scripts/hdd_revisit_density.py --summary

Data gotcha: rtk_pos.csv header reads ",lng,lat" but the values are actually
lat,lng (col3 = 37.x Palo Alto latitude, col4 = -122.x longitude).
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import h3
except ImportError:
    print("ERR: h3 not importable. Install `h3>=4.0` (extraction viz extra) "
          "or run with /opt/conda/bin/python.", file=sys.stderr)
    raise

DEFAULT_ROOT = Path(
    "$PSM_DATA_ROOT/video_retrieval/hdd/release_2019_07_08"
)
DEFAULT_OUT = Path("captures/hdd/revisit_density.json")

RESOLUTIONS = (9, 10, 11, 12)
DECISION_RES = 10          # RTK-noise-robust (66 m edge)
DECISION_BAR = 0.30        # >= 30% coverage over >=2-day cells

# H3 avg edge length (m) -- https://h3geo.org/docs/core-library/restable
_H3_EDGE_M = {9: 174.4, 10: 65.9, 11: 24.9, 12: 9.4}

# Range guard against a header lat/lng swap or garbage fix. Kept wide enough
# to include Santa Cruz (~36.97 N) -- several HDD drives cross Hwy 17 to the
# coast, so a tight 37.0 floor would wrongly drop real data. A true swap puts
# "lat" near -122 (far outside this box), so the guard still catches it.
_LAT_RANGE = (36.5, 38.5)
_LNG_RANGE = (-123.5, -120.5)

# Per-drive outlier rejection: drop fixes more than this far from the drive's
# median position. Real HDD drives stay within one metro (SF<->Santa Cruz is
# ~65 km end to end, ~35 km from a mid-drive median); isolated teleport fixes
# land hundreds of km out and otherwise create spurious singleton cells that
# dilute the revisit fractions downward.
_OUTLIER_KM = 80.0

# Cell membership only needs a few Hz: a car at 30 m/s moves 6 m per 0.2 s,
# below even the r12 edge (9.4 m), so decimating ~100 Hz RTK to this rate
# captures every cell crossing at r9-r11 (and nearly all at r12) while cutting
# the H3 call count ~20x. Slight r12 undercount is conservative for revisits.
_TARGET_HZ = 5.0

DEFAULT_SPEED_GATE = 1.0   # drop points at/below this speed (idling); units
                           # unverified (mph or m/s) but ~1 == "barely moving"
                           # either way, so the stopped/moving split is robust.


def find_drives(root: Path) -> list[tuple[str, str, Path]]:
    """Return (drive_day, drive_id, csv_dir) for every drive with GPS."""
    drives: list[tuple[str, str, Path]] = []
    for day_dir in sorted(root.glob("2017_*_ITS1")):
        if not day_dir.is_dir():
            continue
        for drive_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
            csv_dir = drive_dir / "general" / "csv"
            if (csv_dir / "rtk_pos.csv").is_file():
                drives.append((day_dir.name, drive_dir.name, csv_dir))
    return drives


def _read_numeric(path: Path, usecols: tuple[int, ...]) -> np.ndarray | None:
    """Load selected numeric columns, skipping the '#'-prefixed header."""
    if not path.is_file():
        return None
    try:
        arr = np.genfromtxt(path, delimiter=",", comments="#",
                            usecols=usecols, dtype=np.float64)
    except (OSError, ValueError):
        return None
    if arr.size == 0:
        return None
    return np.atleast_2d(arr)


def _haversine_km(lat, lng, lat0, lng0):
    """Vectorized great-circle distance (km) from each point to (lat0, lng0)."""
    r = 6371.0
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi = np.radians(lat - lat0)
    dlam = np.radians(lng - lng0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_drive(csv_dir: Path, speed_gate: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Return (ts, lat, lng, n_raw) for moving GPS points, or None on bad data.

    rtk_pos.csv columns: 0=unix_ts, 2=lat, 3=lng (header mislabels 2/3 as
    lng/lat -- values are lat,lng). Points are speed-gated via vel.csv.
    """
    pos = _read_numeric(csv_dir / "rtk_pos.csv", (0, 2, 3))
    if pos is None:
        return None
    ts, lat, lng = pos[:, 0], pos[:, 1], pos[:, 2]
    n_raw = ts.shape[0]

    # Range guard: catch a file where the swap does not hold (or garbage).
    finite = np.isfinite(lat) & np.isfinite(lng) & np.isfinite(ts)
    ts, lat, lng = ts[finite], lat[finite], lng[finite]
    if ts.shape[0] == 0:
        return None
    in_box = ((lat >= _LAT_RANGE[0]) & (lat <= _LAT_RANGE[1]) &
              (lng >= _LNG_RANGE[0]) & (lng <= _LNG_RANGE[1]))
    if in_box.mean() < 0.5:
        print(f"WARN: {csv_dir} -- {(~in_box).sum()}/{in_box.size} points "
              f"outside SF Bay box (lat~{lat[0]:.3f}, lng~{lng[0]:.3f}); "
              f"skipping (possible lat/lng swap or bad fix).", file=sys.stderr)
        return None
    ts, lat, lng = ts[in_box], lat[in_box], lng[in_box]

    # Per-drive outlier rejection: drop teleport fixes far from the median.
    med_lat, med_lng = float(np.median(lat)), float(np.median(lng))
    near = _haversine_km(lat, lng, med_lat, med_lng) <= _OUTLIER_KM
    ts, lat, lng = ts[near], lat[near], lng[near]
    if ts.shape[0] == 0:
        return None

    # Stationary gate via vel.csv (col 0=ts, 2=speed). Missing -> keep all.
    vel = _read_numeric(csv_dir / "vel.csv", (0, 2))
    if vel is not None:
        vts, vspeed = vel[:, 0], vel[:, 1]
        order = np.argsort(vts)
        vts, vspeed = vts[order], vspeed[order]
        # unique timestamps for a valid interpolation grid
        uniq = np.concatenate(([True], np.diff(vts) > 0))
        vts, vspeed = vts[uniq], vspeed[uniq]
        speed_at = np.interp(ts, vts, vspeed)
        moving = speed_at > speed_gate
        ts, lat, lng = ts[moving], lat[moving], lng[moving]
    if ts.shape[0] == 0:
        return None

    # Decimate to ~_TARGET_HZ for cell membership (see _TARGET_HZ note).
    span = float(ts[-1] - ts[0])
    if span > 0:
        rate = ts.shape[0] / span
        stride = max(1, int(round(rate / _TARGET_HZ)))
        if stride > 1:
            ts, lat, lng = ts[::stride], lat[::stride], lng[::stride]
    return ts, lat, lng, n_raw


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT,
                    help=f"HDD release root (default {DEFAULT_ROOT})")
    ap.add_argument("--speed-gate", type=float, default=DEFAULT_SPEED_GATE,
                    help="drop GPS points with speed <= this (default 1.0)")
    ap.add_argument("--top-n", type=int, default=50,
                    help="most-revisited cells to dump in JSON (default 50)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"JSON output path (default {DEFAULT_OUT})")
    ap.add_argument("--summary", action="store_true",
                    help="also print histograms, gap distribution, verdict")
    ap.add_argument("--plot", action="store_true",
                    help="write r10 cell map + histogram SVGs to journal/figures")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERR: {root} not a directory", file=sys.stderr)
        return 1

    drives = find_drives(root)
    if not drives:
        print(f"ERR: no drives with rtk_pos.csv under {root}", file=sys.stderr)
        return 1

    # Per resolution: cell -> {drive_id: earliest_ts_in_cell}, cell -> day set.
    cell_drive_ts: dict[int, dict[str, dict[str, float]]] = {
        r: defaultdict(dict) for r in RESOLUTIONS
    }
    cell_days: dict[int, dict[str, set[str]]] = {
        r: defaultdict(set) for r in RESOLUTIONS
    }
    # Coverage denominators: total (drive, cell) visits and gated points.
    total_visits = {r: 0 for r in RESOLUTIONS}
    total_gated_points = 0

    n_ok = 0
    n_skip = 0
    bbox = [90.0, -90.0, 180.0, -180.0]  # min_lat, max_lat, min_lng, max_lng
    max_speed_seen = 0.0
    first_ts: dict[int, dict] = {r: {} for r in RESOLUTIONS}

    for drive_day, drive_id, csv_dir in drives:
        loaded = load_drive(csv_dir, args.speed_gate)
        if loaded is None:
            n_skip += 1
            continue
        ts, lat, lng, _ = loaded
        n_ok += 1
        total_gated_points += ts.shape[0]
        bbox[0] = min(bbox[0], float(lat.min()))
        bbox[1] = max(bbox[1], float(lat.max()))
        bbox[2] = min(bbox[2], float(lng.min()))
        bbox[3] = max(bbox[3], float(lng.max()))

        for r in RESOLUTIONS:
            # cell -> earliest ts this drive was in it
            first_ts[r].clear()
        finest = max(RESOLUTIONS)
        coarser = [r for r in RESOLUTIONS if r != finest]
        for la, lo, t in zip(lat, lng, ts):
            t = float(t)
            c_fine = h3.latlng_to_cell(float(la), float(lo), finest)
            cur = first_ts[finest].get(c_fine)
            if cur is None or t < cur:
                first_ts[finest][c_fine] = t
            for r in coarser:
                cell = h3.cell_to_parent(c_fine, r)
                cur = first_ts[r].get(cell)
                if cur is None or t < cur:
                    first_ts[r][cell] = t
        for r in RESOLUTIONS:
            total_visits[r] += len(first_ts[r])
            for cell, t in first_ts[r].items():
                cell_drive_ts[r][cell][drive_id] = t
                cell_days[r][cell].add(drive_day)

    if n_ok == 0:
        print("ERR: no drives yielded valid moving GPS points.", file=sys.stderr)
        return 1

    # ---- per-resolution stats -------------------------------------------
    per_res: dict[int, dict] = {}
    for r in RESOLUTIONS:
        cells = cell_drive_ts[r]
        n_cells = len(cells)
        n_2drive = sum(1 for d in cells.values() if len(d) >= 2)
        n_2day = sum(1 for c in cells if len(cell_days[r][c]) >= 2)
        # coverage-weighted (by drive-visits) over >=2-day cells
        visits_2day = sum(len(cells[c]) for c in cells
                          if len(cell_days[r][c]) >= 2)
        cov_visits = visits_2day / total_visits[r] if total_visits[r] else 0.0
        # distinct-drive-count histogram
        buckets = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, ">10": 0}
        for d in cells.values():
            n = len(d)
            key = ("1" if n == 1 else "2" if n == 2 else "3-5" if n <= 5
                   else "6-10" if n <= 10 else ">10")
            buckets[key] += 1
        per_res[r] = {
            "edge_m": _H3_EDGE_M[r],
            "n_cells": n_cells,
            "n_cells_ge2_drive": n_2drive,
            "n_cells_ge2_day": n_2day,
            "frac_cells_ge2_drive": n_2drive / n_cells if n_cells else 0.0,
            "frac_cells_ge2_day": n_2day / n_cells if n_cells else 0.0,
            "coverage_visits_ge2_day": cov_visits,
            "drive_count_hist": buckets,
        }

    # ---- inter-visit temporal-gap distribution at DECISION_RES ----------
    gap_days: list[float] = []
    for cell, dts in cell_drive_ts[DECISION_RES].items():
        if len(cell_days[DECISION_RES][cell]) < 2:
            continue
        tvals = list(dts.values())
        gap_days.append((max(tvals) - min(tvals)) / 86400.0)
    gap_days.sort()

    def _pctile(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
        return xs[idx]

    gap_stats = {
        "n_revisited_cells": len(gap_days),
        "min_days": gap_days[0] if gap_days else 0.0,
        "median_days": statistics.median(gap_days) if gap_days else 0.0,
        "p90_days": _pctile(gap_days, 0.90),
        "max_days": gap_days[-1] if gap_days else 0.0,
    }

    # ---- top-N most-revisited cells at DECISION_RES ---------------------
    ranked = sorted(
        cell_drive_ts[DECISION_RES].items(),
        key=lambda kv: (len(cell_days[DECISION_RES][kv[0]]), len(kv[1])),
        reverse=True,
    )
    top_cells = []
    for cell, dts in ranked[:args.top_n]:
        tvals = list(dts.values())
        lat_c, lng_c = h3.cell_to_latlng(cell)
        top_cells.append({
            "cell": cell,
            "lat": lat_c, "lng": lng_c,
            "n_drives": len(dts),
            "n_days": len(cell_days[DECISION_RES][cell]),
            "gap_days": (max(tvals) - min(tvals)) / 86400.0,
            "drive_ids": sorted(dts),
            "days": sorted(cell_days[DECISION_RES][cell]),
        })

    dec = per_res[DECISION_RES]["coverage_visits_ge2_day"]
    verdict = "GO (>= bar)" if dec >= DECISION_BAR else "NO-GO (< bar)"

    # ---- depot-exclusion sensitivity at DECISION_RES --------------------
    # "Your revisits are just the Honda garage" is the likely reviewer attack.
    # Exclude the top-K most-driven (depot-origin) cells from BOTH numerator
    # and denominator and recompute coverage: does cross-day re-driving still
    # dominate once the origin cells are removed?
    dcells = cell_drive_ts[DECISION_RES]
    ddays = cell_days[DECISION_RES]
    by_traffic = sorted(dcells, key=lambda c: len(dcells[c]), reverse=True)

    def _coverage_excl(exclude: set) -> float:
        num = den = 0
        for c, dd in dcells.items():
            if c in exclude:
                continue
            v = len(dd)
            den += v
            if len(ddays[c]) >= 2:
                num += v
        return num / den if den else 0.0

    depot_sens = {str(k): _coverage_excl(set(by_traffic[:k]))
                  for k in (1, 5, 10, 20)}

    # ---- stdout table ---------------------------------------------------
    print(f"# HDD revisit density -- {n_ok} drives OK, {n_skip} skipped, "
          f"speed_gate={args.speed_gate}")
    print(f"# union bbox: lat [{bbox[0]:.4f}, {bbox[1]:.4f}]  "
          f"lng [{bbox[2]:.4f}, {bbox[3]:.4f}]  "
          f"(~{(bbox[1]-bbox[0])*111.32:.1f} km N-S, "
          f"~{(bbox[3]-bbox[2])*111.32*np.cos(np.radians(bbox[0])):.1f} km E-W)")
    print()
    hdr = (f"{'res':>4s} {'edge_m':>7s} {'cells':>8s} "
           f"{'%>=2drv':>8s} {'%>=2day':>8s} {'covEv>=2d':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in RESOLUTIONS:
        s = per_res[r]
        print(f"{r:>4d} {s['edge_m']:>7.1f} {s['n_cells']:>8d} "
              f"{100*s['frac_cells_ge2_drive']:>7.1f}% "
              f"{100*s['frac_cells_ge2_day']:>7.1f}% "
              f"{100*s['coverage_visits_ge2_day']:>9.1f}%")
    print("# covEv>=2d = fraction of (drive,cell) traversal EVENTS that land in "
          ">=2-distinct-day cells (not fraction of driven time/distance)")

    if args.summary:
        print()
        print(f"# distinct-drive-count histogram per resolution:")
        for r in RESOLUTIONS:
            b = per_res[r]["drive_count_hist"]
            print(f"#   r{r}: " + "  ".join(f"{k}={v}" for k, v in b.items()))
        print(f"#")
        print(f"# inter-visit temporal gap over the {gap_stats['n_revisited_cells']} "
              f">=2-day cells at r{DECISION_RES} (days between first & last visit):")
        print(f"#   min={gap_stats['min_days']:.1f}  "
              f"median={gap_stats['median_days']:.1f}  "
              f"p90={gap_stats['p90_days']:.1f}  "
              f"max={gap_stats['max_days']:.1f}")
        print(f"#")
        print(f"# depot-exclusion sensitivity (coverage excl. top-K most-driven "
              f"cells, r{DECISION_RES}):")
        print("#   " + "  ".join(f"exclK={k}:{100*v:.1f}%"
                                 for k, v in depot_sens.items()))
        print(f"#")
        print(f"# DECISION (pre-registered): traversal-event coverage over "
              f">=2-day cells at r{DECISION_RES} = {100*dec:.1f}%  "
              f"vs bar {100*DECISION_BAR:.0f}%  -->  {verdict}")

    # ---- JSON ----------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "n_drives_ok": n_ok,
        "n_drives_skipped": n_skip,
        "speed_gate": args.speed_gate,
        "union_bbox": {"min_lat": bbox[0], "max_lat": bbox[1],
                       "min_lng": bbox[2], "max_lng": bbox[3]},
        "total_gated_points": total_gated_points,
        "resolutions": {str(r): per_res[r] for r in RESOLUTIONS},
        "decision": {
            "resolution": DECISION_RES,
            "metric": "coverage_visits_ge2_day",
            "metric_desc": ("fraction of (drive,cell) traversal events landing "
                            "in >=2-distinct-day cells; NOT fraction of driven "
                            "time or distance"),
            "value": dec,
            "bar": DECISION_BAR,
            "verdict": verdict,
            "depot_exclusion_sensitivity": depot_sens,
        },
        "gap_days_r%d" % DECISION_RES: gap_stats,
        "top_cells_r%d" % DECISION_RES: top_cells,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n# wrote {args.out}")

    if args.plot:
        _plot(cell_drive_ts[DECISION_RES], cell_days[DECISION_RES], gap_days)

    return 0


def _plot(cells: dict, days: dict, gap_days: list[float]) -> None:
    """r10 cell map colored by distinct-day count + a histogram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib unavailable; skipping --plot", file=sys.stderr)
        return
    out_dir = Path("journal/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    lats, lngs, ndays = [], [], []
    for cell in cells:
        la, lo = h3.cell_to_latlng(cell)
        lats.append(la); lngs.append(lo); ndays.append(len(days[cell]))

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(lngs, lats, c=ndays, s=6, cmap="viridis", vmin=1)
    fig.colorbar(sc, ax=ax, label="distinct drive-days")
    ax.set_xlabel("lng"); ax.set_ylabel("lat")
    ax.set_title(f"HDD r{DECISION_RES} cells by distinct-day count")
    fig.tight_layout()
    fig.savefig(out_dir / f"hdd_revisit_density_r{DECISION_RES}.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [len(days[c]) for c in cells]
    ax.hist(counts, bins=range(1, max(counts) + 2), align="left",
            color="steelblue", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("distinct drive-days per cell")
    ax.set_ylabel("cells (log)")
    ax.set_title(f"HDD r{DECISION_RES} revisit histogram")
    fig.tight_layout()
    fig.savefig(out_dir / f"hdd_revisit_hist_r{DECISION_RES}.svg")
    plt.close(fig)
    print(f"# wrote journal/figures/hdd_revisit_density_r{DECISION_RES}.svg + hist")


if __name__ == "__main__":
    sys.exit(main())
