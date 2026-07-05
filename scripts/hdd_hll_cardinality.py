"""F-HDD-2: cross-drive HLL cardinality accrual + ring-buffer time-decay.

The OpenSUN3D reframe claims *unique-event cardinality across revisits* in
bounded memory. On HDD's revisited cells this is the whole story: PSM keeps a
fixed-size (1 KiB) HyperLogLog sketch per H3 cell that

  - ACCRUES across drives: each revisit adds that visit's observations, so the
    per-cell cardinality estimate grows across sessions -- persistent memory;
  - DECAYS via the ring buffer: a windowed estimate keeps only the last
    ``capacity`` time-buckets, so cardinality of *recent* observations falls
    between visits and jumps back up on the next revisit -- bounded, time-aware.

Memory is CONSTANT regardless of how many observations accrue: 2^precision
registers x 1 byte = 1 KiB/cell at precision 10, independent of cardinality or
session length. That is the bounded-memory point the dense alternative (store
every observation) cannot match.

The HLL here is a faithful reimplementation of the C engine's sketch (the engine
has no python bindings). It is validated to match ``targets/psm`` exactly on
real HDD data (--validate; 0.0% median error on the sanity drive), so the
python time-series reflects the real system.

"Item" = the raw float32 embedding bytes, matching the engine. On continuous
embeddings distinct-count ~= observation-count; the systems value is the fixed
memory + mergeability + time-decay, not near-duplicate scene dedup (which would
need quantized codes -- a noted extension).

Seeded by ``captures/hdd/revisit_density.json`` (top revisited r10 cells + their
drive lists). Reads the extraction H5s at <root>/<day>/<drive>/<h5-name>.

Run (needs h3, numpy; matplotlib for --plot; targets/psm for --validate):
  /opt/conda/bin/python scripts/hdd_hll_cardinality.py --root <features> --plot
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h3
except ImportError:
    print("ERR: h3 not importable (need h3>=4.0).", file=sys.stderr)
    raise

DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/hdd/features")
REVISIT_JSON = Path("captures/hdd/revisit_density.json")
DEFAULT_OUT = Path("captures/hdd/hll_cardinality.json")
RESOLUTION = 10
PRECISION = 10                         # 2^10 = 1024 registers = 1 KiB/cell
_MASK64 = (1 << 64) - 1


# ---- faithful HLL (validated == targets/psm) ----------------------------
class HLL:
    __slots__ = ("p", "m", "reg")

    def __init__(self, precision: int = PRECISION, reg=None):
        self.p = precision
        self.m = 1 << precision
        self.reg = np.zeros(self.m, dtype=np.uint8) if reg is None else reg

    def add(self, vecbytes: bytes) -> None:
        hh = int.from_bytes(hashlib.blake2b(vecbytes, digest_size=8).digest(), "big")
        j = hh >> (64 - self.p)
        w = (hh << self.p) & _MASK64
        rho = (64 - self.p + 1) if w == 0 else (64 - w.bit_length() + 1)
        if rho > self.reg[j]:
            self.reg[j] = rho

    def merged_with(self, other: "HLL") -> "HLL":
        return HLL(self.p, np.maximum(self.reg, other.reg))

    def count(self) -> float:
        m = self.m
        alpha = 0.7213 / (1 + 1.079 / m)
        z = 1.0 / np.sum(2.0 ** (-self.reg.astype(np.float64)))
        e = alpha * m * m * z
        if e <= 2.5 * m:                       # small-range correction
            v = int(np.sum(self.reg == 0))
            if v > 0:
                e = m * np.log(m / v)
        return float(e)


class DecayRing:
    """Ring of time-bucketed HLLs; windowed count = merge of live buckets.

    Mirrors the engine ring buffer: observations older than capacity*bucket_sec
    age out, so the windowed estimate decays between revisits.
    """
    def __init__(self, bucket_sec: float, capacity: int, precision: int):
        self.bucket_sec = bucket_sec
        self.capacity = capacity
        self.precision = precision
        self.buckets: dict[int, HLL] = {}

    def add(self, ts: float, vecbytes: bytes) -> None:
        b = int(ts // self.bucket_sec)
        self.buckets.setdefault(b, HLL(self.precision)).add(vecbytes)
        self._evict(b)

    def _evict(self, newest_b: int) -> None:
        cutoff = newest_b - self.capacity + 1
        for b in [b for b in self.buckets if b < cutoff]:
            del self.buckets[b]

    def windowed_count(self, now_ts: float) -> float:
        newest_b = int(now_ts // self.bucket_sec)
        cutoff = newest_b - self.capacity + 1
        live = [h for b, h in self.buckets.items() if b >= cutoff]
        if not live:
            return 0.0
        acc = live[0]
        for h in live[1:]:
            acc = acc.merged_with(h)
        return acc.count()


def _drive_start_unix(drive_id: str) -> float:
    """Parse an HDD drive_id (YYYYMMDDHHMM local) to a unix timestamp.

    tz is treated as UTC: only inter-drive *gaps* (days) matter for the figure,
    and a fixed offset cancels out of gaps.
    """
    return datetime.strptime(drive_id, "%Y%m%d%H%M").replace(
        tzinfo=timezone.utc).timestamp()


def _find_h5(root: Path, drive_id: str, h5_name: str) -> Path | None:
    hits = list(root.glob(f"*/{drive_id}/{h5_name}")) + list(
        root.glob(f"{drive_id}/{h5_name}"))
    return hits[0] if hits else None


def _load(h5: Path, group: str):
    import h5py
    with h5py.File(h5, "r") as f:
        if group not in f or "embeddings" not in f[group]:
            return None
        emb = np.asarray(f[group]["embeddings"], dtype=np.float32)
        lat = np.asarray(f[group]["lat"], dtype=np.float64)
        lng = np.asarray(f[group]["lng"], dtype=np.float64)
        ts = np.asarray(f[group]["timestamps"], dtype=np.float64)
    return (emb, lat, lng, ts) if emb.shape[0] else None


def _validate_against_engine(h5: Path, group: str, resolution: int,
                             precision: int) -> float | None:
    """Median |python_HLL - engine_total|/engine_total over cells, or None."""
    if not Path("targets/psm").exists():
        return None
    loaded = _load(h5, group)
    if loaded is None:
        return None
    emb, lat, lng, _ = loaded
    regs: dict[str, HLL] = {}
    for i in range(emb.shape[0]):
        c = h3.latlng_to_cell(float(lat[i]), float(lng[i]), resolution)
        regs.setdefault(c, HLL(precision)).add(emb[i].tobytes())
    py = {c: h.count() for c, h in regs.items()}
    try:
        out = json.loads(subprocess.run(
            ["targets/psm", str(h5), group, "1000000000", str(resolution),
             "1", str(precision), "-j"],
            capture_output=True, text=True, check=True).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    eng = {t["cell"]: t["total"] for t in out["tiles"]}
    rel = [abs(py[c] - eng[c]) / eng[c] for c in py if c in eng and eng[c] > 0]
    return float(np.median(rel)) if rel else None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--revisit-json", type=Path, default=REVISIT_JSON)
    ap.add_argument("--h5-name", default="clip_l_features.h5")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--precision", type=int, default=PRECISION)
    ap.add_argument("--top-cells", type=int, default=12,
                    help="how many top revisited cells to trace")
    ap.add_argument("--bucket-days", type=float, default=7.0,
                    help="ring-buffer bucket width (days) for the decay curve")
    ap.add_argument("--capacity", type=int, default=12,
                    help="ring-buffer capacity (buckets); window = cap*bucket")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check the python HLL against targets/psm")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if not args.revisit_json.exists():
        print(f"ERR: {args.revisit_json} not found -- run hdd_revisit_density.py "
              f"first (it emits top_cells_r{args.resolution}).", file=sys.stderr)
        return 1
    rev = json.loads(args.revisit_json.read_text())
    top = rev.get(f"top_cells_r{args.resolution}", [])[:args.top_cells]
    if not top:
        print(f"ERR: no top_cells_r{args.resolution} in {args.revisit_json}",
              file=sys.stderr)
        return 1
    target_cells = {c["cell"] for c in top}
    drives_needed = sorted({d for c in top for d in c["drive_ids"]})

    # Gather per-target-cell (abs_ts, embedding_bytes) across drives.
    per_cell: dict[str, list] = defaultdict(list)
    bucket_sec = args.bucket_days * 86400.0
    n_loaded = 0
    loaded_drives: set[str] = set()
    for drive_id in drives_needed:
        h5 = _find_h5(args.root, drive_id, args.h5_name)
        if h5 is None:
            continue
        loaded = _load(h5, args.group)
        if loaded is None:
            continue
        n_loaded += 1
        loaded_drives.add(drive_id)
        emb, lat, lng, ts = loaded
        start = _drive_start_unix(drive_id)
        for i in range(emb.shape[0]):
            c = h3.latlng_to_cell(float(lat[i]), float(lng[i]), args.resolution)
            if c in target_cells:
                per_cell[c].append((start + float(ts[i]), emb[i].tobytes(), drive_id))

    if n_loaded == 0:
        print(f"[hdd-f2] no drive H5s found under {args.root} (looked for "
              f"{len(drives_needed)} drives x {args.h5_name}). Wrote nothing -- "
              f"rerun once the extraction array lands.", file=sys.stderr)
        return 0

    val = None
    if args.validate:
        vh5 = next((h for d in drives_needed
                    if (h := _find_h5(args.root, d, args.h5_name)) is not None),
                   None)
        if vh5 is not None:
            val = _validate_against_engine(vh5, args.group, args.resolution,
                                           args.precision)

    reg_bytes = (1 << args.precision)  # 1 byte/register
    series: list[dict[str, Any]] = []
    n_partial = 0
    for c in top:
        cell = c["cell"]
        obs = sorted(per_cell.get(cell, []), key=lambda x: x[0])
        if not obs:
            continue
        # Coverage: how many of THIS cell's revisits actually loaded, and the
        # days/gap RECOMPUTED from loaded observations (not copied from the
        # revisit JSON, which reflects the full GPS history). Warn on shortfall.
        drives_here = {o[2] for o in obs}
        n_drives_traced = len(drives_here)
        n_days_traced = len({d[:8] for d in drives_here})
        gap_days_traced = (obs[-1][0] - obs[0][0]) / 86400.0
        if n_drives_traced < c["n_drives"]:
            n_partial += 1
            print(f"[hdd-f2] {cell}: partial coverage -- {n_drives_traced}/"
                  f"{c['n_drives']} revisit drives have features; curve + "
                  f"days/gap recomputed from the loaded subset.", file=sys.stderr)

        # Sample accrual (monotone) AND windowed (decay) on a TIME GRID, not
        # only at observation times -- otherwise the windowed estimate is always
        # read at the newest bucket (>=1) and the decay-to-~0 during multi-month
        # gaps is never emitted.
        t0, tN = obs[0][0], obs[-1][0]
        span = tN - t0
        if span <= 0:
            grid = [t0]
        else:
            step = max(bucket_sec / 2.0, span / 400.0)
            grid = list(np.arange(t0, tN + step, step))
        grid = sorted(set(grid) | {o[0] for o in obs})  # land on revisit jumps

        acc = HLL(args.precision)
        ring = DecayRing(bucket_sec, args.capacity, args.precision)
        oi = 0
        pts = []
        for tick in grid:
            while oi < len(obs) and obs[oi][0] <= tick:
                acc.add(obs[oi][1])
                ring.add(obs[oi][0], obs[oi][1])
                oi += 1
            pts.append({"ts": float(tick),
                        "accrual": acc.count(),               # monotone
                        "windowed": ring.windowed_count(tick),  # decays in gaps
                        "n_obs": oi})
        series.append({
            "cell": cell, "lat": c["lat"], "lng": c["lng"],
            "n_drives_expected": c["n_drives"],
            "n_drives_traced": n_drives_traced,
            "n_days_expected": c["n_days"], "n_days_traced": n_days_traced,
            "gap_days_expected": c["gap_days"], "gap_days_traced": gap_days_traced,
            "final_accrual": float(pts[-1]["accrual"]),
            "n_observations": len(obs),
            "points": pts,
        })

    if not series:
        print("[hdd-f2] target cells had no frames in the loaded H5s.",
              file=sys.stderr)
        return 0

    result = {
        "root": str(args.root),
        "resolution": args.resolution, "precision": args.precision,
        "bytes_per_cell": reg_bytes,
        "bucket_days": args.bucket_days, "capacity": args.capacity,
        "n_drives_union_expected": len(drives_needed),
        "n_drives_loaded": n_loaded,
        "top_cells_requested": args.top_cells,
        "n_cells_traced": len(series),
        "n_cells_partial_coverage": n_partial,
        "engine_validation_median_relerr": val,
        "cells": series,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    top1 = series[0]
    print(f"# F-HDD-2 HLL cardinality  ({n_loaded}/{len(drives_needed)} union "
          f"drives loaded, {len(series)} revisited cells traced, r{args.resolution}, "
          f"p{args.precision} = {reg_bytes} B/cell CONSTANT)")
    if n_partial:
        print(f"# NOTE: {n_partial}/{len(series)} cells have partial feature "
              f"coverage; their days/gap are recomputed from loaded drives.")
    if val is not None:
        print(f"# engine cross-check: python HLL vs targets/psm median relerr = {val:.1%}")
    print(f"# example cell {top1['cell']} "
          f"({top1['n_days_traced']}/{top1['n_days_expected']} days traced, "
          f"{top1['n_observations']} obs over {top1['gap_days_traced']:.0f} d): "
          f"accrual {float(top1['points'][0]['accrual']):.0f} -> {top1['final_accrual']:.0f}")
    print(f"# wrote {args.out}")

    if args.plot:
        _plot(series, reg_bytes)
    return 0


def _plot(series, reg_bytes) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib unavailable; skipping --plot", file=sys.stderr)
        return
    out_dir = Path("journal/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in series[:6]:
        t0 = s["points"][0]["ts"]
        days = [(p["ts"] - t0) / 86400.0 for p in s["points"]]
        ax1.plot(days, [p["accrual"] for p in s["points"]], lw=1.6,
                 label=f"{s['cell'][:8]} ({s['n_days_traced']}d)")
        ax2.plot(days, [p["windowed"] for p in s["points"]], lw=1.2)
    ax1.set_xlabel("days since first visit"); ax1.set_ylabel("HLL cardinality")
    ax1.set_title(f"Accrual across drives (bounded {reg_bytes} B/cell)")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("days since first visit"); ax2.set_ylabel("windowed HLL cardinality")
    ax2.set_title("Ring-buffer time-decay (windowed)")
    fig.tight_layout()
    fig.savefig(out_dir / "hdd_hll_cardinality.svg")
    plt.close(fig)
    print("# wrote journal/figures/hdd_hll_cardinality.svg")


if __name__ == "__main__":
    sys.exit(main())
