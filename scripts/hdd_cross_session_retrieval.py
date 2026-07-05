"""F-HDD-3: self-supervised cross-session place retrieval on Honda HDD.

Closes Option B's "is the persistent memory actually *retrievable* across
sessions?" gap without any hand annotation. The claim: a place seen on drive A
looks like the same place seen on drive B (a different day), so an exemplar from
one visit retrieves the other visit's frames in the same H3 cell above frames
from other cells -- purely from CLIP cosine, no labels. The cross-drive same-cell
match IS the supervision signal.

Metric (per query frame q in cell C from drive A):
  positives = frames in cell C from OTHER drives (cross-session, same place)
  negatives = frames in other cells (any drive)
  rank all candidates by cosine(q, .); AUC = P(a random positive outranks a
  random negative). AUC ~0.5 => no cross-session signal (place not visually
  stable, or embeddings collapsed); AUC -> 1 => strong cross-session place
  recognition. Also report hit-rate@k (a cross-session same-cell frame in the top-k of
the full candidate pool, incl. same-drive near-duplicate distractors).

Controls:
  - same-drive same-cell AUC (trivial upper bound; frames seconds apart)
  - shuffled-cell AUC (labels permuted; must fall to ~0.5 -- sanity that the
    metric isn't inflated)

Caveat (for write-up): positives are same-r10-cell/other-drives and negatives
are all-other-cells, so the AUC blends coarse geographic proximity (~66 m cells)
with visual place identity. To isolate visual recognition, add a k-NN-cell
control that uses the nearest neighbouring cells as hard negatives. The reported
AUC is a per-query mean (cells weighted by query count, capped at
--max-queries-per-cell), not a per-cell mean.

Degenerate-embedding contingency (embed-sanity verdict): if windshield-frame
CLIP separability is low, cross-session AUC will sit near 0.5. That is itself
the honest result -- report it with the low-separability caveat rather than
suppressing it. (The CLIP-L gate PASSED, cos spread 0.41-0.94, so we expect
meaningful signal.)

Reuses: h3.latlng_to_cell (r10, RTK-noise-robust), pre-normalized clip
embeddings (dot = cosine). Reads the extraction H5s written by
extract_hdd_sessions.py at <root>/<drive_day>/<drive_id>/<h5-name>.

Run (needs h3, numpy; matplotlib for --plot):
  /opt/conda/bin/python scripts/hdd_cross_session_retrieval.py --root <features>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import h3
except ImportError:
    print("ERR: h3 not importable (need h3>=4.0).", file=sys.stderr)
    raise

DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/hdd/features")
DEFAULT_OUT = Path("captures/hdd/cross_session_retrieval.json")
RESOLUTION = 10          # RTK-noise-robust decision resolution
HIT_KS = (1, 5, 10)


def _find_h5s(root: Path, h5_name: str) -> list[tuple[str, str, Path]]:
    """(drive_day, drive_id, h5) for the <root>/<day>/<drive>/<h5> layout.

    Falls back to a shallow <root>/<drive>/<h5> layout (e.g. a single-drive
    sanity dir) so the plumbing is smoke-testable off one extraction.
    """
    out = []
    for p in sorted(root.glob(f"*/*/{h5_name}")):
        out.append((p.parent.parent.name, p.parent.name, p))
    if not out:
        for p in sorted(root.glob(f"*/{h5_name}")):
            out.append((p.parent.name, p.parent.name, p))
    return out


def _load(h5: Path, group: str):
    import h5py
    with h5py.File(h5, "r") as f:
        if group not in f or "embeddings" not in f[group]:
            return None
        emb = np.asarray(f[group]["embeddings"], dtype=np.float32)
        lat = np.asarray(f[group]["lat"], dtype=np.float64)
        lng = np.asarray(f[group]["lng"], dtype=np.float64)
    if emb.shape[0] == 0:
        return None
    # Re-normalize defensively (attr says normalized, but be safe).
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return emb, lat, lng


def _auc_tie(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC = P(pos > neg) + 0.5 P(pos == neg), tie-corrected.

    Vectorized via searchsorted (no Python loop -- a per-element rank loop is
    intractable when called ~thousands of times over ~370k negatives): sort the
    negatives once, then for each positive count negatives strictly below (0.5
    credit for ties), which is exactly the tie-corrected AUC.
    """
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ns = np.sort(neg)
    lo = np.searchsorted(ns, pos, side="left")    # negatives strictly < pos
    hi = np.searchsorted(ns, pos, side="right")   # negatives <= pos
    return float((lo + 0.5 * (hi - lo)).sum() / (pos.size * neg.size))


def _hit_at_k(sims: np.ndarray, cand_mask: np.ndarray, pos_mask: np.ndarray,
              ks) -> dict:
    """hit-rate@k: is any positive in the top-k of the FULL candidate ranking?

    The candidate pool is every frame except the query -- crucially INCLUDING
    same-drive same-cell frames, the strongest (near-duplicate) distractors, so
    the number reflects realistic retrieval difficulty. Ties are broken
    adversarially (negatives rank ahead of positives).

    O(n) via a threshold count instead of an O(n log n) full-pool sort (the sort
    is intractable when called per-query over ~370k frames): the best positive's
    adversarial rank is the number of candidates that (strictly) outrank it plus
    the negatives tied with it, so it is in top-k iff that rank < k.
    """
    pos_sims = sims[pos_mask]
    if pos_sims.size == 0:
        if not cand_mask.any():
            return {k: float("nan") for k in ks}
        return {k: 0.0 for k in ks}
    best = float(pos_sims.max())
    sc = sims[cand_mask]
    greater = int(np.count_nonzero(sc > best))          # all strictly-better cands
    neg_cand = cand_mask & ~pos_mask
    neg_equal = int(np.count_nonzero(sims[neg_cand] == best))  # tied negs rank ahead
    rank0 = greater + neg_equal                         # 0-indexed pos of 1st positive
    return {k: float(rank0 < k) for k in ks}



def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--h5-name", default="clip_l_features.h5")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--resolution", type=int, default=RESOLUTION)
    ap.add_argument("--max-queries-per-cell", type=int, default=5,
                    help="cap query exemplars per (cell, drive) to bound cost")
    ap.add_argument("--max-total-queries", type=int, default=2000,
                    help="global cap on query exemplars, stratified across cells "
                         "(each query costs one all-frames pass; 2000 gives a "
                         "tight AUC estimate in ~2 min). 0 = no cap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    h5s = _find_h5s(args.root, args.h5_name)
    if not h5s:
        print(f"ERR: no {args.h5_name} under {args.root} "
              f"(expected <root>/<day>/<drive>/{args.h5_name})", file=sys.stderr)
        return 1

    # Load every drive; concatenate embeddings with per-frame (drive_idx, cell).
    embs, drive_of, cell_of = [], [], []
    drive_names = []
    for di, (day, drive_id, h5) in enumerate(h5s):
        loaded = _load(h5, args.group)
        if loaded is None:
            print(f"[hdd-f3] {drive_id}: no usable {args.group} group, skip",
                  file=sys.stderr)
            continue
        emb, lat, lng = loaded
        cells = [h3.latlng_to_cell(float(a), float(o), args.resolution)
                 for a, o in zip(lat, lng)]
        embs.append(emb)
        drive_of.append(np.full(emb.shape[0], len(drive_names), dtype=np.int32))
        cell_of.extend(cells)
        drive_names.append(drive_id)

    if len(drive_names) < 2:
        print(f"[hdd-f3] only {len(drive_names)} drive(s) loaded -- need >=2 for "
              f"cross-session pairs. Plumbing OK; rerun once the extraction array "
              f"has produced multiple drive H5s.", file=sys.stderr)
        return 0

    emb = np.concatenate(embs, axis=0)
    drive = np.concatenate(drive_of)
    # Factorize H3 string cell ids -> integer codes: every `cell == c` mask below
    # runs per query/group over ~370k entries, and string comparison is ~100x
    # slower than int. Codes are all we need (cell ids never appear in output).
    _, cell = np.unique(np.array(cell_of), return_inverse=True)
    cell = cell.astype(np.int32)
    del embs, drive_of

    # Cells revisited by >=2 distinct drives are the cross-session testbed.
    cell_drives: dict[int, set] = defaultdict(set)
    for c, d in zip(cell, drive):
        cell_drives[c].add(int(d))
    revisited = [c for c, ds in cell_drives.items() if len(ds) >= 2]
    if not revisited:
        print("[hdd-f3] no cell visited by >=2 drives; nothing to test.",
              file=sys.stderr)
        return 0

    rng = np.random.default_rng(args.seed)
    cross_auc, same_auc, shuf_auc = [], [], []
    hit = {k: [] for k in HIT_KS}
    cross_cos, diff_cos = [], []   # for the distribution figure
    # Precompute a permuted cell labelling for the shuffled control.
    shuf_cell = cell.copy()
    rng.shuffle(shuf_cell)

    # Collect candidate query exemplars grouped by cell (cap per (cell,drive)),
    # then STRATIFY-sample down to --max-total-queries round-robin across cells.
    # Each query costs one all-frames pass, so an uncapped ~1e5 queries is
    # intractable; stratifying also gives broad cell coverage rather than
    # weighting toward the few busiest cells.
    cand_by_cell: dict[int, list] = defaultdict(list)
    for c in revisited:
        in_c = cell == c
        for a in sorted(cell_drives[c]):
            qs = np.where(in_c & (drive == a))[0]
            if qs.size == 0:
                continue
            if qs.size > args.max_queries_per_cell:
                qs = rng.choice(qs, args.max_queries_per_cell, replace=False)
            cand_by_cell[c].extend((int(a), int(qi)) for qi in qs)
    total_cand = sum(len(v) for v in cand_by_cell.values())
    cells_list = list(cand_by_cell)
    rng.shuffle(cells_list)
    for c in cells_list:
        rng.shuffle(cand_by_cell[c])
    cap = args.max_total_queries
    if cap and total_cand > cap:
        sampled, ptr = [], {c: 0 for c in cells_list}
        while len(sampled) < cap:
            progressed = False
            for c in cells_list:
                p = ptr[c]
                if p < len(cand_by_cell[c]):
                    a, qi = cand_by_cell[c][p]
                    sampled.append((c, a, qi)); ptr[c] = p + 1; progressed = True
                    if len(sampled) >= cap:
                        break
            if not progressed:
                break
        print(f"[hdd-f3] stratified-sampled {len(sampled)}/{total_cand} candidate "
              f"queries across {len(cells_list)} cells (cap {cap})", file=sys.stderr)
    else:
        sampled = [(c, a, qi) for c in cells_list for (a, qi) in cand_by_cell[c]]
        print(f"[hdd-f3] {len(sampled)} queries across {len(cells_list)} cells "
              f"(under cap {cap})", file=sys.stderr)

    # Group sampled queries by (cell, drive) so pos/neg masks are built once.
    by_ca: dict[tuple, list] = defaultdict(list)
    for c, a, qi in sampled:
        by_ca[(c, a)].append(qi)

    for (c, a), qis in by_ca.items():
        in_c = cell == c
        pos_cross = in_c & (drive != a)          # same cell, other drives
        pos_same = in_c & (drive == a)           # same cell, same drive
        neg = ~in_c                              # other cells
        for qi in qis:
            q = emb[qi]
            sims = emb @ q
            au = _auc_tie(sims[pos_cross], sims[neg])
            if not np.isnan(au):
                cross_auc.append(au)
                # hit@k over the FULL pool minus q (incl. same-drive
                # same-cell distractors); positives = cross-session.
                cand = np.ones(emb.shape[0], dtype=bool); cand[qi] = False
                hk = _hit_at_k(sims, cand, pos_cross, HIT_KS)
                for k in HIT_KS:
                    if not np.isnan(hk[k]):
                        hit[k].append(hk[k])
                cross_cos.append(float(sims[pos_cross].mean()))
                diff_cos.append(float(sims[neg].mean()))
            # same-drive positives (upper bound), excluding q itself
            ps = pos_same.copy(); ps[qi] = False
            asame = _auc_tie(sims[ps], sims[neg])
            if not np.isnan(asame):
                same_auc.append(asame)
            # shuffled control: positives = same *shuffled* cell, other drives
            sc = shuf_cell[qi]
            pos_sh = (shuf_cell == sc) & (drive != a)
            neg_sh = shuf_cell != sc
            ash = _auc_tie(sims[pos_sh], sims[neg_sh])
            if not np.isnan(ash):
                shuf_auc.append(ash)

    def _stats(xs):
        a = np.array(xs, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
                "std": float(a.std()) if a.size else float("nan")}

    result = {
        "root": str(args.root),
        "n_drives": len(drive_names),
        "n_frames": int(emb.shape[0]),
        "resolution": args.resolution,
        "seed": args.seed,
        "max_queries_per_cell": args.max_queries_per_cell,
        "max_total_queries": args.max_total_queries,
        "auc_weighting": "per-query mean (cells weighted by query count, "
                         "capped at max_queries_per_cell)",
        "caveat": "positives=same r10 cell/other drives, negatives=all other "
                  "cells: AUC blends coarse-geography with visual place identity. "
                  "If written up, add a k-NN-cell control (nearest cells as hard "
                  "negatives) to isolate visual recognition.",
        "n_revisited_cells": len(revisited),
        "cross_session_auc": _stats(cross_auc),
        "same_drive_auc_upper_bound": _stats(same_auc),
        "shuffled_cell_auc_control": _stats(shuf_auc),
        "hit_rate_at_k": {str(k): _stats(hit[k]) for k in HIT_KS},
        "cross_session_cos_mean": _stats(cross_cos),
        "different_cell_cos_mean": _stats(diff_cos),
    }

    print(f"# F-HDD-3 cross-session retrieval  ({len(drive_names)} drives, "
          f"{emb.shape[0]} frames, {len(revisited)} revisited r{args.resolution} cells)")
    ca, sa, sh = result["cross_session_auc"], result["same_drive_auc_upper_bound"], result["shuffled_cell_auc_control"]
    print(f"# cross-session AUC   = {ca['mean']:.3f} +/- {ca['std']:.3f}  (n={ca['n']})")
    print(f"# same-drive AUC (UB) = {sa['mean']:.3f}  |  shuffled-cell (floor) = {sh['mean']:.3f}")
    print(f"# hit-rate@k (any cross-session frame in top-k of full pool): " + "  ".join(
        f"@{k}={result['hit_rate_at_k'][str(k)]['mean']:.2f}" for k in HIT_KS))
    print(f"# cos(query, same-place other-drive)={result['cross_session_cos_mean']['mean']:.3f}  "
          f"vs cos(query, other-cell)={result['different_cell_cos_mean']['mean']:.3f}")
    verdict = ("STRONG cross-session place recognition" if ca["mean"] > 0.75
               else "WEAK/degenerate -- windshield CLIP separability low; "
                    "report with caveat or pivot to GPS-only consistency"
               if ca["mean"] < 0.6 else "MODERATE cross-session signal")
    print(f"# VERDICT: {verdict}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"# wrote {args.out}")

    if args.plot:
        _plot(cross_cos, diff_cos, ca["mean"], sa["mean"], sh["mean"])
    return 0


def _plot(cross_cos, diff_cos, cross_auc, same_auc, shuf_auc) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARN: matplotlib unavailable; skipping --plot", file=sys.stderr)
        return
    out_dir = Path("journal/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(cross_cos, bins=30, alpha=0.7, label="same place, other drive",
             color="C0", density=True)
    ax1.hist(diff_cos, bins=30, alpha=0.6, label="other cell", color="C3",
             density=True)
    ax1.set_xlabel("mean cosine to query exemplar")
    ax1.set_ylabel("density"); ax1.legend(fontsize=8)
    ax1.set_title("Cross-session place similarity")
    ax2.bar(["shuffled\n(floor)", "cross-\nsession", "same-drive\n(UB)"],
            [shuf_auc, cross_auc, same_auc], color=["gray", "C0", "C2"])
    ax2.axhline(0.5, ls="--", color="k", lw=0.8)
    ax2.set_ylim(0.4, 1.0); ax2.set_ylabel("retrieval AUC")
    ax2.set_title("Cross-session retrieval AUC")
    fig.tight_layout()
    fig.savefig(out_dir / "hdd_cross_session_retrieval.svg")
    plt.close(fig)
    print("# wrote journal/figures/hdd_cross_session_retrieval.svg")


if __name__ == "__main__":
    sys.exit(main())
