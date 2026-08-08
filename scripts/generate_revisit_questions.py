#!/usr/bin/env python3
"""Generate GPS/SLAM-grounded ``last_seen`` look-back questions from revisit
structure — the annotation-free path for the decisive long-multi-revisit test.

See ``journal/genuine_lookback_qa/PREREGISTRATION.md``. This turns any session
with a per-frame position track into a *frozen, deterministic* question bank of
"where/when was I last at this place?" queries, so the fixed-budget cohort
(``scripts/run_wearables_budget_suite.sh``) can be run without manual annotation.

CAVEAT (state it up front): these are annotation-free but still **retrieval
proxies**. They make the cohort affordable; they do NOT answer the reviewers'
"not human-authored look-back questions" objection. Pair with a small
hand-authored bank on 1-2 sessions (``template_questions.yaml`` / PROTOCOL.md)
as a validity check.

Cluster-only: the feature files (``clip_l_features.h5`` with per-frame
``{group}/lat``, ``{group}/lng``, ``{group}/timestamps``) live under
``$PSM_DATA_ROOT``, not in the paper clone. Run this where the data is.

Design:
  * A "place" is a metric cluster of track points (radius ``--place-radius-m``
    on the raw lat/lng), NOT an H3 cell. See "Why metric, not H3" below — this
    decoupling is load-bearing for the preregistered test.
  * A "visit" to a place is a temporally-separated episode of that place's
    frames (``visit_episodes``, gap ``--visit-gap-sec``). A place is *revisited*
    when it has >= ``--min-visits`` episodes whose start-to-start separation is
    >= ``--min-separation-sec`` (a genuine return, not a long dwell).
  * For each revisited place we sample a query timestamp from a *later* visit and
    set the ground-truth interval to an *earlier* visit's span — "when did I last
    see this place before now?". The query carries the earlier visit's mean
    (lat, lng) + a k-ring, so it exercises PSM's spatial substrate (last_seen has
    no text query, by design; retrieval baselines skip it).

Why metric, not H3 (do not "simplify" this back):
    An earlier cut of this generator built the bank from ``h3_cells`` /
    ``group_indices`` — the evaluator's own primitives — so that "place" could
    not drift from how the evaluator scores. That is the wrong trade. It makes
    the question bank a function of the *same* H3 partition and cell-exposure
    statistics that (a) define the rare/common outcome strata and (b) are the
    entire mechanism of the ``spatial_priority`` policy under test. H1 would
    then reduce to "does a policy that retains frames from low-exposure H3 cells
    score well on questions selected to sit in low-exposure H3 cells?", which
    can come out positive by construction. It also blunts the coordinate-null
    control: the permuted arm scrambles the policy's spatial view while the GT
    intervals still encode the true H3-derived structure.
    Selecting places by metric distance on the raw track keeps question
    construction independent of the evaluation partition. Scoring is unchanged —
    ``eval_fixed_budget.py`` still computes cells and strata with ``h3_cells``
    exactly as before — so there is no scoring drift, only construction
    independence. The H3 numbers in the sidecar are diagnostics, never inputs.
  * Output matches the ``eval_lookback.py`` YAML schema exactly, so
    ``QUESTIONS_NAME=<out>`` in the budget suite picks it up unchanged.
  * Fully deterministic under ``--seed``; we print the output SHA-256 so the bank
    can be frozen and cited in the preregistration.

A sidecar ``--metadata-out`` JSON reports per-cell visit counts + exposure and
session-level revisit richness, so a cohort of >= 5 genuine long-multi-revisit
sessions can be *selected* (compare_fixed_budget.py needs >= 2 sessions; the
preregistration targets >= 5 across >= 2 substrates) and the rare/common strata
match the evaluator.

Usage (cluster):
    python scripts/generate_revisit_questions.py \
        --features $PSM_DATA_ROOT/.../<session>/clip_l_features.h5 \
        --out      $PSM_DATA_ROOT/.../<session>/revisit_questions.yaml \
        --metadata-out captures/revisit_meta/<session>.json \
        --place-radius-m 15 --visit-gap-sec 30 --min-separation-sec 120 \
        --n-questions 20 --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
# ``visit_episodes`` is partition-agnostic (it only splits an index array on time
# gaps), so reusing it keeps episode semantics identical to the evaluator without
# importing the H3 partition. ``h3_cells`` is imported for SIDECAR DIAGNOSTICS
# ONLY — it must never feed question selection (see "Why metric, not H3" above).
from eval_fixed_budget import h3_cells, visit_episodes  # noqa: E402


def metric_places(
    lat: np.ndarray,
    lng: np.ndarray,
    *,
    radius_m: float,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Cluster track points into places by metric distance on the raw track.

    Deliberately independent of H3: greedy leader clustering in timestamp order,
    assigning each frame to the first place whose running centroid is within
    ``radius_m``, else opening a new place. Deterministic given the track (no
    RNG, no iteration over unordered containers).

    Returns ``(place_id -> frame indices, per-frame place label)``.
    """
    n = int(lat.size)
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return {}, labels

    # Local equirectangular projection about the track origin: adequate at the
    # session scales here (<= a few km) and avoids a geodesy dependency.
    lat0 = float(lat[0])
    m_per_deg_lat = 110_540.0
    m_per_deg_lng = 111_320.0 * float(np.cos(np.deg2rad(lat0)))
    xs = (lng - float(lng[0])) * m_per_deg_lng
    ys = (lat - lat0) * m_per_deg_lat

    centroids_x: list[float] = []
    centroids_y: list[float] = []
    counts: list[int] = []
    for i in range(n):
        x, y = float(xs[i]), float(ys[i])
        assigned = -1
        best_d2 = radius_m * radius_m
        for p in range(len(centroids_x)):
            dx = x - centroids_x[p]
            dy = y - centroids_y[p]
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                assigned = p
        if assigned < 0:
            centroids_x.append(x)
            centroids_y.append(y)
            counts.append(1)
            assigned = len(centroids_x) - 1
        else:
            c = counts[assigned] + 1
            centroids_x[assigned] += (x - centroids_x[assigned]) / c
            centroids_y[assigned] += (y - centroids_y[assigned]) / c
            counts[assigned] = c
        labels[i] = assigned

    groups = {p: np.flatnonzero(labels == p) for p in range(len(centroids_x))}
    return groups, labels


def load_track(features_h5: Path, group: str):
    """Return (ts_rel, lat, lng, session_start, n_frames) from the feature file.

    Matches eval_fixed_budget.py's expectations: per-frame ``{group}/timestamps``,
    ``{group}/lat``, ``{group}/lng`` (a GPS or geo-anchored SLAM/VIO track).
    """
    with h5py.File(features_h5, "r") as f:
        if group not in f:
            raise SystemExit(f"group {group!r} not in {features_h5}")
        if f"{group}/lat" not in f or f"{group}/lng" not in f:
            raise SystemExit(
                f"{features_h5}::{group} lacks lat/lng datasets — this session "
                "has no position track, so it cannot ground revisit questions. "
                "Use a GPS/SLAM-tracked session (the hypothesis is about revisit "
                "structure)."
            )
        ts = f[f"{group}/timestamps"][:].astype(np.float64)
        lat = f[f"{group}/lat"][:].astype(np.float64)
        lng = f[f"{group}/lng"][:].astype(np.float64)
        n_frames = (
            int(f[f"{group}/embeddings"].shape[0])
            if f"{group}/embeddings" in f
            else int(ts.size)
        )
    if not (ts.size == lat.size == lng.size == n_frames):
        raise SystemExit("timestamps / lat / lng / embeddings row counts disagree")
    session_start = float(ts[0]) if ts.size else 0.0
    ts_rel = (ts - session_start).astype(np.float64)
    return ts_rel, lat, lng, session_start, n_frames


def revisited_places(
    groups: dict[int, np.ndarray],
    ts_rel: np.ndarray,
    *,
    visit_gap_sec: float,
    min_visits: int,
    min_separation_sec: float,
) -> dict[int, list[np.ndarray]]:
    """place -> list of episodes, for places that are *genuinely* revisited."""
    out: dict[int, list[np.ndarray]] = {}
    for cell, indices in groups.items():
        episodes = visit_episodes(indices, ts_rel, visit_gap_sec)
        if len(episodes) < min_visits:
            continue
        starts = [float(ts_rel[ep[0]]) for ep in episodes]
        # Genuine return: at least one pair of consecutive visits separated by
        # >= min_separation_sec (filters out flicker across a cell boundary).
        if not any(
            starts[i + 1] - starts[i] >= min_separation_sec
            for i in range(len(starts) - 1)
        ):
            continue
        out[cell] = episodes
    return out


def build_questions(
    revisits: dict[int, list[np.ndarray]],
    ts_rel: np.ndarray,
    lat: np.ndarray,
    lng: np.ndarray,
    *,
    n_questions: int,
    k_ring: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Deterministic place order: most-revisited first, ties broken by place id.
    cells = sorted(revisits, key=lambda c: (-len(revisits[c]), c))
    questions: list[dict] = []
    for cell in cells:
        if len(questions) >= n_questions:
            break
        episodes = revisits[cell]
        # Query from a later visit (index >= 1); GT = the immediately-preceding
        # visit's span ("last seen before now").
        later_k = int(rng.integers(1, len(episodes)))
        earlier = episodes[later_k - 1]
        later = episodes[later_k]
        gt_start = float(ts_rel[earlier[0]])
        gt_end = float(ts_rel[earlier[-1]])
        q_lat = float(np.mean(lat[earlier]))
        q_lng = float(np.mean(lng[earlier]))
        query_t = float(ts_rel[later[int(rng.integers(0, later.size))]])
        questions.append(
            {
                "id": f"rv_p{cell}_{later_k}",
                "query": f"the place I last visited before t={query_t:.1f}s (place {cell})",
                "category": "location_trace",
                "query_mode": "last_seen",
                "query_lat": round(q_lat, 8),
                "query_lng": round(q_lng, 8),
                "query_k_ring": int(k_ring),
                "query_time_sec": round(query_t, 3),
                "intervals": [[round(gt_start, 3), round(gt_end, 3)]],
                "notes": (
                    f"auto-generated (revisit); place has {len(episodes)} visits; "
                    "metric-clustered (H3-independent) GPS/SLAM-grounded proxy, "
                    "not human-authored"
                ),
            }
        )
    return questions


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--group", default="clip")
    ap.add_argument(
        "--place-radius-m",
        type=float,
        default=15.0,
        help=(
            "metric radius defining a 'place' on the raw track. Question "
            "construction is deliberately H3-independent; see the module "
            "docstring. Do not set this from an H3 edge length."
        ),
    )
    ap.add_argument(
        "--h3-resolution",
        type=int,
        default=12,
        help=(
            "DIAGNOSTIC ONLY — reported in the sidecar for comparison with the "
            "evaluator. Never used to select places or questions."
        ),
    )
    ap.add_argument(
        "--visit-gap-sec",
        type=float,
        default=30.0,
        help="gap that splits a place's frames into visits (matches eval default)",
    )
    ap.add_argument("--min-visits", type=int, default=2)
    ap.add_argument(
        "--min-separation-sec",
        type=float,
        default=120.0,
        help="min start-to-start gap between two visits to count as a genuine return",
    )
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--k-ring", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--metadata-out", type=Path, default=None)
    args = ap.parse_args()

    ts_rel, lat, lng, session_start, n_frames = load_track(args.features, args.group)
    duration_sec = float(ts_rel[-1]) if ts_rel.size else 0.0
    # Places come from the raw track, NOT from the evaluator's H3 partition.
    groups, _place_labels = metric_places(lat, lng, radius_m=args.place_radius_m)
    revisits = revisited_places(
        groups,
        ts_rel,
        visit_gap_sec=args.visit_gap_sec,
        min_visits=args.min_visits,
        min_separation_sec=args.min_separation_sec,
    )

    if not revisits:
        raise SystemExit(
            f"{args.features}: no place has >= {args.min_visits} genuinely-separated "
            f"visits at radius {args.place_radius_m} m — this is NOT a long "
            "multi-revisit session; exclude it from the cohort."
        )

    questions = build_questions(
        revisits,
        ts_rel,
        lat,
        lng,
        n_questions=args.n_questions,
        k_ring=args.k_ring,
        seed=args.seed,
    )

    session_id = args.session_id or args.features.parent.name
    doc = {
        "session_id": session_id,
        "iou_threshold": args.iou_threshold,
        "generator": {
            "script": "generate_revisit_questions.py",
            "place_selection": "metric",
            "place_radius_m": args.place_radius_m,
            "h3_resolution_diagnostic_only": args.h3_resolution,
            "visit_gap_sec": args.visit_gap_sec,
            "min_visits": args.min_visits,
            "min_separation_sec": args.min_separation_sec,
            "seed": args.seed,
            "note": (
                "GPS/SLAM-grounded last_seen proxy; NOT human-authored. Places "
                "are metric clusters on the raw track, independent of the H3 "
                "partition used for scoring/strata (see PREREGISTRATION.md)."
            ),
        },
        "questions": questions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    args.out.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(
        f"[revisit-gen] wrote {len(questions)} questions -> {args.out}", file=sys.stderr
    )
    print(f"[revisit-gen] frozen sha256 = {sha}", file=sys.stderr)

    # Sidecar: revisit richness for cohort selection + strata sanity.
    exposures = {str(c): int(sum(ep.size for ep in eps)) for c, eps in revisits.items()}
    # DIAGNOSTIC ONLY: how the evaluator's partition sees the same track. Useful
    # for sanity-checking that metric places are not degenerate w.r.t. scoring,
    # but never an input to selection.
    n_h3_cells = int(np.unique(h3_cells(lat, lng, args.h3_resolution)).size)
    meta = {
        "session_id": session_id,
        "features": str(args.features),
        "questions_file": str(args.out),
        "questions_sha256": sha,
        "n_frames": n_frames,
        "duration_sec": round(duration_sec, 1),
        "duration_min": round(duration_sec / 60.0, 1),
        "place_selection": "metric",
        "place_radius_m": args.place_radius_m,
        "n_places": len(groups),
        "n_revisited_places": len(revisits),
        "n_h3_cells_diagnostic": n_h3_cells,
        "n_questions": len(questions),
        "revisit_exposure_by_place": exposures,
        # Cohort gate mirrors the preregistration's "long multi-revisit" definition.
        "long_multirevisit": bool(duration_sec >= 1800.0 and len(revisits) >= 5),
    }
    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(meta, indent=2))
        print(f"[revisit-gen] metadata -> {args.metadata_out}", file=sys.stderr)
    print(
        f"[revisit-gen] {session_id}: {meta['duration_min']} min, "
        f"{meta['n_revisited_places']} revisited places "
        f"(metric, r={args.place_radius_m} m), "
        f"long_multirevisit={meta['long_multirevisit']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
