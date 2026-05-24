"""Shared helpers for retrieval-baseline evaluation scripts.

Owns the scoring, JSON-output shape, and CLIP-text-embed plumbing that the
three E11 baselines (`eval_brute_force_clip.py`, `eval_sliding_window.py`,
`eval_uniform_sample.py`) and `eval_lookback.py` all need.

The output JSON schema matches `eval_lookback.py --out` exactly so that
`eval_aggregate.py` can pool runs across methods. We add one extra
top-level field, `baseline_method`, which is a free-form string identifying
the retrieval method (e.g. "brute_force_clip", "sliding_window_5s",
"uniform_sample_75s", or "psm" for the main harness).

Last-seen questions are skipped by every retrieval baseline because they
have no text query — they're PSM-specific by design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


# ----------------------------------------------------------------------------
# IoU primitives — bitwise copies of eval_lookback.py's helpers so the scoring
# is byte-identical. Centralized here so future fixes touch one place.
# ----------------------------------------------------------------------------

def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(
    pred: tuple[float, float], gts: list[tuple[float, float]]
) -> tuple[float, int]:
    """Return (best_iou, idx_of_matching_gt). idx = -1 if no gt or no overlap."""
    if not gts:
        return 0.0, -1
    best = 0.0
    best_idx = -1
    for i, gt in enumerate(gts):
        v = interval_iou(pred, gt)
        if v > best:
            best = v
            best_idx = i
    return best, best_idx


def point_in_any(t: float, gts: list[tuple[float, float]]) -> int:
    for i, (a, b) in enumerate(gts):
        if a <= t <= b:
            return i
    return -1


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------

def load_questions(path: Path) -> dict:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "yaml not installed. Either pip install pyyaml or use a .json file."
            ) from exc
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise SystemExit(f"Unsupported question file format: {path}")


def auto_session_start(features_h5: Path, group: str) -> float:
    with h5py.File(features_h5, "r") as f:
        if group not in f:
            raise SystemExit(f"group {group!r} not in {features_h5}")
        ts = f[f"{group}/timestamps"][:]
    return float(ts[0])


def load_features(
    features_h5: Path, group: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load (embeddings, timestamps_rel, session_start) from the feature file.

    Embeddings are L2-normalized per row so a dot product equals cosine
    similarity. Timestamps are converted to seconds-since-session-start so
    they line up with the YAML interval annotations.
    """
    with h5py.File(features_h5, "r") as f:
        if group not in f:
            raise SystemExit(f"group {group!r} not in {features_h5}")
        emb = f[f"{group}/embeddings"][:].astype(np.float32)
        ts = f[f"{group}/timestamps"][:].astype(np.float64)
    session_start = float(ts[0])
    ts_rel = (ts - session_start).astype(np.float64)
    # Per-row L2 normalize (in place). Zero-norm rows are left at zero, which
    # yields cosine=0 on dot product against any query — harmless.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    emb_unit = (emb / safe).astype(np.float32)
    return emb_unit, ts_rel, session_start


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def score_predictions(
    pred_intervals: list[tuple[float, float, float | None]],
    gts_rel: list[tuple[float, float]],
    *,
    exemplar_tolerance: float,
) -> list[dict]:
    """Score a top-k list of predicted intervals against ground-truth intervals.

    `pred_intervals` is `[(t_min, t_max, similarity)]` where similarity may be
    None (last-seen-style predictions don't carry one). Returns a list of
    per-prediction records in the same shape as `eval_lookback.py` so the
    aggregator can pool brute-force baselines alongside PSM runs.

    For interval predictions that are wider than ±exemplar_tolerance (e.g.
    sliding window), we use the interval midpoint as the synthetic
    `exemplar_t` so the existing exemplar-IoU math still works. The exemplar
    IoU is then computed against `[mid - tol, mid + tol]`, which means a
    sliding-window method's exemplar IoU is essentially "does the window's
    midpoint land near the GT?" — the right semantics for that method.
    """
    preds = []
    for (t_min, t_max, sim) in pred_intervals:
        exemplar_t = (t_min + t_max) / 2.0
        bucket_iou, gt_idx_b = best_iou((t_min, t_max), gts_rel)
        tol = exemplar_tolerance
        exemplar_iou, gt_idx_e = best_iou(
            (exemplar_t - tol, exemplar_t + tol), gts_rel
        )
        exemplar_hit_idx = point_in_any(exemplar_t, gts_rel)
        preds.append({
            "cell": None,           # baselines have no spatial cell
            "lat": None,
            "lng": None,
            "similarity": (None if sim is None else float(sim)),
            "exemplar_t": exemplar_t,
            "t_min": t_min,
            "t_max": t_max,
            "count": 0.0,
            "bucket_iou": bucket_iou,
            "bucket_matched_gt": gt_idx_b,
            "exemplar_iou": exemplar_iou,
            "exemplar_matched_gt": gt_idx_e,
            "exemplar_hits_gt": exemplar_hit_idx >= 0,
        })
    return preds


def summarize_question(
    qid: str,
    text: str,
    category: str,
    notes: str,
    gts_rel: list[tuple[float, float]],
    preds: list[dict],
    *,
    iou_threshold: float,
) -> dict:
    """Question-level aggregation. Mirrors eval_lookback.py exactly."""
    bucket_top1 = preds[0]["bucket_iou"] if preds else 0.0
    bucket_at_k = max((p["bucket_iou"] for p in preds), default=0.0)
    exemplar_top1 = preds[0]["exemplar_iou"] if preds else 0.0
    exemplar_at_k = max((p["exemplar_iou"] for p in preds), default=0.0)
    any_exemplar_hit = any(p["exemplar_hits_gt"] for p in preds)
    return {
        "id": qid,
        "query": text,
        "query_mode": "similarity_search",
        "category": category or "(uncategorized)",
        "notes": notes,
        "intervals_gt": gts_rel,
        "count_gt": None,
        "count_predicted": 0,
        "count_abs_error": None,
        "count_correct": None,
        "expected": None,
        "preds": preds,
        "bucket_iou_top1": bucket_top1,
        "bucket_iou_at_k": bucket_at_k,
        "bucket_hit_at_k": bucket_at_k >= iou_threshold,
        "exemplar_iou_top1": exemplar_top1,
        "exemplar_iou_at_k": exemplar_at_k,
        "exemplar_hit_at_k": any_exemplar_hit,
    }


def write_eval_json(
    out_path: Path,
    *,
    features_h5: Path,
    questions_file: Path,
    group: str,
    top: int,
    records: list[dict],
    session_start: float,
    clip_checkpoint: str,
    clip_backend: str,
    iou_threshold: float,
    exemplar_tolerance: float,
    baseline_method: str,
    extra_settings: dict | None = None,
    seed: int | None = None,
) -> None:
    """Write a JSON record compatible with `scripts/eval_aggregate.py`.

    The schema matches eval_lookback.py's `--out` output: same top-level
    keys, same `summary`, same `by_category`, same `records`. The one
    addition is `baseline_method`, a free-form string identifying the
    retrieval method (lets a future aggregator pool by method).

    `extra_settings` is merged into the top-level JSON so per-method knobs
    (window_size, sample_rate, etc.) survive in the output.
    """
    scored = [r for r in records if r["intervals_gt"]]
    n = max(len(scored), 1)
    miou_bucket_top1 = sum(r["bucket_iou_top1"] for r in scored) / n
    miou_bucket_at_k = sum(r["bucket_iou_at_k"] for r in scored) / n
    miou_exemplar_top1 = sum(r["exemplar_iou_top1"] for r in scored) / n
    miou_exemplar_at_k = sum(r["exemplar_iou_at_k"] for r in scored) / n
    bucket_hit_rate = sum(1 for r in scored if r["bucket_hit_at_k"]) / n
    exemplar_hit_rate = sum(1 for r in scored if r["exemplar_hit_at_k"]) / n

    by_category: dict[str, list[dict]] = {}
    for r in scored:
        by_category.setdefault(r["category"], []).append(r)

    out_data = {
        "features": str(features_h5),
        "questions_file": str(questions_file),
        "group": group,
        "top": top,
        "time_window_sec": None,
        "capacity": None,
        "h3_resolution": None,
        "precision": None,
        "exemplars": None,
        "exemplar_codec": "n/a",
        "iou_threshold": iou_threshold,
        "exemplar_tolerance_sec": exemplar_tolerance,
        "psm_seed": seed,
        "session_start_unix": session_start,
        "clip_checkpoint": clip_checkpoint,
        "clip_backend": clip_backend,
        "baseline_method": baseline_method,
        "summary": {
            "n_questions": len(records),
            "n_scored": len(scored),
            "n_negative_controls": 0,
            "n_counting": 0,
            "n_spatial": 0,
            "bucket_miou_top1": miou_bucket_top1,
            f"bucket_miou_at_{top}": miou_bucket_at_k,
            f"bucket_hit_rate_at_{top}": bucket_hit_rate,
            "exemplar_miou_top1": miou_exemplar_top1,
            f"exemplar_miou_at_{top}": miou_exemplar_at_k,
            f"exemplar_hit_rate_at_{top}": exemplar_hit_rate,
            "count_exact_match_rate": None,
            "count_mae": None,
        },
        "by_category": {
            cat: {
                "n": len(records_in_cat),
                "exemplar_miou_top1": sum(r["exemplar_iou_top1"] for r in records_in_cat) / len(records_in_cat),
                f"exemplar_miou_at_{top}": sum(r["exemplar_iou_at_k"] for r in records_in_cat) / len(records_in_cat),
                f"exemplar_hit_rate_at_{top}": sum(1 for r in records_in_cat if r["exemplar_hit_at_k"]) / len(records_in_cat),
            }
            for cat, records_in_cat in by_category.items()
        },
        "records": records,
    }
    if extra_settings:
        out_data.update(extra_settings)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"[{baseline_method}] wrote {out_path}", file=sys.stderr)


# ----------------------------------------------------------------------------
# CLIP text embedding — same path as eval_lookback.py uses for queries
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "extraction"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def make_clip_runner(checkpoint: str, device: str = "auto"):
    """Construct the same CLIP runner eval_lookback uses for `--search`."""
    from psm_extraction.models import make_runner
    runner = make_runner(
        "clip",
        checkpoint=checkpoint,
        backend="auto",
        device=device,
    )
    print(f"[eval] CLIP runner: {runner.backend}", file=sys.stderr)
    return runner


def embed_query_text(runner, text: str) -> np.ndarray:
    """Return a unit-normalized float32 query vector."""
    q = runner.embed_text(text).astype(np.float32)
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    return q.astype(np.float32)


# ----------------------------------------------------------------------------
# Per-question dispatch (shared by all three baselines)
# ----------------------------------------------------------------------------

def question_text_and_skip_reason(q: dict) -> tuple[str | None, str | None]:
    """Return (query_text, skip_reason). text is None when we should skip.

    Last-seen questions have no embedded text query — they're a GPS-only
    PSM path, which the retrieval baselines don't have. We record the
    question with no preds (zero IoU) so it still appears in the output
    but doesn't contribute a real comparison.
    """
    query_mode = q.get("query_mode", "similarity_search")
    if query_mode == "last_seen":
        return None, "last_seen (no text query for baselines)"
    text = q.get("query")
    if not text:
        return None, "missing query"
    return text, None
