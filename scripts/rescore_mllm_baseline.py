#!/usr/bin/env python3
"""Re-score existing vanilla-MLLM baseline JSONs in place, offline.

The scoring semantics in eval_mllm_baseline.py changed in 658d5a9
(2026-07-08): exemplar Hit is now OR'd over all K candidates (was Hit@1
on the MLLM's single pick), and the bucket-IoU window lost its
`max(0.0, ...)` t_min clamp — both to match the shared scorer used by
eval_lookback.py / _eval_common. The baseline JSONs under
captures/mllm_baseline were produced *before* that change, so their
`summary` + per-prediction fields reflect the old Hit@1 scoring.

That change is purely in local scoring: it does not touch which frames
are sampled or the prompt sent to the MLLM. The MLLM's pick
(`mllm_pick_idx`) and every candidate's `exemplar_t` + ground-truth
intervals are already stored in each record, so the new numbers are
fully recomputable with zero API calls — deterministically, and without
the pick-to-pick noise a live re-run would introduce.

This script recomputes `predictions[]`, `exemplar_hit_idx`, and
`summary` for each record using the SAME functions the live evaluator
uses (imported below, not re-implemented) so there is no logic drift,
then rewrites the JSON atomically. Dry-run by default; pass --apply to
write.

    python scripts/rescore_mllm_baseline.py                 # dry-run, all
    python scripts/rescore_mllm_baseline.py --apply         # write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the live scorer verbatim so re-scored numbers == a deterministic
# re-run's numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_mllm_baseline import _best_iou, _point_in_any  # noqa: E402

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "captures" / "mllm_baseline"


def _rescore_record(rec: dict, tol: float) -> dict:
    """Recompute predictions[] + exemplar_hit_idx for one record.

    Mirrors the per-frame loop in eval_mllm_baseline.main(): the stored
    predictions already carry `exemplar_t` in the pick-first order the
    evaluator wrote them, so we only need to recompute the IoUs and the
    OR-over-k hit index. `intervals_gt` is [[a, b], ...].
    """
    gts = [(float(a), float(b)) for a, b in rec.get("intervals_gt", [])]
    new_preds = []
    exemplar_hit_idx = -1
    for p in rec.get("predictions", []):
        t = float(p["exemplar_t"])
        t_min = t - tol            # unclamped, matching current code
        t_max = t + tol
        bucket_iou, _ = _best_iou((t_min, t_max), gts)
        exemplar_iou, _ = _best_iou((t - tol, t + tol), gts)
        in_gt = _point_in_any(t, gts)
        if in_gt >= 0 and exemplar_hit_idx < 0:
            exemplar_hit_idx = in_gt
        new_preds.append({
            "rank": p["rank"],
            "frame_index": p["frame_index"],
            "exemplar_t": t,
            "t_min": t_min,
            "t_max": t_max,
            "bucket_iou": bucket_iou,
            "exemplar_iou": exemplar_iou,
            "in_gt_idx": in_gt,
        })
    rec = dict(rec)
    rec["predictions"] = new_preds
    rec["exemplar_hit_idx"] = exemplar_hit_idx
    return rec


def _summarize(records: list[dict], iou_threshold: float) -> dict:
    """Recompute the doc-level summary exactly as eval_mllm_baseline._flush."""
    n_hit = sum(1 for r in records if r.get("exemplar_hit_idx", -1) >= 0)
    n_scored = sum(1 for r in records if r.get("intervals_gt"))
    bucket_hit = sum(1 for r in records
                     if r.get("intervals_gt") and any(
                         p["bucket_iou"] >= iou_threshold for p in r["predictions"]
                     ))
    exemplar_miou = (sum(max((p["exemplar_iou"] for p in r["predictions"]), default=0.0)
                         for r in records if r.get("intervals_gt"))
                     / max(1, n_scored))
    bucket_miou = (sum(max((p["bucket_iou"] for p in r["predictions"]), default=0.0)
                       for r in records if r.get("intervals_gt"))
                   / max(1, n_scored))
    return {
        "n_questions": len(records),
        "n_scored": n_scored,
        "bucket_miou_at_5": bucket_miou,
        "exemplar_miou_at_5": exemplar_miou,
        "bucket_hit_rate_at_5": bucket_hit / max(1, n_scored),
        "exemplar_hit_rate_at_5": n_hit / max(1, n_scored),
    }


def _rescore_file(path: Path, apply: bool) -> tuple[float, float] | None:
    doc = json.loads(path.read_text())
    records = doc.get("records") or []
    if not records:
        print(f"  SKIP {path.name}: no records")
        return None
    tol = float(doc.get("exemplar_tolerance_sec", 1.5))
    iou_threshold = float(doc.get("iou_threshold", 0.3))

    old_hit = doc.get("summary", {}).get("exemplar_hit_rate_at_5", 0.0) or 0.0
    new_records = [_rescore_record(r, tol) for r in records]
    new_summary = _summarize(new_records, iou_threshold)
    new_hit = new_summary["exemplar_hit_rate_at_5"]

    arrow = "→" if abs(new_hit - old_hit) > 1e-9 else "="
    print(f"  {path.name:52s} Hit@5 {old_hit*100:5.1f}% {arrow} {new_hit*100:5.1f}%")

    if apply:
        doc["records"] = new_records
        doc["summary"] = new_summary
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        tmp.replace(path)
    return old_hit, new_hit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dir", type=Path, default=_DEFAULT_DIR,
                    help="dir of *_gemini.json baseline outputs")
    ap.add_argument("--glob", default="*_gemini.json")
    ap.add_argument("--apply", action="store_true",
                    help="write changes (default: dry-run)")
    args = ap.parse_args()

    files = sorted(args.dir.glob(args.glob))
    if not files:
        print(f"no files matching {args.glob} under {args.dir}", file=sys.stderr)
        return 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[rescore] {mode}: {len(files)} file(s) under {args.dir}\n")
    for p in files:
        _rescore_file(p, args.apply)
    if not args.apply:
        print("\n[rescore] dry-run only — re-run with --apply to write.")
    else:
        print("\n[rescore] wrote updated JSONs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
