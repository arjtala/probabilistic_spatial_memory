#!/usr/bin/env python3
"""PSM -> MLLM re-ranking eval harness for look-back retrieval.

Pipeline per question:
  1. PSM --search returns top-k (cell, t_min, t_max, exemplar_t).
  2. For each top-k candidate, fetch the cached JPEG nearest exemplar_t
     from <features_dir>/frames/ (must exist; the extractor's
     --keep-frames flag preserves it).
  3. Send the k frames + question + frozen prompt to the MLLM.
  4. MLLM returns the 1-based index of the best-matching frame; we
     resolve back to (t_min, t_max, exemplar_t) of that candidate.
  5. Emit a per-question record in the SAME shape eval_lookback.py
     produces, so eval_aggregate.py can pool MLLM runs alongside
     PSM runs without extra plumbing. The MLLM's choice becomes the
     `top1` prediction; the remaining PSM candidates fill the rest of
     the top-k so the @k metrics still compute. (Otherwise a wrong
     MLLM pick would zero the @k score, which would unfairly compare
     PSM->MLLM to PSM-only at top-1 only.)

Usage:
  python scripts/eval_psm_mllm.py \\
    /path/to/features.h5 /path/to/questions.yaml \\
    --frames-dir /path/to/frames \\
    --mllm gemini \\
    --top 5 \\
    --out captures/eval_<sid>_psm_mllm_<model>.json

Env vars: GEMINI_API_KEY for --mllm gemini, CLAUDE_API_KEY for --mllm claude.

Resume-on-rerun: writes a .partial.json next to --out after every question
so a long sweep can pick up where it left off if interrupted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "extraction"))

# Reuse PSM-search shell-out + IoU math from eval_lookback.
from scripts._mllm_client import Mllm, call_mllm, encode_frame, smoke_test  # noqa: E402
from scripts.eval_lookback import (  # noqa: E402
    auto_session_start,
    best_iou,
    point_in_any,
    run_psm_search,
)


# Frozen system prompt — keep stable across runs so results are comparable.
# The MLLM is asked to pick a frame INDEX, not to write a free-form answer,
# because we need a discrete temporal prediction the IoU metrics can score.
# Tradeoff: this loses the MLLM's free-text reasoning. For v1 that's
# acceptable — the paper claim is "PSM narrows the search space, MLLM
# adjudicates"; the eval metric is whether the adjudication picks the
# correct moment, not whether the MLLM can also describe it.
_FROZEN_PROMPT_TEMPLATE = """You are shown {k} candidate frames from a wearable camera recording, numbered 1 to {k}. They are PSM-retrieved candidates for the following look-back question:

  "{question}"

Pick the SINGLE frame number that best answers the question. Respond with just the number (e.g. "3"). If none of the frames clearly answer the question, respond with the number of the frame you find most plausible — do not refuse to choose."""


@dataclasses.dataclass
class _Candidate:
    """One PSM top-k candidate, post-processed for MLLM input."""
    t_min: float          # session-relative seconds
    t_max: float
    exemplar_t: float
    cell: object | None
    lat: float | None
    lng: float | None
    similarity: float | None
    count: float
    frame_path: Path      # the JPEG nearest exemplar_t


def _find_cached_frame(frames_dir: Path, target_t_s: float, sample_fps: float) -> Path:
    """Return the cached frame_<idx>.jpg closest to target_t_s.

    Extractor writes frames at uniform 1/sample_fps spacing, starting at
    t=0 (or device-clock-rebased to t=0 for VRS). The nearest-frame index
    is therefore round(target_t / dt). Out-of-range targets are clamped
    to the first/last frame so MLLM gets *something* even when PSM
    returns a candidate exemplar that overshoots the recording boundary
    (rare, but possible at session start/end).
    """
    dt = 1.0 / sample_fps
    idx = int(round(max(0.0, target_t_s) / dt))
    # Glob-validate: depending on the extractor's sampling-gate logic the
    # frame count can be slightly less than target_t / dt at the tail.
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not all_frames:
        raise SystemExit(
            f"no frames under {frames_dir}; did the extractor run with --keep-frames?"
        )
    idx = min(idx, len(all_frames) - 1)
    return all_frames[idx]


def _parse_index(text: str, k: int) -> int | None:
    """Parse '1'..'k' out of the MLLM response. None on garbage."""
    # The frozen prompt asks for a bare number, but reasoning models sometimes
    # prefix with "The answer is 3." or wrap in markdown. Scan for the first
    # int that lands in [1, k].
    import re
    for m in re.finditer(r"\b(\d+)\b", text):
        v = int(m.group(1))
        if 1 <= v <= k:
            return v
    return None


def _resolve_features_to_frames_dir(features_h5: Path, override: Path | None) -> Path:
    """Default to <features_h5.parent>/frames/ unless explicitly overridden.

    Matches extract.py's default frames_dir layout when --keep-frames is
    set. Lets the eval harness usually be invoked with no extra flags.
    """
    if override is not None:
        return override
    return features_h5.parent / "frames"


def _make_record(
    *,
    qid: str,
    text: str,
    category: str,
    notes: str,
    gts_rel: list[tuple[float, float]],
    candidates: list[_Candidate],
    mllm_pick_idx: int | None,
    mllm_raw: str,
    iou_threshold: float,
    exemplar_tolerance: float,
) -> dict:
    """Build a per-question record in eval_lookback's schema.

    Reorders candidates so the MLLM-picked one is at position 0 of
    `preds`. If the MLLM gave no parseable pick, we keep PSM's original
    top-1 (this is the right fallback: the harness still measures
    PSM->MLLM end-to-end, but it doesn't punish PSM for the MLLM's
    refusal — we surface refusal separately as `mllm_refused=True`).
    """
    if mllm_pick_idx is not None and 1 <= mllm_pick_idx <= len(candidates):
        chosen = candidates[mllm_pick_idx - 1]
        rest = candidates[:mllm_pick_idx - 1] + candidates[mllm_pick_idx:]
        ordered = [chosen, *rest]
        mllm_refused = False
    else:
        ordered = candidates
        mllm_refused = True

    preds: list[dict] = []
    for c in ordered:
        bucket_iou, gt_idx_b = best_iou((c.t_min, c.t_max), gts_rel)
        exemplar_iou, gt_idx_e = best_iou(
            (c.exemplar_t - exemplar_tolerance, c.exemplar_t + exemplar_tolerance),
            gts_rel,
        )
        exemplar_hit_idx = point_in_any(c.exemplar_t, gts_rel)
        preds.append({
            "cell": c.cell,
            "lat": c.lat,
            "lng": c.lng,
            "similarity": c.similarity,
            "exemplar_t": c.exemplar_t,
            "t_min": c.t_min,
            "t_max": c.t_max,
            "count": c.count,
            "bucket_iou": bucket_iou,
            "bucket_matched_gt": gt_idx_b,
            "exemplar_iou": exemplar_iou,
            "exemplar_matched_gt": gt_idx_e,
            "exemplar_hits_gt": exemplar_hit_idx >= 0,
        })

    top1 = preds[0] if preds else None
    bucket_top1 = top1["bucket_iou"] if top1 else 0.0
    bucket_at_k = max((p["bucket_iou"] for p in preds), default=0.0)
    exemplar_top1 = top1["exemplar_iou"] if top1 else 0.0
    exemplar_at_k = max((p["exemplar_iou"] for p in preds), default=0.0)
    any_exemplar_hit = any(p["exemplar_hits_gt"] for p in preds)

    return {
        "id": qid,
        "query": text,
        "query_mode": "psm_mllm_rerank",
        "category": category,
        "notes": notes,
        "intervals_gt": gts_rel,
        "count_gt": None,
        "count_predicted": len({p["cell"] for p in preds}) if preds else 0,
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
        # MLLM-specific provenance (ignored by eval_aggregate, useful for debugging):
        "mllm_pick_idx": mllm_pick_idx,
        "mllm_raw_response": mllm_raw,
        "mllm_refused": mllm_refused,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", type=Path)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--group", default="clip")
    ap.add_argument("--mllm", choices=("gemini", "claude"), default="gemini")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="JPEG cache from extractor (default: <features>/../frames)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--time-window", type=float, default=75.0)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--h3-resolution", type=int, default=10)
    ap.add_argument("--precision", type=int, default=14)
    ap.add_argument("--exemplars", type=int, default=128)
    ap.add_argument("--exemplar-codec", default="raw")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--exemplar-tolerance", type=float, default=1.5)
    ap.add_argument("--clip-checkpoint",
                    default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
    ap.add_argument("--clip-device", default="cuda")
    ap.add_argument("--psm-binary", type=Path,
                    default=REPO_ROOT / "targets" / "psm")
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="Must match the extractor's sample_fps so frame "
                         "indices align with exemplar_t (default: 1.0)")
    ap.add_argument("--frame-max-size", type=int, default=768)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rate-limit-s", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    mllm = Mllm.GEMINI if args.mllm == "gemini" else Mllm.CLAUDE

    # Fail fast on missing key / proxy issues — would otherwise burn a few
    # PSM searches before the first MLLM call surfaces the bad config.
    print(f"[eval] smoke-testing {mllm.name}...", file=sys.stderr)
    try:
        ok = smoke_test(mllm)
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] FATAL: {mllm.name} smoke test failed: {exc}", file=sys.stderr)
        return 1
    print(f"[eval] {mllm.name} OK: {ok!r}", file=sys.stderr)

    frames_dir = _resolve_features_to_frames_dir(args.features, args.frames_dir)
    if not frames_dir.exists():
        print(f"[eval] FATAL: frames dir {frames_dir} does not exist. "
              "Re-run extractor with --keep-frames.", file=sys.stderr)
        return 1

    with args.questions.open() as f:
        spec = yaml.safe_load(f)
    questions = spec.get("questions") or []
    session_start = float(spec.get("session_start_unix", 0.0))
    if "session_start_unix" not in spec:
        session_start = auto_session_start(args.features, args.group)
    iou_threshold = float(spec.get("iou_threshold", 0.3))

    # CLIP runner for text -> query vector (PSM needs a float32 query file).
    from psm_extraction.models import make_runner
    runner = make_runner("clip", checkpoint=args.clip_checkpoint,
                         backend="auto", device=args.clip_device)

    # Resume support.
    partial_path = args.out.with_suffix(args.out.suffix + ".partial.json")
    completed: dict[str, dict] = {}
    if partial_path.exists():
        try:
            completed = json.loads(partial_path.read_text())
            print(f"[eval] resuming from {len(completed)} cached questions",
                  file=sys.stderr)
        except json.JSONDecodeError:
            print(f"[eval] WARN: ignoring corrupt {partial_path}", file=sys.stderr)
            completed = {}

    records: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="psm-mllm-eval-"))
    try:
        for q in questions:
            qid = q.get("id") or f"q{len(records) + 1}"
            text = q.get("query")
            if not text:
                raise SystemExit(f"question {qid!r} missing 'query'")

            if qid in completed:
                records.append(completed[qid])
                continue

            gts_rel = [
                (float(iv[0]), float(iv[1])) for iv in q.get("intervals", [])
            ]

            # 1. PSM --search for top-k candidates.
            qvec = runner.embed_text(text).astype(np.float32)
            qpath = tmp_dir / f"{qid}.f32"
            qpath.write_bytes(qvec.tobytes())
            payload = run_psm_search(
                args.psm_binary, args.features, args.group, qpath,
                top=args.top,
                time_window=args.time_window,
                capacity=args.capacity,
                h3_resolution=args.h3_resolution,
                precision=args.precision,
                exemplars=args.exemplars,
                seed=(None if args.seed < 0 else args.seed),
                exemplar_codec=args.exemplar_codec,
                verbose=args.verbose,
            )
            results = payload.get("results", [])

            # 2. Resolve each result to its nearest cached frame.
            candidates: list[_Candidate] = []
            for r in results:
                t_min = float(r["t_min"]) - session_start
                t_max = float(r["t_max"]) - session_start
                exemplar_t = float(r.get("exemplar_t", (t_min + t_max) / 2.0)) - session_start
                frame_path = _find_cached_frame(frames_dir, exemplar_t, args.sample_fps)
                candidates.append(_Candidate(
                    t_min=t_min, t_max=t_max, exemplar_t=exemplar_t,
                    cell=r.get("cell"), lat=r.get("lat"), lng=r.get("lng"),
                    similarity=r.get("similarity"), count=r.get("count", 0),
                    frame_path=frame_path,
                ))

            if not candidates:
                # PSM returned nothing — still emit a record so the question
                # count matches between PSM-only and PSM->MLLM runs.
                record = _make_record(
                    qid=qid, text=text,
                    category=q.get("category", "(uncategorized)") or "(uncategorized)",
                    notes=q.get("notes", ""),
                    gts_rel=gts_rel,
                    candidates=[],
                    mllm_pick_idx=None,
                    mllm_raw="",
                    iou_threshold=iou_threshold,
                    exemplar_tolerance=args.exemplar_tolerance,
                )
            else:
                # 3. MLLM rerank.
                frames_b64 = [encode_frame(c.frame_path, max_size=args.frame_max_size)
                              for c in candidates]
                prompt = _FROZEN_PROMPT_TEMPLATE.format(k=len(candidates), question=text)
                try:
                    raw = call_mllm(
                        model=mllm,
                        frames_b64=frames_b64,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[eval] {qid}: MLLM error: {exc}", file=sys.stderr)
                    raw = ""
                pick = _parse_index(raw, len(candidates))
                if args.verbose:
                    print(f"[eval] {qid}: MLLM pick={pick} raw={raw!r}",
                          file=sys.stderr)
                record = _make_record(
                    qid=qid, text=text,
                    category=q.get("category", "(uncategorized)") or "(uncategorized)",
                    notes=q.get("notes", ""),
                    gts_rel=gts_rel,
                    candidates=candidates,
                    mllm_pick_idx=pick,
                    mllm_raw=raw,
                    iou_threshold=iou_threshold,
                    exemplar_tolerance=args.exemplar_tolerance,
                )

            records.append(record)
            completed[qid] = record
            # Checkpoint after every question; harness can crash and resume.
            partial_path.write_text(json.dumps(completed))

            if args.rate_limit_s > 0:
                import time as _t
                _t.sleep(args.rate_limit_s)
    finally:
        if not args.verbose:
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()
        runner.close()

    # Final output. Same shape as eval_lookback so eval_aggregate pools it.
    scored = [r for r in records if r["intervals_gt"]]
    bucket_at_k = float(np.mean([r["bucket_iou_at_k"] for r in scored])) if scored else 0.0
    exemplar_at_k = float(np.mean([r["exemplar_iou_at_k"] for r in scored])) if scored else 0.0
    bucket_hit_at_k = float(np.mean([r["bucket_hit_at_k"] for r in scored])) if scored else 0.0
    exemplar_hit_at_k = float(np.mean([r["exemplar_hit_at_k"] for r in scored])) if scored else 0.0

    out_doc = {
        "features": str(args.features),
        "questions": str(args.questions),
        "group": args.group,
        "mllm": mllm.name,
        "mllm_model_id": mllm.model_id,
        "top": args.top,
        "n_questions": len(records),
        "n_scored": len(scored),
        "summary": {
            "bucket_mIoU_at_k": bucket_at_k,
            "exemplar_mIoU_at_k": exemplar_at_k,
            "bucket_hit_at_k": bucket_hit_at_k,
            "exemplar_hit_at_k": exemplar_hit_at_k,
        },
        "questions_out": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"[eval] wrote {args.out} (bucket@k={bucket_at_k:.3f}, "
          f"exemplar@k={exemplar_at_k:.3f}, hit@k={exemplar_hit_at_k:.3f})",
          file=sys.stderr)

    # Clean up partial on success.
    if partial_path.exists():
        partial_path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
