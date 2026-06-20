#!/usr/bin/env python3
"""Vanilla-MLLM baseline (paper item 3 / E10).

For each question, sample K frames uniformly across the entire video
and ask the MLLM "which frame best answers this?" — no PSM in the
loop. The MLLM picks an index; we convert that to a timestamp
window and score with the same exemplar/bucket IoU + Hit@k metrics
as eval_lookback.py + eval_psm_mllm.py, so numbers go side-by-side
in the §5 table.

This is the "frontier MLLM as standalone retrieval" baseline. PSM's
headline claim is that it matches this number at bounded memory +
query cost. Without this baseline, PSM's numbers have nothing to
compare against on the street-scale corpora.

Frame sourcing: same FrameSource abstraction as eval_psm_mllm.py
(cached frames dir, MP4 via ffmpeg, or VRS via the project's reader).

Usage:
    python scripts/eval_mllm_baseline.py \\
        --features /path/to/clip_l_features.h5 \\
        --questions /path/to/questions.yaml \\
        --video /path/to/video.mp4 \\
        --out captures/mllm_baseline_<sid>_<model>.json \\
        --k-frames 8 \\
        --model gemini

Requires GEMINI_API_KEY (default) or CLAUDE_API_KEY.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "extraction"))

from _mllm_client import Mllm, call_mllm  # noqa: E402


# Same frozen-prompt shape as eval_psm_mllm.py so numbers are
# directly comparable. The only change is the candidate provenance
# (uniform sample of all frames, not PSM top-k).
_FROZEN_PROMPT_TEMPLATE = """You are shown {k} candidate frames from a wearable camera recording, sampled uniformly across the full session. They are numbered 1 to {k} in temporal order. The recording is from a person walking outdoors.

Look-back question:

  "{question}"

Pick the SINGLE frame number that best answers the question. Respond with just the number (e.g. "3"). If none of the frames clearly answer the question, respond with the number of the frame you find most plausible — do not refuse to choose."""


def _parse_index(text: str, k: int) -> int | None:
    """Parse 1..k out of a possibly-prefixed MLLM response."""
    import re
    for m in re.finditer(r"\b(\d+)\b", text):
        v = int(m.group(1))
        if 1 <= v <= k:
            return v
    return None


def _best_iou(pred: tuple[float, float], gts: list[tuple[float, float]]) -> tuple[float, int]:
    """Return (best_iou, gt_index) of pred vs the GT interval list."""
    best = 0.0
    best_idx = -1
    p0, p1 = pred
    for i, (g0, g1) in enumerate(gts):
        inter = max(0.0, min(p1, g1) - max(p0, g0))
        if inter <= 0:
            continue
        union = max(p1, g1) - min(p0, g0)
        iou = inter / union if union > 0 else 0.0
        if iou > best:
            best = iou
            best_idx = i
    return best, best_idx


def _point_in_any(t: float, gts: list[tuple[float, float]]) -> int:
    """Return GT index containing t, or -1."""
    for i, (g0, g1) in enumerate(gts):
        if g0 <= t <= g1:
            return i
    return -1


# --- Frame source: ffmpeg single-pass uniform sample ---------------------


def _pick_uniform_from_cache(
    frames_dir: Path,
    n_frames: int,
    h5_ts: np.ndarray,
) -> list[tuple[float, Path]]:
    """Pick `n_frames` evenly-spaced frames from an existing JPEG cache.

    The orchestrator (and our SLOPER4D / Aria extractors) drops 1 fps
    JPEGs into `<session>/frames*/frame_*.jpg` when called with
    --keep-frames or via VRS reader. We just need to align the JPEG
    list with the H5 timestamps and pick N evenly-spaced indices.

    Returns (h5_clock_ts, path) tuples. Frame ts are inferred to be
    the H5 timestamps at the same index — both the extractor's
    JPEGs and the H5 are produced together at the same sample_fps,
    so frame_i ↔ ts_i is the natural mapping.
    """
def _pick_uniform_from_cache(
    frames_dir: Path,
    n_frames: int,
    h5_ts: np.ndarray,
) -> list[tuple[float, Path]]:
    """Pick `n_frames` evenly-spaced frames from an existing JPEG/PNG cache.

    Two layouts handled:

      1. **LookOut layout** (rgb_info.pkl exists at frames_dir.parent):
         the frames dir holds the FULL 20-fps PNGs (e.g. 12,824 frames
         for Mainquad_jan10), but the H5 only has 624 entries because
         the extractor subsampled to 1 fps. We redo the same greedy
         1-fps subsample of rgb_info.pkl's timestamps so the kept PNGs
         align 1:1 with the H5 timestamps before picking N from those.
         Same logic as scripts/aria_generate_questions.py.

      2. **Aria/orchestrator layout** (no rgb_info.pkl): the frames
         dir holds frames already at H5 cadence. Sorted-list position
         maps 1:1 to H5 timestamps.

    Returns (h5_clock_ts, path) tuples for N evenly-spaced selections.
    """
    # LookOut layout detection: rgb_info.pkl sibling to the frames dir.
    rgb_info_path = frames_dir.parent / "rgb_info.pkl"
    if rgb_info_path.exists():
        import pickle
        with rgb_info_path.open("rb") as f:
            rgb_info = pickle.load(f)
        full_ts_ns = np.array([int(r[1]) for r in rgb_info], dtype=np.int64)
        full_ts_s = (full_ts_ns - full_ts_ns[0]) / 1e9
        target_fps = 1.0
        period = 1.0 / target_fps
        kept_idx = [0]
        next_t = float(full_ts_s[0]) + period
        for i in range(1, len(full_ts_s)):
            if float(full_ts_s[i]) >= next_t:
                kept_idx.append(i)
                next_t = float(full_ts_s[i]) + period
        all_frames = [
            frames_dir / f"{idx}_undistorted_512_243.png" for idx in kept_idx
        ]
        # Use H5 timestamps directly (they should match what kept_idx
        # produces up to a few-frame off-by-one at the tail).
        # Truncate to min length defensively.
        n_common = min(len(all_frames), len(h5_ts))
        all_frames = all_frames[:n_common]
        h5_ts = h5_ts[:n_common]
    else:
        # Aria/orchestrator layout: any extension, sorted by name.
        all_frames = (
            sorted(frames_dir.glob("frame_*.jpg"))
            or sorted(frames_dir.glob("frame_*.png"))
            or sorted(frames_dir.glob("*.jpg"))
            or sorted(frames_dir.glob("*.png"))
        )
        if not all_frames:
            raise SystemExit(f"no JPEG/PNG frames found under {frames_dir}")
        if len(all_frames) != len(h5_ts):
            n = min(len(all_frames), len(h5_ts))
            print(
                f"[mllm-baseline] WARN: {len(all_frames)} frames vs "
                f"{len(h5_ts)} H5 ts; truncating to {n}",
                file=sys.stderr,
            )
            all_frames = all_frames[:n]
            h5_ts = h5_ts[:n]

    if not all_frames:
        raise SystemExit(f"no frames available under {frames_dir}")
    if n_frames > len(all_frames):
        n_frames = len(all_frames)
    idx = np.linspace(0, len(all_frames) - 1, n_frames, dtype=np.int64)
    return [(float(h5_ts[i]), all_frames[i]) for i in idx]


def _decode_frames_uniform(
    video_path: Path,
    n_frames: int,
    duration_sec: float,
    out_dir: Path,
) -> list[tuple[float, Path]]:
    """Decode `n_frames` evenly-spaced frames from `video_path` via one
    ffmpeg pass using the `fps` filter. Returns (presentation_ts, path)
    tuples. Reused across all questions for a sequence — same frame set
    every time, so the cache is hit after the first call.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) == n_frames:
        # Recompute ts assuming the same uniform sampling.
        ts = np.linspace(duration_sec / (2 * n_frames),
                         duration_sec - duration_sec / (2 * n_frames),
                         n_frames)
        return list(zip(ts, existing))

    # Wipe and re-decode if count doesn't match (param change).
    for p in existing:
        p.unlink()

    # fps = n / duration; pixel-format guard for HEVC/yuv tv-range.
    fps = float(n_frames) / float(duration_sec)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-pix_fmt", "yuvj420p",
        "-q:v", "2",
        pattern,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: rc={r.returncode}\n{r.stderr[-1500:]}")
    paths = sorted(out_dir.glob("frame_*.jpg"))
    if not paths:
        raise RuntimeError(f"ffmpeg produced no frames into {out_dir}")
    # Trim to exactly n_frames if ffmpeg rounding produced one extra.
    paths = paths[:n_frames]
    ts = np.linspace(duration_sec / (2 * n_frames),
                     duration_sec - duration_sec / (2 * n_frames),
                     len(paths))
    return list(zip(ts, paths))


def _jpg_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


# --- Driver ---------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--features", type=Path, required=True,
                    help="features.h5 for the session (timestamps + session_id)")
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None,
                    help="source MP4 (decode frames via ffmpeg). One of "
                         "--video or --frames-dir is required.")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="dir of pre-extracted JPEGs (from extraction "
                         "--keep-frames or our SLOPER4D/Aria pipelines). "
                         "Frames are aligned 1:1 with the H5 timestamps; "
                         "we pick K evenly-spaced indices. Cheaper + works "
                         "for VRS-source sessions where no MP4 exists.")
    ap.add_argument("--vrs-session-dir", type=Path, default=None,
                    help="Aria session dir containing a VRS file (video.vrs "
                         "or recording_head/data/data.vrs). Triggers a 1 fps "
                         "JPEG cache via the project's VRS reader, then "
                         "K evenly-spaced picks. Useful for Nymeria where "
                         "no MP4 exists and frames weren't kept during the "
                         "original extraction.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-frames", type=int, default=8,
                    help="number of frames sampled per question (default 8). "
                         "MLLM context limits + cost trade-off; 8 matches "
                         "eval_psm_mllm.py's default candidate count.")
    ap.add_argument("--exemplar-tolerance-sec", type=float, default=1.5)
    ap.add_argument("--model", choices=["gemini", "claude"], default="gemini")
    ap.add_argument("--frame-cache", type=Path, default=None,
                    help="dir for the decoded JPEGs (default: <out_dir>/frames_baseline)")
    args = ap.parse_args()

    # Load H5 for session_id + duration.
    with h5py.File(args.features, "r") as h:
        session_id = h.attrs.get("session_id", args.features.parent.name)
        if isinstance(session_id, bytes):
            session_id = session_id.decode()
        g = next((h[k] for k in ("clip", "dino", "jepa") if k in h), None)
        if g is None or "timestamps" not in g:
            raise SystemExit(f"{args.features} has no embedding group with timestamps")
        h5_ts = g["timestamps"][:].astype(np.float64)

    duration = float(h5_ts[-1] - h5_ts[0])
    ts_offset = float(h5_ts[0])  # how much to shift video-clock back into H5 clock for GT comparison
    print(f"[mllm-baseline] {session_id}: H5 ts span [{h5_ts[0]:.1f}, {h5_ts[-1]:.1f}], "
          f"duration={duration:.1f}s, ts_offset={ts_offset:.1f}", file=sys.stderr)

    questions_doc = yaml.safe_load(args.questions.read_text())
    questions = questions_doc.get("questions", [])
    iou_threshold = float(questions_doc.get("iou_threshold", 0.3))
    session_start_unix = float(questions_doc.get("session_start_unix", 0.0))
    print(f"[mllm-baseline] {len(questions)} questions; iou_threshold={iou_threshold}",
          file=sys.stderr)

    # Decode the uniform-sample frame set once. Same K frames for every
    # question — that's the whole point of "vanilla MLLM with no PSM
    # prefilter": the MLLM sees the same candidate set every time.
    if args.frames_dir is not None:
        # Pre-extracted JPEGs path: align with the H5 timestamps,
        # pick K evenly-spaced. Cheapest path; works for VRS-source
        # sessions where no MP4 exists.
        decoded = _pick_uniform_from_cache(args.frames_dir, args.k_frames, h5_ts)
        frame_ts_h5 = np.array([t for t, _ in decoded], dtype=np.float64)
    elif args.vrs_session_dir is not None:
        # VRS path: call the project's VRS reader to populate a 1 fps
        # JPEG cache once, then pick K evenly-spaced from it. Cache is
        # reused across runs.
        from psm_extraction.io.aria_vrs import read_vrs_session  # noqa
        frame_cache = args.frame_cache or (args.out.parent / "frames_baseline" / session_id)
        vrs_out = read_vrs_session(args.vrs_session_dir, sample_fps=1.0,
                                   output_dir=frame_cache, verbose=False)
        # Align extracted JPEGs to H5 ts (parallel construction: both
        # are emitted from the same VRS frames at the same 1 fps).
        vrs_h5 = vrs_out.timestamps_s.astype(np.float64)
        decoded = _pick_uniform_from_cache(frame_cache, args.k_frames, vrs_h5)
        frame_ts_h5 = np.array([t for t, _ in decoded], dtype=np.float64)
    elif args.video is not None:
        # MP4 path: ffmpeg single-pass decode at K/duration fps.
        frame_cache = args.frame_cache or (args.out.parent / "frames_baseline" / session_id)
        decoded = _decode_frames_uniform(args.video, args.k_frames, duration, frame_cache)
        # Frame ts are in video clock (0..duration); shift to H5 clock for
        # consistent IoU vs GT (which is in H5 / questions.yaml clock).
        frame_ts_h5 = np.array([t + ts_offset for t, _ in decoded], dtype=np.float64)
    else:
        raise SystemExit("one of --video, --frames-dir, or --vrs-session-dir is required")
    frame_paths = [p for _, p in decoded]
    print(f"[mllm-baseline] uniform-sampled K={args.k_frames} frames at H5 ts "
          f"{[f'{t:.1f}' for t in frame_ts_h5]}", file=sys.stderr)

    # Pre-encode the frames once (Gemini accepts them at the per-call
    # level so we'd re-send anyway, but caching b64 strings saves I/O).
    frames_b64 = [_jpg_to_b64(p) for p in frame_paths]

    model = Mllm.GEMINI if args.model == "gemini" else Mllm.CLAUDE

    # Stream records to disk after every question to survive cancellation.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def _flush() -> None:
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
        summary = {
            "n_questions": len(records),
            "n_scored": n_scored,
            "bucket_miou_at_5": bucket_miou,
            "exemplar_miou_at_5": exemplar_miou,
            "bucket_hit_rate_at_5": bucket_hit / max(1, n_scored),
            "exemplar_hit_rate_at_5": n_hit / max(1, n_scored),
        }
        doc = {
            "session_id": session_id,
            "features": str(args.features),
            "questions_file": str(args.questions),
            "video": str(args.video),
            "k_frames": args.k_frames,
            "iou_threshold": iou_threshold,
            "exemplar_tolerance_sec": args.exemplar_tolerance_sec,
            "mllm_model": model.name,
            "query_mode": "mllm_baseline",
            "session_start_unix": session_start_unix,
            "summary": summary,
            "records": records,
        }
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        tmp.replace(args.out)

    # Resume: skip questions whose ids are already in records on disk.
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text())
            prev_records = prev.get("records") or []
            done_ids = {r.get("question_id") for r in prev_records if isinstance(r, dict)}
            records = list(prev_records)
            if done_ids:
                print(f"[mllm-baseline] resuming: {len(done_ids)} questions already on disk",
                      file=sys.stderr)
        except Exception as e:
            print(f"[mllm-baseline] WARN: could not parse {args.out}: {e}", file=sys.stderr)
            records = []
    done_ids = {r.get("question_id") for r in records if isinstance(r, dict)}

    for i, q in enumerate(questions):
        qid = q.get("id", f"q{i+1}")
        if qid in done_ids:
            continue
        text = q.get("query", "")
        gts: list[tuple[float, float]] = [
            (float(a), float(b)) for a, b in q.get("intervals", [])
        ]

        prompt = _FROZEN_PROMPT_TEMPLATE.format(k=args.k_frames, question=text)
        try:
            resp = call_mllm(model=model, frames_b64=frames_b64, prompt=prompt).strip()
        except Exception as e:
            print(f"  {qid} MLLM error: {e}", file=sys.stderr)
            continue

        pick = _parse_index(resp, args.k_frames)
        if pick is None:
            print(f"  {qid} unparseable response: {resp[:200]!r}", file=sys.stderr)
            mllm_refused = True
            pick = 1  # arbitrary fallback for scoring
        else:
            mllm_refused = False

        chosen_t = float(frame_ts_h5[pick - 1])
        # Reorder so chosen is first, then the rest in their original order.
        order = [pick - 1] + [j for j in range(args.k_frames) if j != pick - 1]

        predictions: list[dict] = []
        exemplar_hit_idx = -1
        for rank, j in enumerate(order):
            t = float(frame_ts_h5[j])
            t_min = max(0.0, t - args.exemplar_tolerance_sec)
            t_max = t + args.exemplar_tolerance_sec
            bucket_iou, _ = _best_iou((t_min, t_max), gts)
            exemplar_iou, _ = _best_iou(
                (t - args.exemplar_tolerance_sec, t + args.exemplar_tolerance_sec),
                gts,
            )
            in_gt = _point_in_any(t, gts)
            if rank == 0:
                exemplar_hit_idx = in_gt
            predictions.append({
                "rank": rank + 1,
                "frame_index": int(j) + 1,
                "exemplar_t": t,
                "t_min": t_min,
                "t_max": t_max,
                "bucket_iou": bucket_iou,
                "exemplar_iou": exemplar_iou,
                "in_gt_idx": in_gt,
            })

        records.append({
            "question_id": qid,
            "query": text,
            "intervals_gt": gts,
            "mllm_response": resp,
            "mllm_pick_idx": pick,
            "mllm_refused": mllm_refused,
            "exemplar_t": chosen_t,
            "exemplar_hit_idx": exemplar_hit_idx,
            "predictions": predictions,
        })
        _flush()
        hit_mark = "✓" if exemplar_hit_idx >= 0 else "·"
        print(f"  {qid} [{i+1}/{len(questions)}] pick=#{pick} ({chosen_t:.1f}s) {hit_mark}  "
              f"q='{text[:60]}'", file=sys.stderr)

    _flush()
    # Final summary line.
    n_scored = sum(1 for r in records if r.get("intervals_gt"))
    n_hit = sum(1 for r in records if r.get("exemplar_hit_idx", -1) >= 0)
    print(f"\n[mllm-baseline] done. {n_hit}/{n_scored} exemplar-hits "
          f"({100*n_hit/max(1,n_scored):.1f}%) ", file=sys.stderr)
    print(f"[mllm-baseline] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
