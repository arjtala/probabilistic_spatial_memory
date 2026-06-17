#!/usr/bin/env python3
"""Generate questions.yaml for an Aria Gen 2 session via Gemini frame
captions. Aria sibling to scripts/sloper4d_generate_questions.py.

Key differences from the SLOPER4D path:
  - Source is VRS, not MP4. We call the project's existing
    `read_vrs_session` to populate a frame cache (1 fps JPEGs +
    timestamps) once per session, then pick N frames evenly along the
    cache and feed each to the MLLM.
  - Aria walks ship real GPS (track_mode=vrs_gps), so no fake-origin
    surgery; the H5 lat/lng we read for sanity printing is genuine.
  - Otherwise identical: anti-example prompt to push visual diversity,
    incremental flush + resume, atomic write.

Usage:
    python scripts/aria_generate_questions.py \\
        --session-dir /checkpoint/.../aria_gen2_pilot/walk_0 \\
        --features    /checkpoint/.../aria_gen2_pilot/walk_0/clip_l_features.h5 \\
        --out         /checkpoint/.../aria_gen2_pilot/walk_0/questions.yaml \\
        --n-questions 30 --model gemini

Requires GEMINI_API_KEY (or CLAUDE_API_KEY for --model claude).
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "extraction"))

from _mllm_client import Mllm, call_mllm  # noqa: E402
from psm_extraction.io.aria_vrs import read_vrs_session  # noqa: E402


_CAPTION_PROMPT_OUTDOOR = (
    "You are looking at one frame of an egocentric video from a person "
    "walking outdoors. Describe the SINGLE most visually distinctive "
    "feature you can identify in this frame — a specific object (sign, "
    "building, landmark, sculpture, vehicle, person doing something "
    "specific), visible text or numbers, or an unusual viewpoint. Avoid "
    "generic scene words like 'path', 'trees', 'sky', 'grass', 'person "
    "walking', 'cloudy'. One concise sentence, ≤20 words. Start "
    "directly with the description, no preamble."
)

_ANTI_EXAMPLE_SUFFIX = (
    "\n\nIMPORTANT: avoid descriptions that would also fit any of these "
    "earlier frames from the same recording. Pick a DIFFERENT "
    "distinguishing detail.\n"
)


def _jpg_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--session-dir", type=Path, required=True,
                    help="Aria session dir containing video.vrs (or similar)")
    ap.add_argument("--features", type=Path, required=True,
                    help="features.h5 to derive H5 timestamps + session_id from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame-cache", type=Path, default=None,
                    help="dir for the 1 fps frame cache (default: <session>/frames_qg)")
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--interval-half-window-sec", type=float, default=1.5)
    ap.add_argument("--model", choices=["gemini", "claude"], default="gemini")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--anti-example-k", type=int, default=5)
    args = ap.parse_args()

    frame_cache = args.frame_cache or (args.session_dir / "frames_qg")

    # Populate / reuse the 1 fps JPEG cache via the project's VRS reader.
    # This is the same code path the orchestrator uses during extraction
    # with --keep-frames; the manifest check makes re-runs free.
    print(f"[aria-qg] reading VRS frames from {args.session_dir}", file=sys.stderr)
    vrs_out = read_vrs_session(args.session_dir, sample_fps=1.0,
                               output_dir=frame_cache, verbose=True)
    frame_paths = vrs_out.frame_paths
    frame_ts = vrs_out.timestamps_s  # device-clock seconds, monotonic, ≈1Hz spacing
    print(f"[aria-qg] {len(frame_paths)} cached frames span "
          f"{frame_ts[0]:.1f}..{frame_ts[-1]:.1f}s", file=sys.stderr)

    # Load H5 timestamps to keep the questions.yaml intervals
    # consistent with eval_lookback's matching against clip/timestamps.
    with h5py.File(args.features, "r") as h:
        session_id = h.attrs.get("session_id", args.session_dir.name)
        if isinstance(session_id, bytes):
            session_id = session_id.decode()
        g = next((h[k] for k in ("clip", "dino", "jepa") if k in h), None)
        if g is None or "timestamps" not in g:
            raise RuntimeError(f"{args.features} has no embedding group with timestamps")
        h5_ts = g["timestamps"][:].astype(np.float64)
    print(f"[aria-qg] H5 timestamps: n={len(h5_ts)} "
          f"span {h5_ts[0]:.1f}..{h5_ts[-1]:.1f}s", file=sys.stderr)

    # Pick N evenly-spaced indices into the H5 timestamp array; then
    # for each one map to the nearest cached frame.
    n_q = min(args.n_questions, len(h5_ts))
    h5_idx = np.linspace(0, len(h5_ts) - 1, n_q, dtype=np.int64)
    picked_h5_ts = h5_ts[h5_idx]

    # Aria H5 ts and the VRS-cache ts are in the same device clock
    # (read_vrs_session emits device-time-of-capture). So a direct
    # nearest-neighbour map works.
    frame_picks: list[Path] = []
    for t in picked_h5_ts:
        nearest = int(np.argmin(np.abs(frame_ts - t)))
        frame_picks.append(frame_paths[nearest])

    model = Mllm.GEMINI if args.model == "gemini" else Mllm.CLAUDE
    print(f"[aria-qg] {session_id}: captioning {n_q} frames with {model.name}",
          file=sys.stderr)

    # Resume + incremental flush (same shape as the SLOPER4D version).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    questions: list[dict] = []
    if args.out.exists():
        try:
            existing = yaml.safe_load(args.out.read_text()) or {}
            questions = list(existing.get("questions") or [])
            if questions:
                print(f"[aria-qg] resuming: {len(questions)} on disk", file=sys.stderr)
        except Exception as e:
            print(f"[aria-qg] WARN: could not parse existing {args.out}: {e}",
                  file=sys.stderr)

    done_ids = {q.get("id") for q in questions if isinstance(q, dict)}

    def _flush() -> None:
        out_doc = {
            "session_id": session_id,
            "session_start_unix": 0.0,
            "iou_threshold": float(args.iou_threshold),
            "questions": questions,
        }
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with open(tmp, "w") as f:
            yaml.safe_dump(out_doc, f, sort_keys=False, allow_unicode=True)
        tmp.replace(args.out)

    for i, (h5_t, frame_path) in enumerate(zip(picked_h5_ts, frame_picks)):
        qid = f"q{i+1}"
        if qid in done_ids:
            continue
        b64 = _jpg_to_b64(frame_path)
        recent_captions = [
            q["query"] for q in questions[-args.anti_example_k:]
            if isinstance(q, dict) and q.get("query")
        ]
        if recent_captions:
            anti_block = "\n".join(f"- {c}" for c in recent_captions)
            prompt = _CAPTION_PROMPT_OUTDOOR + _ANTI_EXAMPLE_SUFFIX + anti_block
        else:
            prompt = _CAPTION_PROMPT_OUTDOOR
        try:
            caption = call_mllm(model=model, frames_b64=[b64], prompt=prompt).strip()
        except Exception as e:
            print(f"  WARN: caption failed at h5_ts={h5_t:.1f}: {e}", file=sys.stderr)
            continue
        caption = caption.strip().strip('"').strip("'").strip()
        t_lo = max(0.0, float(h5_t) - args.interval_half_window_sec)
        t_hi = float(h5_t) + args.interval_half_window_sec
        questions.append({
            "id": qid,
            "query": caption,
            "intervals": [[round(t_lo, 3), round(t_hi, 3)]],
            "notes": f"auto-captioned via {model.name} at h5_ts={h5_t:.3f}s (frame={frame_path.name})",
        })
        _flush()
        print(f"  q{i+1} (h5_ts={h5_t:.1f}s): {caption[:80]}", file=sys.stderr)

    print(f"[aria-qg] wrote {len(questions)} questions to {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
