#!/usr/bin/env python3
"""Generate questions.yaml for a SLOPER4D sequence via MLLM frame captions.

Picks N frames evenly spaced along the trajectory, captions each with
Gemini 3.1 Pro (or Claude 4.6 Opus), and writes them as
``query_mode: similarity_search`` questions with a small interval
window. The resulting questions.yaml matches the schema the existing
eval pipeline already consumes — drop-in compatible with
eval_lookback.py, the H3-resolution sweep, baselines, everything.

Usage:
    python scripts/sloper4d_generate_questions.py \\
        --features /checkpoint/.../sloper4d/seq009_running_002/clip_l_features.h5 \\
        --video    /checkpoint/.../SLOPER4D-unzipped/seq009_running_002/rgb_data/seq009_running_002.MP4 \\
        --out      /checkpoint/.../sloper4d/seq009_running_002/questions.yaml \\
        --n-questions 20 \\
        --model gemini

The MLLM cost is ~20 calls × ~1 KB image × short response =
sub-penny per sequence. Free in practice via the internal proxy.
"""
from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml


_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
from _mllm_client import Mllm, call_mllm  # noqa: E402


_CAPTION_PROMPT = (
    "You are looking at one frame of an egocentric video taken with a "
    "head-mounted camera. The wearer is walking or running outdoors at a "
    "university campus. Describe what the wearer can SEE in this frame in "
    "ONE concise sentence (≤20 words). Focus on visible objects, "
    "landmarks, and the scene type. Do not include preamble; start "
    "directly with the description."
)


def _decode_frame_at_timestamp(
    mp4_path: Path, ts_sec: float, out_jpg: Path
) -> None:
    """One-shot ffmpeg frame extract. Slow seek (post-input -ss) for
    accuracy. Output goes to `out_jpg`."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(mp4_path),
        "-ss", f"{ts_sec:.3f}",
        "-frames:v", "1",
        "-pix_fmt", "yuvj420p",
        "-q:v", "2",
        str(out_jpg),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_jpg.exists():
        raise RuntimeError(
            f"ffmpeg failed at ts={ts_sec}: {r.stderr[-400:]}"
        )


def _jpg_to_b64(jpg_path: Path) -> str:
    return base64.b64encode(jpg_path.read_bytes()).decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--features", type=Path, required=True,
                    help="features.h5 to derive timestamps + session_id from")
    ap.add_argument("--video", type=Path, required=True,
                    help="source MP4 (for frame decoding)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output questions.yaml path")
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--interval-half-window-sec", type=float, default=1.5,
                    help="±window around the sampled timestamp for the GT interval")
    ap.add_argument("--model", choices=["gemini", "claude"], default="gemini")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0,
                    help="numpy seed for evenly-spaced-but-jittered timestamp picking")
    args = ap.parse_args()

    # Load timestamps + session_id from the H5.
    with h5py.File(args.features, "r") as h:
        session_id = h.attrs.get("session_id", args.features.parent.name)
        if isinstance(session_id, bytes):
            session_id = session_id.decode()
        g = h["clip"]
        timestamps = g["timestamps"][:].astype(np.float64)

    n_total = len(timestamps)
    if n_total < args.n_questions:
        print(
            f"WARN: only {n_total} frames available, generating "
            f"{n_total} questions instead of {args.n_questions}",
            file=sys.stderr,
        )
        n_q = n_total
    else:
        n_q = args.n_questions

    # Evenly-spaced indices (no jitter — deterministic picks make
    # reruns reproducible).
    idx_picks = np.linspace(0, n_total - 1, n_q, dtype=np.int64)
    picked_ts = timestamps[idx_picks]

    model = Mllm.GEMINI if args.model == "gemini" else Mllm.CLAUDE
    print(f"[sloper4d-qg] {session_id}: picking {n_q} frames; using {model.name}", file=sys.stderr)

    questions: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, ts in enumerate(picked_ts):
            jpg = td_path / f"frame_{i:03d}.jpg"
            _decode_frame_at_timestamp(args.video, float(ts), jpg)
            b64 = _jpg_to_b64(jpg)
            try:
                caption = call_mllm(
                    model=model,
                    frames_b64=[b64],
                    prompt=_CAPTION_PROMPT,
                ).strip()
            except Exception as e:
                print(f"  WARN: caption failed at ts={ts}: {e}", file=sys.stderr)
                continue
            # Strip trailing punctuation duplicates and surrounding quotes.
            caption = caption.strip().strip('"').strip("'").strip()
            t_lo = max(0.0, float(ts) - args.interval_half_window_sec)
            t_hi = float(ts) + args.interval_half_window_sec
            questions.append({
                "id": f"q{i+1}",
                "query": caption,
                "intervals": [[round(t_lo, 3), round(t_hi, 3)]],
                "notes": f"auto-captioned via {model.name} at ts={ts:.3f}s (frame_idx={int(idx_picks[i])})",
            })
            print(f"  q{i+1} (ts={ts:.1f}s): {caption[:80]}", file=sys.stderr)

    out_doc = {
        "session_id": session_id,
        "session_start_unix": 0.0,
        "iou_threshold": float(args.iou_threshold),
        "questions": questions,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(out_doc, f, sort_keys=False, allow_unicode=True)

    print(f"[sloper4d-qg] wrote {len(questions)} questions to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
