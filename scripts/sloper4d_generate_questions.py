#!/usr/bin/env python3
"""Generate questions.yaml for a tracked session via MLLM frame captions.

Originally SLOPER4D-only; now also the captioned-`similarity_search` path for
the HDD long-revisit cohort (PREREGISTRATION.md Amendment B). Picks N frames,
captions each with Gemini 3.1 Pro (or Claude 4.6 Opus), and writes them as
``query_mode: similarity_search`` questions with a small interval
window. The resulting questions.yaml matches the schema the existing
eval pipeline already consumes — drop-in compatible with
eval_lookback.py, eval_fixed_budget.py, the H3-resolution sweep, baselines.

Frame selection is either ``--place-selection even`` (legacy: evenly spaced
along the trajectory) or ``--place-selection metric`` (samples inside revisited
places, clustered by metric distance on the raw track). Metric mode exists so a
bank can exercise revisit structure without becoming a function of the H3
partition that also defines the evaluator's strata — see §3b. It requires an
explicit ``--gt-visits`` because "which visits count as ground truth" changes
what the bank measures and must be preregistered rather than defaulted.

For a preregistered bank, pass ``--prompt-file``: the prompt is then a frozen
artifact whose sha256 is recorded in the output alongside the model id.

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
import hashlib
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
from generate_revisit_questions import (  # noqa: E402
    metric_places,
    revisited_places,
)


_CAPTION_PROMPT = (
    "You are looking at one frame of an egocentric video from a runner "
    "on a university campus. Describe the SINGLE most visually "
    "distinctive feature you can identify in this frame — a specific "
    "object (sign, building, landmark, sculpture, vehicle, person doing "
    "something specific), visible text or numbers, or an unusual "
    "viewpoint. Avoid generic scene words like 'path', 'trees', 'sky', "
    "'grass', 'person walking', 'cloudy'. One concise sentence, ≤20 "
    "words. Start directly with the description, no preamble."
)

_ANTI_EXAMPLE_SUFFIX = (
    "\n\nIMPORTANT: avoid descriptions that would also fit any of these "
    "earlier frames from the same recording. Pick a DIFFERENT "
    "distinguishing detail.\n"
)


#: Seconds of accurate (post-input) seek to keep in front of the target frame.
#: Long enough to cross any sane GOP so the decode is frame-accurate, short
#: enough that we never decode more than this much video per frame.
_SEEK_REFINE_SEC = 5.0


def _decode_frame_at_timestamp(
    mp4_path: Path, ts_sec: float, out_jpg: Path
) -> None:
    """One-shot ffmpeg frame extract, frame-accurate but not O(video length).

    Hybrid seek: a fast pre-input ``-ss`` to ``ts - _SEEK_REFINE_SEC`` (index
    seek, keyframe-accurate, constant time) followed by an accurate post-input
    ``-ss`` over the remaining few seconds. Pure post-input seek is exact but
    decodes the whole file up to the target — fine for SLOPER4D's few-minute
    sequences, untenable on HDD's 1-3 h drives (minutes per frame, and 20
    frames x 60 drives of it). Pure pre-input seek is fast but lands on a
    keyframe, which can fall outside a +/-1.5 s ground-truth interval.
    """
    coarse = max(0.0, ts_sec - _SEEK_REFINE_SEC)
    fine = ts_sec - coarse
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{coarse:.3f}",
        "-i", str(mp4_path),
        "-ss", f"{fine:.3f}",
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
    ap.add_argument("--anti-example-k", type=int, default=5,
                    help="inject the last K captions as anti-examples to force "
                         "the MLLM toward distinguishing details (default 5; "
                         "set to 0 to disable)")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="caption prompt to use instead of the built-in "
                         "SLOPER4D one. Externalising the prompt makes it a "
                         "frozen, sha256-able artifact; required for any "
                         "preregistered bank.")
    ap.add_argument("--place-selection", choices=["even", "metric"], default="even",
                    help="'even' (default, legacy) spaces caption frames evenly "
                         "along the session. 'metric' samples inside revisited "
                         "places clustered by metric distance on the raw track "
                         "— H3-independent, per PREREGISTRATION.md §3b.")
    ap.add_argument("--gt-visits", choices=["all", "captioned"], default=None,
                    help="REQUIRED with --place-selection metric. 'captioned': "
                         "the GT interval is the captioned visit only (measures "
                         "episode discrimination; two passes of one corner look "
                         "alike, so a correct place retrieval can score as a "
                         "miss). 'all': every visit of that place is a GT "
                         "interval (measures place recall). No default — this "
                         "choice changes what is being measured and must be "
                         "preregistered, not inherited.")
    ap.add_argument("--place-radius-m", type=float, default=15.0)
    ap.add_argument("--visit-gap-sec", type=float, default=30.0)
    ap.add_argument("--min-visits", type=int, default=2)
    ap.add_argument("--min-separation-sec", type=float, default=120.0)
    args = ap.parse_args()

    if args.place_selection == "metric" and args.gt_visits is None:
        ap.error("--place-selection metric requires an explicit --gt-visits")

    caption_prompt = (
        args.prompt_file.read_text().strip() if args.prompt_file else _CAPTION_PROMPT
    )
    prompt_sha = hashlib.sha256(caption_prompt.encode()).hexdigest()

    # Load timestamps + session_id from the H5.
    with h5py.File(args.features, "r") as h:
        session_id = h.attrs.get("session_id", args.features.parent.name)
        if isinstance(session_id, bytes):
            session_id = session_id.decode()
        g = h["clip"]
        timestamps = g["timestamps"][:].astype(np.float64)
        lat = g["lat"][:].astype(np.float64) if "lat" in g else None
        lng = g["lng"][:].astype(np.float64) if "lng" in g else None

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

    # Frame selection. 'even' is the legacy path. 'metric' targets revisited
    # places so the bank actually exercises revisit structure, selecting them
    # by metric distance on the raw track rather than by H3 cell — question
    # construction must stay independent of the partition used for scoring and
    # strata (PREREGISTRATION.md §3b).
    gt_intervals_by_pick: list[list[tuple[float, float]]] | None = None
    place_of_pick: list[int] = []
    if args.place_selection == "metric":
        if lat is None or lng is None:
            print(f"[caption-qg] {args.features} has no clip/lat,lng — metric "
                  "place selection needs a position track", file=sys.stderr)
            return 2
        groups, _ = metric_places(lat, lng, radius_m=args.place_radius_m)
        revisits = revisited_places(
            groups, timestamps,
            visit_gap_sec=args.visit_gap_sec,
            min_visits=args.min_visits,
            min_separation_sec=args.min_separation_sec,
        )
        if not revisits:
            print(f"[caption-qg] {session_id}: no genuinely-separated revisits "
                  f"at radius {args.place_radius_m} m — not a revisit session",
                  file=sys.stderr)
            return 3
        # Deterministic: richest places first (most visits, then most frames),
        # tie-broken by place id. No RNG.
        ranked = sorted(
            revisits.items(),
            key=lambda kv: (-len(kv[1]), -sum(ep.size for ep in kv[1]), kv[0]),
        )
        picks: list[int] = []
        gt_intervals_by_pick = []
        # Richest-first alone concentrates the bank: the most-revisited places
        # cluster in one congested stretch (a lot, or stop-and-go traffic), so
        # the first N picks can all land within a couple of minutes of each
        # other and the bank then samples one location, not the session. Require
        # each captioned frame to be --min-separation-sec from every earlier
        # pick; on a driving route that spreads geography too. Reuses the
        # existing separation parameter rather than adding a knob.
        for place_id, episodes in ranked:
            if len(picks) >= n_q:
                break
            cand = episodes[-1]
            cand_t = float(timestamps[int(cand[cand.size // 2])])
            if any(abs(cand_t - float(timestamps[p])) < args.min_separation_sec
                   for p in picks):
                continue
            # Caption the LAST visit: "when was I last here before now?" reads
            # forward from a return, so the captioned frame should be the
            # return, not the first arrival.
            frame_idx = int(cand[cand.size // 2])
            picks.append(frame_idx)
            place_of_pick.append(int(place_id))
            if args.gt_visits == "all":
                gt_intervals_by_pick.append([
                    (float(timestamps[ep[0]]), float(timestamps[ep[-1]]))
                    for ep in episodes
                ])
            else:
                gt_intervals_by_pick.append([])  # filled with ±window below
        idx_picks = np.asarray(picks, dtype=np.int64)
        n_q = idx_picks.size
        print(f"[caption-qg] {session_id}: {len(revisits)} revisited places, "
              f"captioning {n_q} (gt-visits={args.gt_visits})", file=sys.stderr)
    else:
        # Evenly-spaced indices (no jitter — deterministic picks make
        # reruns reproducible).
        idx_picks = np.linspace(0, n_total - 1, n_q, dtype=np.int64)
    # h5_ts: keep H5's original (possibly session-relative) timestamps
    # for the questions.yaml intervals — eval_lookback.py matches them
    # against the same H5 field.
    # video_ts: shifted to start at 0 so ffmpeg `-ss <video_ts>` lands
    # on a frame the MP4 actually contains. Identical to h5_ts for
    # sequences whose H5 already starts at 0 (the common case).
    h5_ts = timestamps[idx_picks]
    if len(timestamps) > 0 and timestamps[0] > 5.0:
        offset = float(timestamps[0])
        print(
            f"[caption-qg] H5 timestamps start at {offset:.1f}s (session-relative); "
            f"shifting to video-clock for ffmpeg seek only",
            file=sys.stderr,
        )
        video_ts = h5_ts - offset
    else:
        video_ts = h5_ts

    model = Mllm.GEMINI if args.model == "gemini" else Mllm.CLAUDE
    print(f"[caption-qg] {session_id}: picking {n_q} frames; using {model.name}", file=sys.stderr)

    questions: list[dict] = []
    # Stream questions to disk after every successful caption so a
    # mid-run cancellation (or an MLLM transient) doesn't lose all
    # the prior work. On rerun, we re-load and resume from the next
    # un-captioned frame instead of paying for already-done frames.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        try:
            existing = yaml.safe_load(args.out.read_text()) or {}
            existing_q = existing.get("questions") or []
            done_ids = {q.get("id") for q in existing_q if isinstance(q, dict)}
            questions = list(existing_q)
            if done_ids:
                print(
                    f"[caption-qg] resuming: {len(done_ids)} questions already on disk",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"[caption-qg] WARN: could not parse existing {args.out}: {e}", file=sys.stderr)
            questions = []

    done_ids = {q.get("id") for q in questions if isinstance(q, dict)}

    def _flush() -> None:
        out_doc = {
            "session_id": session_id,
            "session_start_unix": 0.0,
            "iou_threshold": float(args.iou_threshold),
            "generator": {
                "script": "sloper4d_generate_questions.py",
                "model": model.model_id,
                "prompt_file": str(args.prompt_file) if args.prompt_file else None,
                "prompt_sha256": prompt_sha,
                "place_selection": args.place_selection,
                "place_radius_m": (args.place_radius_m
                                   if args.place_selection == "metric" else None),
                "visit_gap_sec": (args.visit_gap_sec
                                  if args.place_selection == "metric" else None),
                "min_separation_sec": (args.min_separation_sec
                                       if args.place_selection == "metric" else None),
                "gt_visits": args.gt_visits,
                "anti_example_k": args.anti_example_k,
                "note": ("MLLM-captioned similarity_search proxy; NOT "
                         "human-authored. Frame selection is H3-independent."),
            },
            "questions": questions,
        }
        # Atomic write: write to .tmp then rename, so a kill mid-write
        # never leaves a corrupted yaml on disk.
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with open(tmp, "w") as f:
            yaml.safe_dump(out_doc, f, sort_keys=False, allow_unicode=True)
        tmp.replace(args.out)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for i, (h5_t, video_t) in enumerate(zip(h5_ts, video_ts)):
            qid = f"q{i+1}"
            if qid in done_ids:
                continue
            jpg = td_path / f"frame_{i:03d}.jpg"
            _decode_frame_at_timestamp(args.video, float(video_t), jpg)
            b64 = _jpg_to_b64(jpg)
            # Build the prompt with anti-example captions from the
            # last K already-captioned frames. Forces Gemini to pick
            # a distinguishing detail rather than re-describing the
            # same scenery primitives ("paved path / sky / trees").
            recent_captions = [
                q["query"] for q in questions[-args.anti_example_k:]
                if isinstance(q, dict) and q.get("query")
            ]
            if recent_captions:
                anti_block = "\n".join(f"- {c}" for c in recent_captions)
                prompt = caption_prompt + _ANTI_EXAMPLE_SUFFIX + anti_block
            else:
                prompt = caption_prompt
            try:
                caption = call_mllm(
                    model=model,
                    frames_b64=[b64],
                    prompt=prompt,
                ).strip()
            except Exception as e:
                print(f"  WARN: caption failed at h5_ts={h5_t}: {e}", file=sys.stderr)
                continue
            # Strip trailing punctuation duplicates and surrounding quotes.
            caption = caption.strip().strip('"').strip("'").strip()
            # Intervals are written in the H5 clock so eval_lookback can
            # match them against `clip/timestamps` directly (which uses
            # the H5 clock — same field).
            # Declines: the prompt may offer the model an explicit opt-out for
            # frames with nothing distinctive in them. Also drop captions that
            # echo prompt text back (observed: a model returning the sentence
            # describing the opt-out instead of using it) — an echoed
            # instruction would otherwise be embedded and scored as a query.
            decline = caption.upper().startswith(("NOTHING DISTINCTIVE", "SKIP FRAME"))
            echoed = any(
                fragment in caption.lower()
                for fragment in ("reply exactly", "reply with the two words",
                                 "nothing distinctive", "skip frame",
                                 "one concise sentence")
            )
            if decline or echoed:
                why = "declined — nothing distinctive" if decline else "dropped — echoed prompt"
                print(f"  q{i+1} (h5_ts={h5_t:.1f}s): {why}", file=sys.stderr)
                continue
            if gt_intervals_by_pick is not None and gt_intervals_by_pick[i]:
                # Pad every visit span by the half-window. A visit episode can
                # be a single frame, whose raw span is 0 s — and a zero-length
                # interval scores IoU 0 against anything, so it would be an
                # automatic miss no matter what the policy retained.
                half = args.interval_half_window_sec
                intervals = [
                    [round(max(0.0, lo - half), 3), round(hi + half, 3)]
                    for lo, hi in gt_intervals_by_pick[i]
                ]
            else:
                t_lo = max(0.0, float(h5_t) - args.interval_half_window_sec)
                t_hi = float(h5_t) + args.interval_half_window_sec
                intervals = [[round(t_lo, 3), round(t_hi, 3)]]
            note = (f"auto-captioned via {model.name} at h5_ts={h5_t:.3f}s "
                    f"(video_ts={video_t:.3f}s, frame_idx={int(idx_picks[i])})")
            if place_of_pick:
                note += (f"; metric place {place_of_pick[i]} "
                         f"(r={args.place_radius_m} m), gt_visits={args.gt_visits}")
            questions.append({
                "id": qid,
                "query": caption,
                "intervals": intervals,
                "notes": note,
            })
            _flush()
            print(f"  q{i+1} (h5_ts={h5_t:.1f}s): {caption[:80]}", file=sys.stderr)

    print(f"[caption-qg] wrote {len(questions)} questions to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
