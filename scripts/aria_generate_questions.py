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
    "walking outdoors with head-mounted Project Aria glasses. The Aria "
    "RGB sensor is mounted rotated 90 degrees by design, so EVERY frame "
    "in this dataset appears sideways. This is a hardware feature, not "
    "a property of any specific frame. DO NOT mention rotation, tilt, "
    "sideways orientation, vertical/horizontal framing, or the camera "
    "viewpoint itself. Mentally re-orient the frame upright and describe "
    "the WORLD content as if you were standing there looking at the "
    "scene. "
    "Describe the SINGLE most visually distinctive feature in the scene "
    "— a specific object (sign, building, landmark, sculpture, vehicle, "
    "person doing something specific), visible text or numbers, or a "
    "distinctive arrangement of objects. Avoid generic scene words like "
    "'path', 'trees', 'sky', 'grass', 'person walking', 'cloudy'. Also "
    "avoid meta-commentary like 'the image shows' or 'the camera "
    "captures' — just describe the scene content directly. "
    "One concise sentence, ≤20 words. Start directly with the description, "
    "no preamble, no chain-of-thought."
)

_ANTI_EXAMPLE_SUFFIX = (
    "\n\nIMPORTANT: avoid descriptions that would also fit any of these "
    "earlier frames from the same recording. Pick a DIFFERENT "
    "distinguishing detail.\n"
)


def _jpg_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def _kmeans_pick(emb: np.ndarray, k: int, *, rng_seed: int = 0,
                 max_iter: int = 50, n_init: int = 5) -> np.ndarray:
    """k-means on `emb` (N, D), then pick the medoid of each cluster.

    Returns int64 indices into `emb` of length `k`. Pure NumPy — avoids
    a sklearn dependency for what is a few-hundred-row, few-iteration
    job. CLIP embeddings are L2-normalised so Euclidean distance is
    monotone with cosine similarity; using Euclidean keeps the math
    simple and the picks ranking-equivalent.

    Multi-init: run `n_init` random starts and return the picks of the
    run with the lowest total within-cluster sum of squares.
    """
    rng = np.random.default_rng(rng_seed)
    n, d = emb.shape
    if k >= n:
        return np.arange(n, dtype=np.int64)

    best_score = np.inf
    best_picks = None
    for init_i in range(n_init):
        # k-means++ style init: pick first center at random, then each
        # subsequent center proportional to its squared distance to the
        # nearest already-chosen center.
        centers_idx = [int(rng.integers(n))]
        for _ in range(1, k):
            d2 = np.min(
                np.linalg.norm(emb[:, None, :] - emb[centers_idx][None, :, :], axis=2) ** 2,
                axis=1,
            )
            d2[centers_idx] = 0.0
            if d2.sum() <= 0:
                break
            probs = d2 / d2.sum()
            centers_idx.append(int(rng.choice(n, p=probs)))
        centers = emb[centers_idx].copy()

        # Lloyd iterations.
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(max_iter):
            # Distances: (n, k).
            dists = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centers[ci] = emb[mask].mean(axis=0)

        # Score = sum of within-cluster squared distances.
        score = float(np.sum(np.min(
            np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2) ** 2,
            axis=1,
        )))

        # Medoid of each cluster.
        picks = np.empty(k, dtype=np.int64)
        for ci in range(k):
            members = np.where(labels == ci)[0]
            if len(members) == 0:
                picks[ci] = int(rng.integers(n))  # degenerate; fallback
                continue
            d_to_center = np.linalg.norm(emb[members] - centers[ci], axis=1)
            picks[ci] = int(members[np.argmin(d_to_center)])
        # Dedup (rare degenerate case).
        picks = np.unique(picks)
        # If dedup shrunk us, fill from random un-picked indices.
        if len(picks) < k:
            remaining = np.setdiff1d(np.arange(n), picks)
            rng.shuffle(remaining)
            picks = np.concatenate([picks, remaining[: k - len(picks)]])

        if score < best_score:
            best_score = score
            best_picks = picks[:k]

    assert best_picks is not None
    return best_picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--session-dir", type=Path, default=None,
                    help="Aria session dir containing video.vrs (or similar). "
                         "Mutually exclusive with --frames-dir.")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="Pre-extracted JPEG/PNG dir (e.g. LookOut's "
                         "rgb_data/undistorted_aa/). Skips VRS read; "
                         "frames are aligned 1:1 with H5 timestamps by "
                         "list position. Mutually exclusive with "
                         "--session-dir.")
    ap.add_argument("--features", type=Path, required=True,
                    help="features.h5 to derive H5 timestamps + session_id from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame-cache", type=Path, default=None,
                    help="dir for the 1 fps frame cache when reading VRS "
                         "(default: <session>/frames_qg). Ignored when "
                         "--frames-dir is set (frames are already on disk).")
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--interval-half-window-sec", type=float, default=1.5)
    ap.add_argument("--model", choices=["gemini", "claude"], default="gemini")
    ap.add_argument("--iou-threshold", type=float, default=0.3)
    ap.add_argument("--anti-example-k", type=int, default=5)
    ap.add_argument("--diverse-sample", action="store_true",
                    help="pick frames via k-means on CLIP embeddings instead of "
                         "evenly-spaced timestamps. Forces queries to span visually "
                         "distinct stretches of the trajectory; useful when "
                         "evenly-spaced picks land in a stationary phase (e.g. "
                         "walk_0 evenly-spaced ⇒ all parking-lot captions).")
    args = ap.parse_args()

    if (args.session_dir is None) == (args.frames_dir is None):
        raise SystemExit(
            "exactly one of --session-dir or --frames-dir is required"
        )

    if args.frames_dir is not None:
        # Pre-extracted-frames path. The LookOut layout is the prototype:
        # rgb_data/undistorted_aa/ holds the full-rate (20 fps) PNGs and
        # rgb_data/rgb_info.pkl holds (frame_idx, capture_ts_ns) tuples
        # for each PNG. The features.h5 was written from a 1-fps subsample
        # of those frames, so we need to redo the same subsample logic
        # here to align H5 indices with PNG file paths.
        rgb_info_path = args.frames_dir.parent / "rgb_info.pkl"
        if rgb_info_path.exists():
            import pickle
            with rgb_info_path.open("rb") as f:
                rgb_info = pickle.load(f)
            full_ts_ns = np.array([int(r[1]) for r in rgb_info], dtype=np.int64)
            full_ts_s = (full_ts_ns - full_ts_ns[0]) / 1e9
            # Same greedy 1-fps subsample the extractor used.
            target_fps = 1.0
            period = 1.0 / target_fps
            kept_idx = [0]
            next_t = float(full_ts_s[0]) + period
            for i in range(1, len(full_ts_s)):
                if float(full_ts_s[i]) >= next_t:
                    kept_idx.append(i)
                    next_t = float(full_ts_s[i]) + period
            # Frame_i ↔ kept_idx[i]'th PNG, named "<kept_idx[i]>_*.png"
            # in the LookOut layout (see scripts/extract_lookout_sessions.py).
            frame_paths = [
                args.frames_dir / f"{idx}_undistorted_512_243.png"
                for idx in kept_idx
            ]
            missing = [p for p in frame_paths if not p.exists()]
            if missing:
                raise SystemExit(
                    f"{len(missing)}/{len(frame_paths)} subsampled PNGs missing under "
                    f"{args.frames_dir} (first: {missing[0].name})"
                )
            frame_ts = np.array([full_ts_s[i] for i in kept_idx], dtype=np.float64)
            print(f"[aria-qg] using {len(frame_paths)} LookOut-subsampled frames "
                  f"from {args.frames_dir}", file=sys.stderr)
        else:
            # Generic glob path (no rgb_info.pkl) — assume frames are
            # already in 1:1 correspondence with H5 timestamps (e.g. an
            # extractor variant that kept them at 1 fps directly).
            frame_paths = sorted(
                list(args.frames_dir.glob("*.jpg"))
                + list(args.frames_dir.glob("*.JPG"))
                + list(args.frames_dir.glob("*.png"))
                + list(args.frames_dir.glob("*.PNG"))
            )
            if not frame_paths:
                raise SystemExit(f"no .jpg/.png frames under {args.frames_dir}")
            frame_ts = np.arange(len(frame_paths), dtype=np.float64)
            print(f"[aria-qg] using {len(frame_paths)} pre-extracted frames "
                  f"(no rgb_info.pkl; assuming 1:1 with H5)", file=sys.stderr)
    else:
        # VRS path. Populate / reuse the 1 fps JPEG cache via the
        # project's VRS reader. Same code path the orchestrator uses
        # during extraction with --keep-frames; the manifest check
        # makes re-runs free.
        frame_cache = args.frame_cache or (args.session_dir / "frames_qg")
        print(f"[aria-qg] reading VRS frames from {args.session_dir}", file=sys.stderr)
        vrs_out = read_vrs_session(args.session_dir, sample_fps=1.0,
                                   output_dir=frame_cache, verbose=True)
        frame_paths = vrs_out.frame_paths
        frame_ts = vrs_out.timestamps_s
        print(f"[aria-qg] {len(frame_paths)} cached frames span "
              f"{frame_ts[0]:.1f}..{frame_ts[-1]:.1f}s", file=sys.stderr)

    # Load H5 timestamps to keep the questions.yaml intervals
    # consistent with eval_lookback's matching against clip/timestamps.
    with h5py.File(args.features, "r") as h:
        session_id = h.attrs.get(
            "session_id",
            (args.session_dir.name if args.session_dir else args.frames_dir.parent.name),
        )
        if isinstance(session_id, bytes):
            session_id = session_id.decode()
        g = next((h[k] for k in ("clip", "dino", "jepa") if k in h), None)
        if g is None or "timestamps" not in g:
            raise RuntimeError(f"{args.features} has no embedding group with timestamps")
        h5_ts = g["timestamps"][:].astype(np.float64)
        # Load embeddings only if we'll need them for diverse sampling.
        h5_emb = g["embeddings"][:].astype(np.float32) if args.diverse_sample else None
    print(f"[aria-qg] H5 timestamps: n={len(h5_ts)} "
          f"span {h5_ts[0]:.1f}..{h5_ts[-1]:.1f}s", file=sys.stderr)

    # Pick N indices into the H5 timestamp array, then map each one to
    # the nearest cached frame.
    n_q = min(args.n_questions, len(h5_ts))
    if args.diverse_sample:
        assert h5_emb is not None
        # k-means on the L2-normalized CLIP embeddings, pick the
        # closest-to-centroid frame from each cluster. Forces queries
        # to span the visually-distinct regions of the trajectory
        # instead of clustering on a stationary phase (e.g. walk_0
        # spent its evenly-spaced picks all in a parking lot).
        h5_idx = _kmeans_pick(h5_emb, n_q, rng_seed=0)
        # Sort picks by timestamp so the anti-example prompt stays
        # temporally coherent (recent captions = recent stretches).
        h5_idx = h5_idx[np.argsort(h5_ts[h5_idx])]
        print(f"[aria-qg] k-means diverse sample: {n_q} frames across {len(h5_emb)} embeddings",
              file=sys.stderr)
    else:
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
