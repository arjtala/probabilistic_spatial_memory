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


class FrameSource:
    """Resolve a target timestamp -> JPEG file the MLLM can ingest.

    Three concrete sources, picked by `make_frame_source` at startup:
      - CachedFramesSource: the extractor's frames/ dir already exists
        (--keep-frames was set). Cheapest path; lookup is an index
        into the sorted glob.
      - Mp4FrameSource: ffmpeg one-shot per timestamp into a temp dir.
        Slow per query (one ffmpeg invocation each), but no extractor
        re-run needed. Default for non-VRS sessions.
      - EgoExo4dFrameSource: Mp4FrameSource variant pointing at the
        take's `frame_aligned_videos/aria*_214-1.mp4`. Same code path
        underneath; broken out so detection stays explicit.
    """
    def fetch(self, t_s: float) -> Path:
        raise NotImplementedError

    def cleanup(self) -> None:
        """Release any temp resources; safe to call multiple times."""


class CachedFramesSource(FrameSource):
    def __init__(self, frames_dir: Path, sample_fps: float):
        self.frames_dir = frames_dir
        self.sample_fps = sample_fps
        self._all = sorted(frames_dir.glob("frame_*.jpg"))
        if not self._all:
            raise SystemExit(
                f"no frames under {frames_dir}; did the extractor run with --keep-frames?"
            )

    def fetch(self, t_s: float) -> Path:
        dt = 1.0 / self.sample_fps
        idx = int(round(max(0.0, t_s) / dt))
        return self._all[min(idx, len(self._all) - 1)]


class Mp4FrameSource(FrameSource):
    """Pull a single JPEG via ffmpeg seek+select. ~0.5-1s per fetch.

    ffmpeg's `-ss <t> -i <mp4> -frames:v 1` is fast because seek is
    keyframe-aligned (we don't need exact-frame accuracy — the MLLM
    only cares about the approximate moment, and PSM's exemplar_t is
    itself approximate). Caches by integer-second so two top-k
    candidates within the same second don't pay twice.
    """
    def __init__(self, mp4: Path, scratch_root: Path | None = None):
        self.mp4 = mp4
        self._scratch = Path(tempfile.mkdtemp(prefix="psm-mllm-frames-",
                                              dir=str(scratch_root) if scratch_root else None))
        self._cache: dict[int, Path] = {}

    def fetch(self, t_s: float) -> Path:
        # Quantize to whole seconds for caching — keyframe seek is coarse
        # anyway, so sub-second precision doesn't survive ffmpeg's -ss.
        bucket = int(round(max(0.0, t_s)))
        if bucket in self._cache:
            return self._cache[bucket]
        out = self._scratch / f"frame_t{bucket:06d}.jpg"
        import subprocess
        # -ss BEFORE -i = fast keyframe seek; accuracy is ~2s at worst,
        # which is fine since PSM exemplar_t is itself ~time_window
        # bucketed. Quality q=2 (high JPEG) so we don't double-degrade
        # before sending to the MLLM at max_size=768.
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{bucket}", "-i", str(self.mp4),
            "-frames:v", "1", "-q:v", "2",
            str(out),
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                f"ffmpeg failed extracting t={bucket}s from {self.mp4.name}: "
                f"{proc.stderr.strip()[:200]}"
            )
        self._cache[bucket] = out
        return out

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._scratch, ignore_errors=True)


def _read_source_video_attr(features_h5: Path) -> Path | None:
    """Return the `source_video` root attr from a features.h5, or None.

    Both extractors record where the source came from in this attr.
    Path may be a file (MP4) or a directory (Aria/Ego-Exo4D session).
    """
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(features_h5, "r") as f:
            v = f.attrs.get("source_video")
    except (OSError, KeyError):
        return None
    if v is None:
        return None
    s = v.decode() if isinstance(v, bytes) else str(v)
    return Path(s)


class VrsFrameSource(FrameSource):
    """Pull a single JPEG from an Aria VRS file on-the-fly.

    Reuses projectaria-tools' RGB stream decoder. The first fetch
    incurs the VRS provider open + RGB-stream enumeration (~1-2 s);
    subsequent fetches are sub-second since the provider is cached.

    Trade-off vs Mp4FrameSource: VRS decode is slower per-frame because
    we walk to the right device-clock timestamp instead of using
    ffmpeg's keyframe seek. For PSM->MLLM eval at ~5 frames/query that's
    fine (<5s/query overhead) and saves us re-extracting Nymeria's
    8.7 GB VRS files just to keep JPEGs around.
    """
    def __init__(self, vrs_path: Path, scratch_root: Path | None = None):
        self.vrs_path = vrs_path
        self._scratch = Path(tempfile.mkdtemp(
            prefix="psm-mllm-vrs-",
            dir=str(scratch_root) if scratch_root else None,
        ))
        self._provider = None
        self._rgb_stream = None
        self._first_ts_s: float | None = None
        # All RGB frame timestamps in device-clock seconds, indexed
        # by frame index. Cached on provider open so subsequent
        # fetches use np.searchsorted (O(log N)) instead of an O(N)
        # linear scan per fetch.
        self._frame_ts_s: np.ndarray | None = None
        self._cache: dict[int, Path] = {}

    def _ensure_provider(self):
        if self._provider is not None:
            return
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId
        self._provider = data_provider.create_vrs_data_provider(str(self.vrs_path))
        if self._provider is None:
            raise RuntimeError(f"projectaria-tools failed to open {self.vrs_path}")
        # 214-1 is the primary RGB camera on Aria Gen 1 and Gen 2.
        self._rgb_stream = StreamId("214-1")
        n = self._provider.get_num_data(self._rgb_stream)
        if n == 0:
            raise RuntimeError(f"{self.vrs_path}: no frames on stream 214-1")
        # Walk the stream once at open time: cache every frame's
        # device-clock timestamp into a contiguous numpy array. The
        # walk costs ~1-2s for a 1000-frame session and avoids paying
        # an O(N) get_image_data_by_index scan per fetch. (Linear-scan-
        # per-fetch made the multi-session sweep ~24h instead of ~2h
        # in the first attempt at this code path.)
        ts = np.empty(n, dtype=np.float64)
        for i in range(n):
            _, rec = self._provider.get_image_data_by_index(self._rgb_stream, i)
            ts[i] = rec.capture_timestamp_ns / 1e9
        self._first_ts_s = float(ts[0])
        # Store device-clock seconds *re-based to 0* so the lookup is
        # against the same timeline as the features.h5 timestamps the
        # caller passes in. Matches what aria_vrs.read_vrs_session
        # writes at extraction time.
        self._frame_ts_s = ts - ts[0]

    def fetch(self, t_s: float) -> Path:
        # Quantize to whole seconds for caching — VRS decode is heavy
        # enough that we don't want to decode twice for two top-k
        # candidates a few hundred ms apart.
        bucket = int(round(max(0.0, t_s)))
        if bucket in self._cache:
            return self._cache[bucket]
        self._ensure_provider()
        # Binary search for the closest frame to the target (in the
        # rebased 0-relative timeline). np.searchsorted gives the
        # insertion point; we pick whichever of the two neighbors is
        # closer in time.
        ts = self._frame_ts_s
        idx = int(np.searchsorted(ts, float(bucket)))
        if idx == 0:
            best_i = 0
        elif idx >= len(ts):
            best_i = len(ts) - 1
        else:
            best_i = idx if abs(ts[idx] - bucket) < abs(ts[idx - 1] - bucket) else idx - 1
        image_data, _ = self._provider.get_image_data_by_index(self._rgb_stream, best_i)
        arr = image_data.to_numpy_array()
        from PIL import Image
        out = self._scratch / f"frame_t{bucket:06d}.jpg"
        Image.fromarray(arr).save(out, format="JPEG", quality=90)
        self._cache[bucket] = out
        return out

    def cleanup(self) -> None:
        import shutil
        shutil.rmtree(self._scratch, ignore_errors=True)


def _resolve_video_from_source_attr(source: Path) -> Path | None:
    """Map `source_video` attr -> a usable video path, or None.

    The attr might be:
      - An MP4 file: return as-is (caller uses Mp4FrameSource).
      - A VRS file: return as-is (caller uses VrsFrameSource).
      - An Ego-Exo4D take directory: pick frame_aligned_videos/aria*_214-1.mp4.
      - A Nymeria/AEA session directory: pick recording_head/data/data.vrs.
      - An Aria Gen 2 take directory: pick video.vrs.
      - None of the above: None.
    """
    if source.is_file() and source.suffix.lower() == ".mp4":
        return source
    if source.is_file() and source.suffix.lower() == ".vrs":
        return source
    if source.is_dir():
        # Ego-Exo4D layout.
        ee_mp4 = next(source.glob("frame_aligned_videos/aria*_214-1.mp4"), None)
        if ee_mp4 is not None:
            return ee_mp4
        # Nymeria / AEA layout.
        nymeria_vrs = source / "recording_head" / "data" / "data.vrs"
        if nymeria_vrs.exists():
            return nymeria_vrs
        # Aria Gen 2 Pilot layout.
        gen2_vrs = source / "video.vrs"
        if gen2_vrs.exists():
            return gen2_vrs
        # Self-organized.
        for candidate in ("main.vrs", "aria01.vrs"):
            p = source / candidate
            if p.exists():
                return p
        return None
    return None


def _make_source_for(path: Path) -> FrameSource:
    """Pick Mp4FrameSource vs VrsFrameSource based on file suffix.

    Used by both the --video override and the source_video-attr
    resolution path so the two stay consistent. Suffix matched
    case-insensitively — SLOPER4D's DJI-Action2 source ships
    uppercase .MP4 extensions, others may vary.
    """
    if path.suffix.lower() == ".vrs":
        return VrsFrameSource(path)
    return Mp4FrameSource(path)


def make_frame_source(
    features_h5: Path,
    *,
    override_frames_dir: Path | None,
    override_video: Path | None,
    sample_fps: float,
) -> FrameSource:
    """Pick the frame-fetch strategy for this features.h5.

    Precedence:
      1. `--frames-dir` explicitly set + exists  -> CachedFramesSource.
      2. `--video` explicitly set                -> Mp4FrameSource on that file.
      3. Default `<features>/../frames/` exists  -> CachedFramesSource.
      4. h5 root attr `source_video` points at an existing directory
         (Ego-Exo4D take, Aria session) or MP4 file. The extractor
         always records this — handles the common case where
         features.h5 lives under /checkpoint/.../<name>/ but the
         source video lives under /datasets/.../takes/<name>/.
      5. Auto-detect from <features>/.. layout (legacy single-tree
         setups like Aria Gen 2 Pilot):
           - Ego-Exo4D take dir (frame_aligned_videos/aria*_214-1.mp4) -> Mp4FrameSource.
           - Aria session dir with video.vrs                            -> SystemExit
             (VRS-on-the-fly is too slow; tell user to re-run with --keep-frames).
           - sibling *.mp4                                              -> Mp4FrameSource.
      6. Otherwise: SystemExit telling the user how to fix it.

    The auto-detect deliberately doesn't fall through to a global glob
    -- silent picks of the wrong video file are exactly the kind of
    bug that wastes a paper-deadline afternoon.
    """
    if override_frames_dir is not None:
        if not override_frames_dir.exists():
            raise SystemExit(f"--frames-dir {override_frames_dir} does not exist")
        return CachedFramesSource(override_frames_dir, sample_fps)

    if override_video is not None:
        if not override_video.exists():
            raise SystemExit(f"--video {override_video} does not exist")
        return _make_source_for(override_video)

    default_frames = features_h5.parent / "frames"
    if default_frames.exists() and any(default_frames.glob("frame_*.jpg")):
        return CachedFramesSource(default_frames, sample_fps)

    # Cross-tree resolution via the h5's source_video attr. Covers:
    #   - Ego-Exo4D: features.h5 under /checkpoint/..., MP4 under /datasets/...
    #   - Nymeria:   features.h5 under /checkpoint/.../nymeria_atomic/,
    #                VRS under /checkpoint/.../nymeria_partial/
    #   - Aria Gen 2 Pilot: features.h5 + VRS share a dir; either layout works
    source_attr = _read_source_video_attr(features_h5)
    if source_attr is not None:
        recovered = _resolve_video_from_source_attr(source_attr)
        if recovered is not None:
            return _make_source_for(recovered)

    # Auto-detect from session-dir layout (legacy single-tree setups).
    session_dir = features_h5.parent
    egoexo_mp4 = next(session_dir.glob("frame_aligned_videos/aria*_214-1.mp4"), None)
    if egoexo_mp4 is not None:
        return Mp4FrameSource(egoexo_mp4)

    for vrs_candidate in (
        session_dir / "video.vrs",
        session_dir / "recording_head" / "data" / "data.vrs",
    ):
        if vrs_candidate.exists():
            return VrsFrameSource(vrs_candidate)

    sibling_mp4s = sorted(session_dir.glob("*.mp4")) + sorted(session_dir.glob("*.MP4"))
    if len(sibling_mp4s) == 1:
        return Mp4FrameSource(sibling_mp4s[0])
    if len(sibling_mp4s) > 1:
        raise SystemExit(
            f"multiple *.mp4 next to {features_h5}; specify one with --video: "
            f"{[p.name for p in sibling_mp4s]}"
        )

    raise SystemExit(
        f"no frame source found for {features_h5}. Pass --frames-dir or --video, "
        "or re-run extraction with --keep-frames."
    )


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
                    help="JPEG cache from extractor (default: auto-detect; "
                         "see make_frame_source docstring for precedence)")
    ap.add_argument("--video", type=Path, default=None,
                    help="Source MP4 for on-the-fly frame extraction via "
                         "ffmpeg (alternative to --frames-dir).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N questions (smoke-test mode).")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--time-window", type=float, default=75.0)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--h3-resolution", type=int, default=10)
    ap.add_argument("--precision", type=int, default=14)
    ap.add_argument("--exemplars", type=int, default=128)
    ap.add_argument("--exemplar-codec", default="raw")
    ap.add_argument("--per-cell-cap", type=int, default=1,
                    help="Max top-k slots a single H3 cell may fill (see "
                         "eval_lookback.py for ablation details).")
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

    frame_source = make_frame_source(
        args.features,
        override_frames_dir=args.frames_dir,
        override_video=args.video,
        sample_fps=args.sample_fps,
    )
    print(f"[eval] frame source: {type(frame_source).__name__}", file=sys.stderr)

    with args.questions.open() as f:
        spec = yaml.safe_load(f)
    questions = spec.get("questions") or []
    if args.limit is not None:
        questions = questions[: args.limit]
        print(f"[eval] --limit applied: processing {len(questions)} questions",
              file=sys.stderr)
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
                per_cell_cap=args.per_cell_cap,
            )
            results = payload.get("results", [])

            # 2. Resolve each result to its nearest cached frame.
            candidates: list[_Candidate] = []
            for r in results:
                t_min = float(r["t_min"]) - session_start
                t_max = float(r["t_max"]) - session_start
                exemplar_t = float(r.get("exemplar_t", (t_min + t_max) / 2.0)) - session_start
                frame_path = frame_source.fetch(exemplar_t)
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
        frame_source.cleanup()

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
