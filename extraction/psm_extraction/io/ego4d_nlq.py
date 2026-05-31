"""Ego4D NLQ (Natural Language Queries) reader.

Parses the official `nlq_val.json` / `nlq_train.json` annotation file
into a per-source-video structure that PSM's eval_lookback.py can
consume directly. NLQ queries come with `(video_start_sec,
video_end_sec)` intervals already expressed in source-video time, so
no clip-to-video remapping is needed.

The schema (NLQ v2, ego4d-data.org docs):
  {
    "videos": [
      {
        "video_uid": "...",
        "clips": [
          {
            "clip_uid": "...",
            "video_start_sec": float,
            "video_end_sec": float,
            "annotations": [
              {
                "annotation_uid": "...",
                "language_queries": [
                  {
                    "query": "...",           # may be missing -> skip
                    "template": "...",        # category-ish; passed through
                    "video_start_sec": float,
                    "video_end_sec": float
                  },
                  ...
                ]
              },
              ...
            ]
          },
          ...
        ]
      },
      ...
    ]
  }

Per-video grouping is intentional: NLQ-relevant Ego4D videos are
~30 min each, the embedding extraction is dominated by the full video
decode, and clips that share a video would otherwise force redundant
work. One features.h5 per video_uid; per-video questions.yaml.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NlqQuery:
    """One natural-language query with its ground-truth time interval.

    `template` is Ego4D's category-ish field (e.g. "Where was object X
    before/after event Y?"); we pass it through as the `category` of
    the produced questions.yaml entry so the eval aggregator can
    bucket performance by template if we want.
    """
    query: str
    template: str
    t_start_sec: float          # video-frame seconds (not clip-relative)
    t_end_sec: float
    annotation_uid: str          # for traceability back to the source JSON
    clip_uid: str


@dataclass
class NlqVideo:
    """All NLQ queries that ground to one source video, plus its UID.

    `queries` are deduplicated by (query, t_start, t_end) — the same
    natural-language question can appear under multiple annotation_uids
    when multiple annotators tagged the same clip. We keep the first
    occurrence; the rest would inflate the question count without
    adding evaluation signal.
    """
    video_uid: str
    queries: list[NlqQuery] = field(default_factory=list)


def read_nlq_annotations(
    nlq_json_path: Path,
    *,
    skip_missing_query: bool = True,
) -> list[NlqVideo]:
    """Load nlq_val.json (or nlq_train.json) -> per-video question lists.

    `skip_missing_query` drops language_queries entries with no
    `query` field. The val split has a handful of these (annotation
    in-flight at release time); silently skipping matches what the
    official NLQ baselines do.

    Returns the videos sorted by video_uid for deterministic output.
    """
    raw = json.loads(nlq_json_path.read_text())
    if "videos" not in raw:
        raise ValueError(
            f"{nlq_json_path}: missing top-level 'videos' key — "
            "is this an NLQ annotation file?"
        )

    by_video: dict[str, list[NlqQuery]] = defaultdict(list)
    seen_per_video: dict[str, set[tuple[str, float, float]]] = defaultdict(set)
    for v in raw["videos"]:
        vuid = v.get("video_uid")
        if not vuid:
            continue
        for clip in v.get("clips") or []:
            clip_uid = clip.get("clip_uid", "")
            for ann in clip.get("annotations") or []:
                ann_uid = ann.get("annotation_uid", "")
                for lq in ann.get("language_queries") or []:
                    text = lq.get("query")
                    if skip_missing_query and not text:
                        continue
                    try:
                        t_start = float(lq["video_start_sec"])
                        t_end = float(lq["video_end_sec"])
                    except (KeyError, TypeError, ValueError):
                        # Malformed interval — skip rather than poison the bank.
                        continue
                    if t_end <= t_start:
                        # Zero/negative-width intervals exist in the raw data; drop.
                        continue
                    key = (text or "", t_start, t_end)
                    if key in seen_per_video[vuid]:
                        continue
                    seen_per_video[vuid].add(key)
                    by_video[vuid].append(
                        NlqQuery(
                            query=text or "",
                            template=lq.get("template") or "",
                            t_start_sec=t_start,
                            t_end_sec=t_end,
                            annotation_uid=ann_uid,
                            clip_uid=clip_uid,
                        )
                    )

    return [
        NlqVideo(video_uid=vuid, queries=by_video[vuid])
        for vuid in sorted(by_video)
    ]


def summarize_nlq_split(videos: list[NlqVideo]) -> dict:
    """Compact stats for an NLQ split — handy for the converter CLI banner.

    Returns a dict the caller can json.dumps or pretty-print:
      - n_videos
      - n_questions
      - n_unique_templates
      - mean / median / max questions per video
      - median interval duration
    """
    import statistics

    n_q_per_video = [len(v.queries) for v in videos]
    durations = [
        q.t_end_sec - q.t_start_sec
        for v in videos for q in v.queries
    ]
    templates = {q.template for v in videos for q in v.queries}
    return {
        "n_videos": len(videos),
        "n_questions": sum(n_q_per_video),
        "n_unique_templates": len(templates),
        "q_per_video_mean": (statistics.mean(n_q_per_video) if n_q_per_video else 0.0),
        "q_per_video_median": (statistics.median(n_q_per_video) if n_q_per_video else 0.0),
        "q_per_video_max": (max(n_q_per_video) if n_q_per_video else 0),
        "duration_sec_median": (statistics.median(durations) if durations else 0.0),
        "duration_sec_mean": (statistics.mean(durations) if durations else 0.0),
    }
