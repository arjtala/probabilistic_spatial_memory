"""Unit tests for the Ego4D NLQ reader.

The fixture mirrors the shape of nlq_val.json — top-level `videos`,
each with `clips`, each with `annotations`, each with
`language_queries`. The cases below are the realistic edge cases I've
seen in NLQ release files, not synthetic ones:

  - Multiple annotators tag the same clip with the same (text,
    interval) — must dedupe per video.
  - Some language_queries entries omit the `query` field (in-flight
    annotations) — skip.
  - Zero/negative-width intervals exist in the raw data — drop.
  - Templates can be missing — treat as "".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from psm_extraction.io.ego4d_nlq import (
    NlqQuery,
    read_nlq_annotations,
    summarize_nlq_split,
)


def _write_nlq(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "nlq.json"
    p.write_text(json.dumps(payload))
    return p


def test_basic_extraction(tmp_path: Path) -> None:
    p = _write_nlq(tmp_path, {
        "videos": [
            {
                "video_uid": "vA",
                "clips": [
                    {
                        "clip_uid": "cA1",
                        "video_start_sec": 0.0,
                        "video_end_sec": 300.0,
                        "annotations": [
                            {
                                "annotation_uid": "ann1",
                                "language_queries": [
                                    {
                                        "query": "where did I leave my keys?",
                                        "template": "Where is X?",
                                        "video_start_sec": 10.0,
                                        "video_end_sec": 14.0,
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    })
    videos = read_nlq_annotations(p)
    assert len(videos) == 1
    assert videos[0].video_uid == "vA"
    assert len(videos[0].queries) == 1
    q = videos[0].queries[0]
    assert q.query == "where did I leave my keys?"
    assert q.template == "Where is X?"
    assert q.t_start_sec == 10.0
    assert q.t_end_sec == 14.0


def test_dedupe_per_video(tmp_path: Path) -> None:
    # Two annotators tagged the same clip + same query + same interval.
    p = _write_nlq(tmp_path, {
        "videos": [{
            "video_uid": "v1",
            "clips": [{
                "clip_uid": "c1",
                "video_start_sec": 0.0,
                "video_end_sec": 300.0,
                "annotations": [
                    {
                        "annotation_uid": "annA",
                        "language_queries": [
                            {"query": "the red mug", "template": "Where?",
                             "video_start_sec": 5.0, "video_end_sec": 8.0},
                        ],
                    },
                    {
                        "annotation_uid": "annB",
                        "language_queries": [
                            {"query": "the red mug", "template": "Where?",
                             "video_start_sec": 5.0, "video_end_sec": 8.0},
                        ],
                    },
                ],
            }],
        }],
    })
    videos = read_nlq_annotations(p)
    assert len(videos[0].queries) == 1


def test_skip_missing_query(tmp_path: Path) -> None:
    p = _write_nlq(tmp_path, {
        "videos": [{
            "video_uid": "v1",
            "clips": [{"clip_uid": "c1", "video_start_sec": 0.0,
                       "video_end_sec": 300.0, "annotations": [{
                "annotation_uid": "a1",
                "language_queries": [
                    {"video_start_sec": 1.0, "video_end_sec": 2.0},  # no query
                    {"query": "valid", "video_start_sec": 3.0, "video_end_sec": 4.0},
                ],
            }]}],
        }],
    })
    videos = read_nlq_annotations(p)
    assert len(videos[0].queries) == 1
    assert videos[0].queries[0].query == "valid"


def test_drop_degenerate_intervals(tmp_path: Path) -> None:
    p = _write_nlq(tmp_path, {
        "videos": [{
            "video_uid": "v1",
            "clips": [{"clip_uid": "c1", "video_start_sec": 0.0,
                       "video_end_sec": 300.0, "annotations": [{
                "annotation_uid": "a1",
                "language_queries": [
                    # zero width
                    {"query": "q1", "video_start_sec": 10.0, "video_end_sec": 10.0},
                    # negative width
                    {"query": "q2", "video_start_sec": 20.0, "video_end_sec": 15.0},
                    {"query": "q3", "video_start_sec": 30.0, "video_end_sec": 31.0},
                ],
            }]}],
        }],
    })
    videos = read_nlq_annotations(p)
    assert len(videos[0].queries) == 1
    assert videos[0].queries[0].query == "q3"


def test_multiple_videos_sorted(tmp_path: Path) -> None:
    p = _write_nlq(tmp_path, {
        "videos": [
            {"video_uid": "vZ", "clips": [{"clip_uid": "c", "video_start_sec": 0.0,
                "video_end_sec": 100.0, "annotations": [{"annotation_uid": "a",
                "language_queries": [{"query": "z", "video_start_sec": 1.0,
                                      "video_end_sec": 2.0}]}]}]},
            {"video_uid": "vA", "clips": [{"clip_uid": "c", "video_start_sec": 0.0,
                "video_end_sec": 100.0, "annotations": [{"annotation_uid": "a",
                "language_queries": [{"query": "a", "video_start_sec": 1.0,
                                      "video_end_sec": 2.0}]}]}]},
        ],
    })
    videos = read_nlq_annotations(p)
    assert [v.video_uid for v in videos] == ["vA", "vZ"]


def test_summarize(tmp_path: Path) -> None:
    p = _write_nlq(tmp_path, {
        "videos": [{
            "video_uid": "v1",
            "clips": [{"clip_uid": "c1", "video_start_sec": 0.0,
                       "video_end_sec": 300.0, "annotations": [{
                "annotation_uid": "a1",
                "language_queries": [
                    {"query": "q1", "template": "T1",
                     "video_start_sec": 0.0, "video_end_sec": 10.0},
                    {"query": "q2", "template": "T2",
                     "video_start_sec": 5.0, "video_end_sec": 15.0},
                ],
            }]}],
        }],
    })
    videos = read_nlq_annotations(p)
    stats = summarize_nlq_split(videos)
    assert stats["n_videos"] == 1
    assert stats["n_questions"] == 2
    assert stats["n_unique_templates"] == 2
    assert stats["duration_sec_median"] == 10.0


def test_missing_videos_key_raises(tmp_path: Path) -> None:
    p = _write_nlq(tmp_path, {"clips": []})
    with pytest.raises(ValueError, match="missing top-level 'videos' key"):
        read_nlq_annotations(p)
