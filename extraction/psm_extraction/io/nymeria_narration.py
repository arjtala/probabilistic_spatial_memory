"""Nymeria narration reader.

Parses each session's `narration/atomic_action.csv` into per-session
QA records. Same shape as `egoexo4d_atomic.read_atomic_descriptions`
but adapted to Nymeria's CSV schema:

  request_id, gaia_id, start_time, end_time, annotator,
  creation_time, "Describe my atomic actions"  <-- last column = text

Time units are seconds since Aria capture start (head VRS device clock).
Narration timestamps here are rebased by subtracting the SLAM trajectory
CSV's first `tracking_timestamp_us` (`trajectory_t0`). The VRS extractor,
however, rebases features.h5 frame timestamps against the first *sampled
RGB frame* (`rgb_t0`), not the trajectory's first row. Those two origins
are not guaranteed to coincide, so narration intervals produced here can
be offset from the features.h5 timeline by `(rgb_t0 - trajectory_t0)`.

CAVEAT: this module cannot see `rgb_t0` (it lives in the VRS extractor),
so it rebases against `trajectory_t0` and exposes that value on
`NymeriaSession.trajectory_t0_sec`. A consumer that also knows `rgb_t0`
should re-align by subtracting `(rgb_t0 - trajectory_t0)` from the
`t_start_sec` / `t_end_sec` fields. When `rgb_t0 == trajectory_t0` (both
clocks start at the head VRS device-clock origin) no further adjustment
is needed, but that equality is unverified for the general case.

Unlike Ego-Exo4D atomic_descriptions, Nymeria narrations come with
real `[start_time, end_time]` intervals (not single timestamps), so
no half-window-expansion trick is needed. The intervals are typically
4-5 seconds, matching the underlying 5s sliding-window annotation
protocol.

The default filter is permissive: keep every annotation with a
non-empty text + valid interval. Nymeria doesn't have Ego-Exo4D's
`ego_visible` flag — by construction the narrations describe what
the wearer is doing, which is always ego-visible.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# The CSV's last column holds the free-text narration; the column name
# is the questionnaire prompt itself ("Describe my atomic actions"),
# which is awkward to hardcode but stable across the v0.0 release.
_NARRATION_TEXT_COL = "Describe my atomic actions"

# Standard Nymeria session-dir layout (matches what aria_vrs._locate_vrs_file
# already handles via the Nymeria/AEA candidate path).
_NARRATION_CSV = "narration/atomic_action.csv"


@dataclass
class NymeriaNarration:
    """One atomic-action narration from a Nymeria session.

    Time fields are seconds relative to the SLAM trajectory's first row
    (`trajectory_t0`; see module docstring). The VRS extractor rebases
    features.h5 frame timestamps against the first sampled RGB frame
    (`rgb_t0`) instead, so `(t_start_sec, t_end_sec)` may be offset from
    the features.h5 timeline by `(rgb_t0 - trajectory_t0)`. A consumer
    that knows `rgb_t0` should subtract that offset before matching; the
    basis used here is exposed via `NymeriaSession.trajectory_t0_sec`.
    """
    text: str
    t_start_sec: float
    t_end_sec: float
    annotator_id: str
    request_uid: str
    gaia_id: str


@dataclass
class NymeriaSession:
    """All atomic-action narrations for one Nymeria session.

    `narrations` are deduplicated by (text, t_start_sec) within a
    session — different `request_id`s sometimes re-annotate the same
    window with paraphrases; keep the first.

    `trajectory_t0_sec` is the SLAM-trajectory device-clock origin (seconds)
    the narration timestamps were rebased against. Exposed so a consumer
    that also knows the extractor's RGB-frame origin (`rgb_t0`) can correct
    the `(rgb_t0 - trajectory_t0)` offset described in the module docstring.
    """
    session_id: str          # the directory name, e.g. 20230607_s0_james_johnson_act0_e72nhq
    narrations: list[NymeriaNarration] = field(default_factory=list)
    trajectory_t0_sec: float | None = None


# Trajectory CSV's first row gives us the Aria device-clock time the
# recording started. The narration's start_time / end_time are on the
# same device clock, so we subtract trajectory_t0 to put them on the
# extractor's 0-relative timeline. Path mirrors aria_vrs._locate_slam_trajectory's
# Nymeria/AEA candidate.
_TRAJECTORY_CSV = "recording_head/mps/slam/closed_loop_trajectory.csv"


def _trajectory_t0_seconds(session_dir: Path) -> float | None:
    """Return the first tracking_timestamp_us / 1e6 from the SLAM CSV.

    Used by `read_session_narrations` to rebase narration timestamps
    onto the same 0-origin timeline the VRS extractor writes for frame
    timestamps. Both narration `start_time`/`end_time` and trajectory
    `tracking_timestamp_us` are on the head VRS device clock; subtracting
    the trajectory's first row aligns them with the rebased features.h5
    timeline.

    Returns None if the CSV is missing or unreadable; caller will skip
    the rebase (matching the legacy behavior that produced t=0-mismatched
    questions.yaml — better to error visibly than rebase against a
    guessed zero).
    """
    p = session_dir / _TRAJECTORY_CSV
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            reader = csv.DictReader(f)
            if "tracking_timestamp_us" not in (reader.fieldnames or []):
                return None
            first = next(reader, None)
            if first is None:
                return None
            return float(first["tracking_timestamp_us"]) / 1e6
    except (OSError, ValueError, KeyError):
        return None


def read_session_narrations(
    session_dir: Path,
) -> NymeriaSession | None:
    """Load `narration/atomic_action.csv` for one Nymeria session.

    Rebases narration timestamps by subtracting the trajectory CSV's
    first `tracking_timestamp_us` (`trajectory_t0`), the closest in-module
    proxy for the extractor's frame-timeline origin. NOTE: the extractor
    actually rebases features.h5 against the first sampled RGB frame
    (`rgb_t0`), which this module cannot see; when the two origins differ
    the intervals are offset by `(rgb_t0 - trajectory_t0)`. The basis used
    is returned on `NymeriaSession.trajectory_t0_sec` so a consumer that
    knows `rgb_t0` can correct it (see module docstring). When the
    trajectory CSV is missing (no SLAM), returns None — without any rebase
    the narrations would land at unphysical times like `33446s` for a
    1207s-long recording.

    Returns None if the CSV is missing — caller should skip the
    session rather than emit an empty record (otherwise downstream
    eval sweeps would log "0 questions" for sessions that are
    genuinely unannotatable).
    """
    csv_path = session_dir / _NARRATION_CSV
    if not csv_path.is_file():
        return None
    t0 = _trajectory_t0_seconds(session_dir)
    if t0 is None:
        # Nymeria sessions without a SLAM trajectory CSV can't be
        # rebased; without rebase the narration intervals are unusable
        # (off by ~33,000s). Better to drop the session than to write
        # questions that always miss the ground truth.
        return None

    out: list[NymeriaNarration] = []
    seen: set[tuple[str, float]] = set()
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if _NARRATION_TEXT_COL not in (reader.fieldnames or []):
            return None  # schema drift; surface as no-data.
        for row in reader:
            text = (row.get(_NARRATION_TEXT_COL) or "").strip()
            if not text:
                continue
            try:
                t_start = float(row["start_time"]) - t0
                t_end = float(row["end_time"]) - t0
            except (KeyError, TypeError, ValueError):
                continue
            if t_end <= t_start:
                continue  # degenerate interval; skip.
            if t_start < 0:
                # Narration recorded before the trajectory started —
                # would land at a negative timestamp in features.h5,
                # which the eval IoU math can't match. Drop.
                continue
            key = (text, t_start)
            if key in seen:
                continue
            seen.add(key)
            out.append(NymeriaNarration(
                text=text,
                t_start_sec=t_start,
                t_end_sec=t_end,
                annotator_id=row.get("annotator", "") or "",
                request_uid=row.get("request_id", "") or "",
                gaia_id=row.get("gaia_id", "") or "",
            ))
    if not out:
        return None
    return NymeriaSession(
        session_id=session_dir.name,
        narrations=out,
        trajectory_t0_sec=t0,
    )


def read_nymeria_root(
    root: Path,
) -> list[NymeriaSession]:
    """Walk a Nymeria root (`nymeria_partial` / `nymeria_dataset` layout)
    and return one NymeriaSession per session dir with usable narrations.

    Sessions are returned sorted by session_id for deterministic output.
    Sessions without narrations are silently dropped — the caller
    typically wants only the sessions it can actually evaluate.
    """
    sessions: list[NymeriaSession] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        s = read_session_narrations(d)
        if s is not None:
            sessions.append(s)
    return sessions


def summarize_nymeria_split(sessions: list[NymeriaSession]) -> dict:
    """Compact stats for a Nymeria split — for CLI banner.

    Mirrors `summarize_atomic_split` / `summarize_nlq_split` so the
    converter scripts can swap one for the other.
    """
    import statistics

    n_per_session = [len(s.narrations) for s in sessions]
    durations = [
        n.t_end_sec - n.t_start_sec
        for s in sessions for n in s.narrations
    ]
    n_unique_gaia = len({n.gaia_id for s in sessions for n in s.narrations if n.gaia_id})
    return {
        "n_sessions": len(sessions),
        "n_narrations": sum(n_per_session),
        "n_unique_gaia_ids": n_unique_gaia,
        "n_per_session_mean": (statistics.mean(n_per_session) if n_per_session else 0.0),
        "n_per_session_median": (statistics.median(n_per_session) if n_per_session else 0.0),
        "n_per_session_max": (max(n_per_session) if n_per_session else 0),
        "duration_sec_median": (statistics.median(durations) if durations else 0.0),
        "duration_sec_mean": (statistics.mean(durations) if durations else 0.0),
    }
