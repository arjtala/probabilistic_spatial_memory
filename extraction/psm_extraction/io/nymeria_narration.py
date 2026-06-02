"""Nymeria narration reader.

Parses each session's `narration/atomic_action.csv` into per-session
QA records. Same shape as `egoexo4d_atomic.read_atomic_descriptions`
but adapted to Nymeria's CSV schema:

  request_id, gaia_id, start_time, end_time, annotator,
  creation_time, "Describe my atomic actions"  <-- last column = text

Time units are seconds since Aria capture start (head VRS device clock).
The VRS reader rebases device-clock to t=0 when extracting frames, so
narration timestamps end up in the same frame timeline as the model
group's `timestamps` dataset — no additional rebase needed at eval
time (verified against atomic_action's column-7 floats vs the head
VRS's `_read_vrs_timestamps` start offset).

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

    Time fields are seconds in the head VRS device-clock frame; the
    same offset the extractor rebases to t=0 when writing the model
    group. So a (t_start, t_end) here lines up with the features.h5
    timestamps without further adjustment.
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
    """
    session_id: str          # the directory name, e.g. 20230607_s0_james_johnson_act0_e72nhq
    narrations: list[NymeriaNarration] = field(default_factory=list)


def read_session_narrations(
    session_dir: Path,
) -> NymeriaSession | None:
    """Load `narration/atomic_action.csv` for one Nymeria session.

    Returns None if the CSV is missing — caller should skip the
    session rather than emit an empty record (otherwise downstream
    eval sweeps would log "0 questions" for sessions that are
    genuinely unannotatable).
    """
    csv_path = session_dir / _NARRATION_CSV
    if not csv_path.is_file():
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
                t0 = float(row["start_time"])
                t1 = float(row["end_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if t1 <= t0:
                continue  # degenerate interval; skip.
            key = (text, t0)
            if key in seen:
                continue
            seen.add(key)
            out.append(NymeriaNarration(
                text=text,
                t_start_sec=t0,
                t_end_sec=t1,
                annotator_id=row.get("annotator", "") or "",
                request_uid=row.get("request_id", "") or "",
                gaia_id=row.get("gaia_id", "") or "",
            ))
    if not out:
        return None
    return NymeriaSession(session_id=session_dir.name, narrations=out)


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
