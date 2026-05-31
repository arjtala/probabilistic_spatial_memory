"""Ego-Exo4D atomic-descriptions reader.

Ego-Exo4D ships `atomic_descriptions_{train,val}.json` with timestamped
one-line narrations per take ("C picks up the knife at 41.6s"). We
treat each description as a look-back retrieval target: the text is
the query, and the ground-truth time interval is a small window
around the description's `timestamp` field.

Schema (Ego-Exo4D v2):
  {
    "ds": "<release-version-string>",
    "take_cam_id_map": {...},
    "annotations": {
      "<take_uid>": [
        {
          "annotation_uid": "...",
          "annotator_id": "...",
          "descriptions": [
            {
              "text": "...",
              "timestamp": float,           # seconds from take start
              "subject": "C" | "O" | ...,   # C = camera-wearer, O = other
              "ego_visible": bool,
              "best_exo": {...} | null,
              "unsure": bool | null
            },
            ...
          ]
        },
        ...
      ]
    }
  }

For look-back retrieval against ego-only CLIP features, we keep
`ego_visible == True and not unsure` descriptions. Filter rationale:

  - `ego_visible: false` means the described content is only in the
    exo cameras; the ego camera literally cannot see it, so a CLIP
    retrieval over ego frames *cannot* succeed. Including them would
    measure annotation noise rather than retrieval quality.
  - `unsure: true` means the annotator wasn't confident; ~0.3% of the
    val set, easy drop, removes false-negative interval bounds.

The `subject` field is intentionally NOT filtered to `C` — many
"O"-subject descriptions are still ego-visible (the wearer sees the
other person doing X) and form a useful retrieval target.

Each description has a point `timestamp`, not an interval. We expand
to `[t - half_window, t + half_window]` (default 1.5s) so the same
bucket-IoU / exemplar-IoU metrics used for NLQ work unchanged.
1.5s matches the typical CLIP-at-1Hz frame cadence + ~1 frame slop.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# How wide an interval to put around each point-in-time description.
# 1.5s on each side = 3s total — wide enough that frame quantization
# (typical extraction at 1 Hz) doesn't drop the matching frame, narrow
# enough that bucket-IoU stays meaningful.
_DEFAULT_HALF_WINDOW_SEC = 1.5


@dataclass
class AtomicDescription:
    """One atomic description from the Ego-Exo4D val/train split.

    Fields preserved for traceability back to the source JSON
    (annotation_uid / annotator_id) and for downstream filtering
    (subject = "C" / "O" / etc).
    """
    text: str
    t_start_sec: float       # = timestamp - half_window
    t_end_sec: float         # = timestamp + half_window
    timestamp_sec: float     # the original point timestamp, preserved
    subject: str
    annotation_uid: str
    annotator_id: str


@dataclass
class AtomicTake:
    """All ego-visible atomic descriptions for one Ego-Exo4D take.

    `descriptions` are deduplicated by (text, timestamp_sec) within
    a take — different annotators sometimes write the same text at
    the same moment; keeping one is enough.
    """
    take_uid: str
    descriptions: list[AtomicDescription] = field(default_factory=list)


def read_atomic_descriptions(
    json_path: Path,
    *,
    half_window_sec: float = _DEFAULT_HALF_WINDOW_SEC,
    require_ego_visible: bool = True,
    drop_unsure: bool = True,
) -> list[AtomicTake]:
    """Load atomic_descriptions_val.json -> per-take description lists.

    Returns takes sorted by uid for deterministic output. Takes with
    zero usable descriptions (after filtering) are dropped.

    The default filter (`require_ego_visible=True`, `drop_unsure=True`)
    is the right one for PSM look-back retrieval against ego CLIP
    features. Relax it only when you know what you're doing —
    `ego_visible=False` content is impossible to retrieve from ego
    frames, so including it will tank apparent accuracy without any
    signal about the retrieval method.
    """
    raw = json.loads(json_path.read_text())
    if "annotations" not in raw:
        raise ValueError(
            f"{json_path}: missing top-level 'annotations' key — "
            "is this an Ego-Exo4D atomic_descriptions file?"
        )

    by_take: dict[str, list[AtomicDescription]] = defaultdict(list)
    seen: dict[str, set[tuple[str, float]]] = defaultdict(set)

    for take_uid, ann_list in raw["annotations"].items():
        if not isinstance(ann_list, list):
            continue
        for ann in ann_list:
            ann_uid = ann.get("annotation_uid", "")
            annotator = ann.get("annotator_id", "")
            for desc in ann.get("descriptions") or []:
                text = (desc.get("text") or "").strip()
                if not text:
                    continue
                if require_ego_visible and not desc.get("ego_visible"):
                    continue
                if drop_unsure and desc.get("unsure"):
                    continue
                try:
                    ts = float(desc["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (text, ts)
                if key in seen[take_uid]:
                    continue
                seen[take_uid].add(key)
                by_take[take_uid].append(
                    AtomicDescription(
                        text=text,
                        t_start_sec=max(0.0, ts - half_window_sec),
                        t_end_sec=ts + half_window_sec,
                        timestamp_sec=ts,
                        subject=desc.get("subject") or "",
                        annotation_uid=ann_uid,
                        annotator_id=annotator,
                    )
                )

    takes = [
        AtomicTake(take_uid=uid, descriptions=by_take[uid])
        for uid in sorted(by_take)
        if by_take[uid]  # drop takes with zero usable descriptions
    ]
    return takes


def summarize_atomic_split(takes: list[AtomicTake]) -> dict:
    """Compact stats for an atomic-descriptions split — for CLI banner.

    Mirrors `summarize_nlq_split` so the converter scripts can swap
    one for the other.
    """
    import statistics

    n_d_per_take = [len(t.descriptions) for t in takes]
    n_unique_subjects = len({
        d.subject for t in takes for d in t.descriptions if d.subject
    })
    return {
        "n_takes": len(takes),
        "n_descriptions": sum(n_d_per_take),
        "n_unique_subjects": n_unique_subjects,
        "d_per_take_mean": (statistics.mean(n_d_per_take) if n_d_per_take else 0.0),
        "d_per_take_median": (statistics.median(n_d_per_take) if n_d_per_take else 0.0),
        "d_per_take_max": (max(n_d_per_take) if n_d_per_take else 0),
    }
