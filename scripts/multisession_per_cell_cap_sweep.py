"""Cross-session per_cell_cap sweep on a Nymeria mobility-tiered subset.

Picks N sessions from each mobility tier (street/building/room/sub-room
per the displacement probe) and runs the per_cell_cap sweep on each.
Writes per-session JSONs under captures/multisession_pcc_sweep/<session>/.

Useful for the v1 generality claim: "the per_cell_cap effect holds
across N sessions spanning M meters of bbox extent."

Run from the repo root:
  python scripts/multisession_per_cell_cap_sweep.py [--mobility-json /checkpoint/.../nymeria_mobility.json]

Each session × 4 caps × CLIP-CPU = ~12 min/session. Default 4 sessions
= ~50 min wall. Use --sessions to subset / --caps to narrow.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


_DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic")
_DEFAULT_OUT = Path("captures/multisession_pcc_sweep")
_DEFAULT_MOBILITY = None  # set per-session-dir scan if not provided


def _pick_sessions_from_disk(root: Path, limit: int) -> list[str]:
    """Fallback when no mobility manifest: pick first `limit` sessions sorted by name."""
    sessions = sorted(d.name for d in root.iterdir() if d.is_dir())
    return sessions[:limit]


def _pick_sessions_by_mobility(
    mobility_json: Path, limit_per_tier: int = 1,
) -> list[str]:
    """One session per mobility tier from the displacement manifest.

    Tiers (from scripts/nymeria_slam_displacement.py interpretation):
      - street:    >= 50 m bbox extent
      - building:  >= 15 m
      - room:      >= 5 m
      - sub-room:  < 5 m  (typically excluded but include 1 for the limitation slot)
    """
    # The mobility JSON we built for Ego-Exo4D has a slightly different
    # schema; here we just want a list with traj_extent_m per session.
    # If the Nymeria-side JSON doesn't exist yet, fall back to disk order.
    d = json.loads(mobility_json.read_text())
    rows = d.get("takes") or d.get("sessions") or []
    if not rows:
        return []
    # Sort by traj_extent_m / traj_len_m / head_trajectory_m (whichever exists)
    def ext(r: dict) -> float:
        for k in ("traj_extent_m", "head_trajectory_m", "traj_len_m"):
            if k in r and r[k]:
                return float(r[k])
        return 0.0
    rows.sort(key=ext, reverse=True)
    picked: list[str] = []
    tiers = {"street": (50, float("inf")),
             "building": (15, 50),
             "room": (5, 15),
             "sub-room": (0, 5)}
    for tier_name, (lo, hi) in tiers.items():
        for r in rows:
            e = ext(r)
            name = r.get("take_name") or r.get("session_name") or r.get("name")
            if not name:
                continue
            if lo <= e < hi and name not in picked:
                picked.append(name)
                if sum(1 for p in picked if lo <= ext(next(x for x in rows if (x.get("take_name") or x.get("session_name") or x.get("name")) == p)) < hi) >= limit_per_tier:
                    break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--out-root", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--mobility-json", type=Path, default=_DEFAULT_MOBILITY,
        help="Nymeria displacement manifest (optional; falls back to disk-order pick).",
    )
    ap.add_argument(
        "--sessions", type=str, nargs="*", default=None,
        help="Explicit session names to sweep; overrides mobility-based picking.",
    )
    ap.add_argument(
        "--limit-sessions", type=int, default=4,
        help="When picking by disk order, take this many sessions.",
    )
    ap.add_argument(
        "--caps", type=int, nargs="+", default=[1, 2, 3, 5],
    )
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--h3-resolution", type=int, default=12)
    ap.add_argument("--time-window", type=float, default=30.0)
    ap.add_argument("--capacity", type=int, default=60)
    ap.add_argument("--exemplars", type=int, default=1024)
    ap.add_argument("--clip-device", default="cpu")
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    )
    args = ap.parse_args()

    if args.sessions:
        sessions = args.sessions
    elif args.mobility_json and args.mobility_json.exists():
        sessions = _pick_sessions_by_mobility(args.mobility_json)
        print(f"[multi] picked {len(sessions)} mobility-tiered sessions:"
              f" {sessions}", file=sys.stderr)
    else:
        sessions = _pick_sessions_from_disk(args.root, args.limit_sessions)
        print(f"[multi] picked {len(sessions)} sessions by disk order:"
              f" {sessions}", file=sys.stderr)

    args.out_root.mkdir(parents=True, exist_ok=True)

    for i, sid in enumerate(sessions, start=1):
        out_dir = args.out_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[multi] === ({i}/{len(sessions)}) {sid} ===", file=sys.stderr)
        cmd = [
            sys.executable, "scripts/sweep_per_cell_cap.py",
            "--take", sid,
            "--root", str(args.root),
            "--out-root", str(out_dir),
            "--caps", *[str(c) for c in args.caps],
            "--top", str(args.top),
            "--h3-resolution", str(args.h3_resolution),
            "--time-window", str(args.time_window),
            "--capacity", str(args.capacity),
            "--exemplars", str(args.exemplars),
            "--clip-device", args.clip_device,
            "--clip-checkpoint", args.clip_checkpoint,
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"[multi] FAIL on {sid}; continuing", file=sys.stderr)

    # Summary table at the end.
    print("\n[multi] === summary ===", file=sys.stderr)
    print(f"{'session':<55s} {'pcc':>4s} {'hit@5':>7s} {'bIoU':>7s} {'eIoU':>7s}")
    for sid in sessions:
        for cap in args.caps:
            f = args.out_root / sid / f"eval_{sid}_pcc{cap}.json"
            if not f.exists():
                print(f"{sid:<55s} {cap:>4d}  (no output)")
                continue
            try:
                s = json.loads(f.read_text())["summary"]
                print(f"{sid:<55s} {cap:>4d} "
                      f"{s['exemplar_hit_rate_at_5']*100:>6.1f}% "
                      f"{s['bucket_miou_at_5']:>7.3f} "
                      f"{s['exemplar_miou_at_5']:>7.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"{sid}: ERR {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
