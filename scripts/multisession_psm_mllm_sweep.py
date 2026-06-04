"""Multi-session PSM->Gemini sweep across 4 sessions × {cap=1, cap=5}.

The PSM-only sweep already showed the cap effect generalizes across
mobility tiers. This sweep adds the MLLM rerank to that table to
report whether Gemini's localization-quality contribution also
generalizes.

Only runs cap=1 and cap=5 (the two endpoints) per session — the
PSM-only sweep showed the middle caps (2, 3) interpolate linearly
between them, so they add little to the rerank story but cost 30 min
extra per session each in Gemini calls.

Each session × cap takes ~30 min (187-220 questions × ~10s/call).
4 sessions × 2 caps = ~4 hours wall. Run in tmux/screen on the
login node.

Run:
  python scripts/multisession_psm_mllm_sweep.py

Output: captures/multisession_psm_mllm/<sid>/eval_<sid>_mllm_pcc<cap>.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic")
_DEFAULT_OUT = Path("captures/multisession_psm_mllm")
_DEFAULT_SESSIONS = [
    "20230608_s0_shelby_arroyo_act0_3ciwl8",
    "20230607_s0_james_johnson_act0_e72nhq",
    "20230609_s0_angela_harrell_act4_egucf6",
    "20230612_s0_jason_smith_act3_c6na21",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--out-root", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--sessions", nargs="+", default=_DEFAULT_SESSIONS)
    ap.add_argument("--caps", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--mllm", default="gemini", choices=("gemini", "claude"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--h3-resolution", type=int, default=12)
    ap.add_argument("--time-window", type=float, default=30.0)
    ap.add_argument("--capacity", type=int, default=60)
    ap.add_argument("--exemplars", type=int, default=1024)
    ap.add_argument("--exemplar-tolerance", type=float, default=5.0)
    ap.add_argument("--clip-device", default="cpu")
    ap.add_argument(
        "--clip-checkpoint",
        default="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    )
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    total = len(args.sessions) * len(args.caps)
    n_done = 0
    for sid in args.sessions:
        feat = args.root / sid / "clip_l_features.h5"
        q = args.root / sid / "questions.yaml"
        if not feat.exists() or not q.exists():
            print(f"[psm-mllm-sweep] skip {sid}: missing features/questions", file=sys.stderr)
            continue
        sess_dir = args.out_root / sid
        sess_dir.mkdir(parents=True, exist_ok=True)
        for cap in args.caps:
            n_done += 1
            out_path = sess_dir / f"eval_{sid}_mllm_pcc{cap}.json"
            print(f"\n[psm-mllm-sweep] === ({n_done}/{total}) "
                  f"{sid} pcc={cap} -> {out_path.name} ===", file=sys.stderr)
            if out_path.exists():
                print(f"[psm-mllm-sweep] skip; output already exists", file=sys.stderr)
                continue
            cmd = [
                sys.executable, "scripts/eval_psm_mllm.py",
                str(feat), str(q),
                "--mllm", args.mllm,
                "--top", str(args.top),
                "--per-cell-cap", str(cap),
                "--h3-resolution", str(args.h3_resolution),
                "--time-window", str(args.time_window),
                "--capacity", str(args.capacity),
                "--exemplars", str(args.exemplars),
                "--exemplar-tolerance", str(args.exemplar_tolerance),
                "--clip-device", args.clip_device,
                "--clip-checkpoint", args.clip_checkpoint,
                "--out", str(out_path),
            ]
            # PYTHONUNBUFFERED forces the child's stdout to flush per
            # line so the parent terminal shows per-query progress as
            # it happens. Also passes --verbose so the child emits the
            # [eval] qN: pick=X lines.
            import os
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd.append("--verbose")
            proc = subprocess.run(cmd, check=False, env=env)
            if proc.returncode != 0:
                print(f"[psm-mllm-sweep] FAIL on {sid} pcc={cap}; continuing",
                      file=sys.stderr)

    # Summary table
    print("\n[psm-mllm-sweep] === summary ===", file=sys.stderr)
    print(f"{'session':<55s} {'pcc':>4s} {'hit@5':>7s} {'eIoU':>7s}")
    for sid in args.sessions:
        for cap in args.caps:
            f = args.out_root / sid / f"eval_{sid}_mllm_pcc{cap}.json"
            if not f.exists():
                print(f"{sid:<55s} {cap:>4d}  (no output)")
                continue
            try:
                s = json.loads(f.read_text())["summary"]
                print(f"{sid:<55s} {cap:>4d} "
                      f"{s['exemplar_hit_at_k']*100:>6.1f}% "
                      f"{s['exemplar_mIoU_at_k']:>7.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"{sid}: ERR {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
