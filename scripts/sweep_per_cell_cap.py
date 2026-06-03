"""Per-cell-cap operating point sweep on one Nymeria session.

Runs eval_lookback at a fixed (h3, time-window, capacity, exemplars)
operating point while varying `--per-cell-cap` ∈ {1, 2, 3, 5}. Default
psm_binary, default seed; writes one JSON per (take, cap) under
`captures/` (or `--out-root`).

Usage:
  python scripts/sweep_per_cell_cap.py
      [--features /checkpoint/.../<take>/clip_l_features.h5]
      [--questions /checkpoint/.../<take>/questions.yaml]
      [--out-root captures]
      [--caps 1 2 3 5]

Output JSON names: eval_<session_id>_pcc<cap>.json — drop-in for
eval_aggregate.py's existing glob pattern.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


_DEFAULT_TAKE = "20230608_s0_shelby_arroyo_act0_3ciwl8"
_DEFAULT_ROOT = Path("/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic")
_DEFAULT_OUT = Path("captures")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--take", type=str, default=_DEFAULT_TAKE,
                    help="Used to derive features/questions paths when those flags omitted.")
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--out-root", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3, 5])
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

    features = args.features or (args.root / args.take / "clip_l_features.h5")
    questions = args.questions or (args.root / args.take / "questions.yaml")
    if not features.exists():
        print(f"ERR: {features} not found", file=sys.stderr)
        return 1
    if not questions.exists():
        print(f"ERR: {questions} not found", file=sys.stderr)
        return 1

    args.out_root.mkdir(parents=True, exist_ok=True)
    sid = args.take

    print(f"[sweep] take={sid} caps={args.caps}", file=sys.stderr)
    print(f"[sweep] features={features}", file=sys.stderr)
    print(f"[sweep] params: r{args.h3_resolution} "
          f"tw={args.time_window} cap={args.capacity} ex={args.exemplars}",
          file=sys.stderr)

    for cap in args.caps:
        out_path = args.out_root / f"eval_{sid}_pcc{cap}.json"
        cmd = [
            sys.executable, "scripts/eval_lookback.py",
            str(features), str(questions),
            "--top", str(args.top),
            "--per-cell-cap", str(cap),
            "--h3-resolution", str(args.h3_resolution),
            "--time-window", str(args.time_window),
            "--capacity", str(args.capacity),
            "--exemplars", str(args.exemplars),
            "--clip-device", args.clip_device,
            "--clip-checkpoint", args.clip_checkpoint,
            "--out", str(out_path),
        ]
        print(f"\n[sweep] === pcc={cap} -> {out_path.name} ===", file=sys.stderr)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"[sweep] FAIL at pcc={cap}; stopping", file=sys.stderr)
            return proc.returncode

    print("\n[sweep] done. Compare with:", file=sys.stderr)
    print(f"  ls -la {args.out_root}/eval_{sid}_pcc*.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
