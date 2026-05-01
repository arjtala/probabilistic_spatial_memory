#!/usr/bin/env python3
"""Self-evaluation harness for PSM look-back queries.

For each (question, ground-truth interval[s]) pair in a YAML/JSON file,
embed the question via CLIP, call `targets/psm --search`, and report
IoU vs. ground truth. Aggregates mIoU and Hit@k for the run.

This is the analogue of E5/E7 in EXPERIMENTS.md, scoped to a single
session annotated by hand. It exists so that "PSM hits non-zero mIoU
on look-back queries while a vanilla MLLM hits zero" stops being a
rhetorical claim and becomes a measurement we can put alongside the
embedding atlas figure in the paper.

Usage:
    python scripts/eval_lookback.py \\
        datasets/1501677363692556/clip_features.h5 \\
        datasets/1501677363692556/questions.yaml \\
        --top 5 --time-window 75 --capacity 12

Output:
    - stdout: a markdown table summary + aggregate metrics
    - --out PATH: detailed JSON record (one entry per question)

Question file (YAML or JSON):

    session_id: 1501677363692556
    session_start_unix: 1678365188.0   # optional; auto-detected if absent
    iou_threshold: 0.3                 # for Hit@k metric
    questions:
      - id: q1
        query: "a red bus"
        intervals:                     # seconds from session start
          - [253.0, 268.0]
        notes: "stop on Wandsworth Bridge Rd"
      - id: q2
        query: "a zebra crossing"
        intervals:
          - [161.0, 250.0]
          - [382.0, 467.0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "extraction"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def load_questions(path: Path) -> dict:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "yaml not installed. Either pip install pyyaml or use a .json file."
            ) from exc
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise SystemExit(f"Unsupported question file format: {path}")


def auto_session_start(features_h5: Path, group: str) -> float:
    with h5py.File(features_h5, "r") as f:
        if group not in f:
            raise SystemExit(f"group {group!r} not in {features_h5}")
        ts = f[f"{group}/timestamps"][:]
    return float(ts[0])


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred: tuple[float, float], gts: list[tuple[float, float]]) -> tuple[float, int]:
    """Return (best_iou, idx_of_matching_gt). idx = -1 if no gt or no overlap."""
    if not gts:
        return 0.0, -1
    best = 0.0
    best_idx = -1
    for i, gt in enumerate(gts):
        v = interval_iou(pred, gt)
        if v > best:
            best = v
            best_idx = i
    return best, best_idx


def run_psm_search(
    psm_binary: Path,
    features_h5: Path,
    group: str,
    query_path: Path,
    *,
    top: int,
    time_window: float,
    capacity: int,
    h3_resolution: int,
    precision: int,
    exemplars: int,
    verbose: bool,
) -> dict:
    cmd = [
        str(psm_binary),
        "-f", str(features_h5),
        "-g", group,
        "-t", f"{time_window:.6f}",
        "-r", str(h3_resolution),
        "-C", str(capacity),
        "-p", str(precision),
        "--top", str(top),
        "--exemplars", str(exemplars),
        "--search", str(query_path),
        "-j",
    ]
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"psm failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("features", type=Path, help="HDF5 features file (must contain --group)")
    ap.add_argument("questions", type=Path, help="YAML/JSON question file")
    ap.add_argument("--group", default="clip")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--time-window", type=float, default=75.0)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--h3-resolution", type=int, default=10)
    ap.add_argument("--precision", type=int, default=10)
    ap.add_argument("--exemplars", type=int, default=8)
    ap.add_argument("--psm-binary", type=Path, default=REPO_ROOT / "targets" / "psm")
    ap.add_argument("--clip-checkpoint", default="openai/clip-vit-base-patch32")
    ap.add_argument("--clip-device", default="auto")
    ap.add_argument(
        "--iou-threshold", type=float, default=None,
        help="override Hit@k threshold (default: from question file or 0.3)",
    )
    ap.add_argument("--out", type=Path, help="write detailed JSON record here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    spec = load_questions(args.questions)
    questions = spec.get("questions") or []
    if not questions:
        raise SystemExit(f"no questions in {args.questions}")

    if "session_start_unix" in spec:
        session_start = float(spec["session_start_unix"])
    else:
        session_start = auto_session_start(args.features, args.group)
        print(
            f"[eval] session_start_unix not specified; using {session_start:.3f} "
            f"from {args.features.name}::{args.group}",
            file=sys.stderr,
        )

    iou_threshold = args.iou_threshold
    if iou_threshold is None:
        iou_threshold = float(spec.get("iou_threshold", 0.3))

    # Late import: keeps the script lightweight when only --help is asked.
    from psm_extraction.models import make_runner

    runner = make_runner(
        family="clip",
        checkpoint=args.clip_checkpoint,
        backend="auto",
        device=args.clip_device,
    )
    print(f"[eval] CLIP runner: {runner.backend}", file=sys.stderr)

    records: list[dict] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="psm-eval-"))
    try:
        for q in questions:
            qid = q.get("id") or f"q{len(records) + 1}"
            text = q.get("query")
            if not text:
                raise SystemExit(f"question {qid!r} missing 'query'")
            gts_rel = [tuple(map(float, iv)) for iv in q.get("intervals", [])]

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
                verbose=args.verbose,
            )

            results = payload.get("results", [])
            preds = []
            for r in results:
                t_min = float(r["t_min"]) - session_start
                t_max = float(r["t_max"]) - session_start
                pred_iou, gt_idx = best_iou((t_min, t_max), gts_rel)
                preds.append({
                    "cell": r["cell"],
                    "lat": r.get("lat"),
                    "lng": r.get("lng"),
                    "similarity": r["similarity"],
                    "exemplar_t": float(r["exemplar_t"]) - session_start,
                    "t_min": t_min,
                    "t_max": t_max,
                    "count": r["count"],
                    "iou": pred_iou,
                    "matched_gt": gt_idx,
                })

            top1_iou = preds[0]["iou"] if preds else 0.0
            best_at_k = max((p["iou"] for p in preds), default=0.0)
            hit_at_k = best_at_k >= iou_threshold

            records.append({
                "id": qid,
                "query": text,
                "notes": q.get("notes", ""),
                "intervals_gt": gts_rel,
                "preds": preds,
                "iou_top1": top1_iou,
                "best_iou_at_k": best_at_k,
                "hit_at_k": hit_at_k,
            })
    finally:
        # Leave the .f32 files only if --verbose; cleanup otherwise.
        if not args.verbose:
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()
        else:
            print(f"[eval] kept queries in {tmp_dir}", file=sys.stderr)

    runner.close()

    # ---- Summary ----
    n = len(records)
    miou_top1 = sum(r["iou_top1"] for r in records) / n
    miou_at_k = sum(r["best_iou_at_k"] for r in records) / n
    hit_rate = sum(1 for r in records if r["hit_at_k"]) / n

    print()
    print(
        f"## Evaluation: {args.questions.name} on {args.features.name}::{args.group}"
    )
    print(
        f"_top={args.top}, retention={args.time_window:.1f}s × {args.capacity}, "
        f"IoU threshold={iou_threshold}_"
    )
    print()
    print("| id | query | top-1 IoU | best IoU @k | hit @k | top-1 cell | top-1 t_range (s) |")
    print("|---|---|---|---|---|---|---|")
    for r in records:
        top1 = r["preds"][0] if r["preds"] else None
        cell = top1["cell"] if top1 else ""
        t_range = f"[{top1['t_min']:.1f}, {top1['t_max']:.1f}]" if top1 else ""
        print(
            f"| `{r['id']}` | {r['query']} | {r['iou_top1']:.3f} | "
            f"{r['best_iou_at_k']:.3f} | {'✓' if r['hit_at_k'] else '✗'} | "
            f"`{cell}` | {t_range} |"
        )
    print()
    print(f"**mIoU @1**: {miou_top1:.3f}")
    print(f"**mIoU @{args.top}**: {miou_at_k:.3f}")
    print(f"**Hit @{args.top}** (IoU ≥ {iou_threshold}): "
          f"{hit_rate:.1%} ({sum(1 for r in records if r['hit_at_k'])}/{n})")

    if args.out:
        out_data = {
            "features": str(args.features),
            "questions_file": str(args.questions),
            "group": args.group,
            "top": args.top,
            "time_window_sec": args.time_window,
            "capacity": args.capacity,
            "h3_resolution": args.h3_resolution,
            "precision": args.precision,
            "exemplars": args.exemplars,
            "iou_threshold": iou_threshold,
            "session_start_unix": session_start,
            "clip_checkpoint": args.clip_checkpoint,
            "clip_backend": runner.backend,
            "summary": {
                "n_questions": n,
                "miou_top1": miou_top1,
                f"miou_at_{args.top}": miou_at_k,
                f"hit_rate_at_{args.top}": hit_rate,
            },
            "records": records,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_data, indent=2))
        print(f"\n[eval] wrote detailed record to {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
