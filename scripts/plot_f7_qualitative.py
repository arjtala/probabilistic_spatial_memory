#!/usr/bin/env python3
"""F7 paper figure: qualitative top-5 PSM candidates on shelby_arroyo_act0.

A 3-row x 6-column grid. Each row is one query from the headline operating
point (PSM cap=K + Gemini rerank). Columns:
  - col 1: the query text in a small text panel.
  - cols 2-6: the K=5 PSM candidate frames, annotated with:
      * frame timestamp ("t=NNN.Ns")
      * cosine similarity ("sim=0.XXX")
      * green border + checkmark if exemplar_hits_gt
      * red border + cross otherwise
      * gold "* Gemini" badge on the candidate mllm_pick_idx selected.

The three rows illustrate three patterns:
  R1: PSM hit + Gemini picked the hitting frame -> success case.
  R2: PSM hit + Gemini picked a non-hitting frame -> Hit@5 invariant,
      Hit@1 misses (bimodal rerank story from sec:results-mllm-rerank).
  R3: PSM all-miss with revisit-heavy candidates -> honest failure
      mode + benchmark ceiling (sec:results-memory-latency).

Inputs:
  - eval JSON: captures/multisession_psm_mllm/<sess>/eval_<sess>_mllm_pcc5.json
  - frame JPEGs: captures/mllm_baseline/frames_baseline/<sess>/frame_NNNNNN.jpg

The session was sampled at sample_fps=1.0 (read from the H5 attrs at
clip_l_features.h5 -> clip.attrs.sample_fps) which yields a sampling
period of ~1.0665s (1133 frames over 1207.273s). We use
  frame_idx = round(exemplar_t / 1.0665)
and clamp to [0, 1132]. The mapping was spot-checked against the H5
timestamps: target_t=279.422 -> idx 262 (ts[262]=279.422), etc.

Output:
  journal/figures/f7_qualitative.{pdf,svg}

Usage:
  python scripts/plot_f7_qualitative.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# Headline operating-point eval file.
EVAL_REL = (
    "captures/multisession_psm_mllm/20230608_s0_shelby_arroyo_act0_3ciwl8/"
    "eval_20230608_s0_shelby_arroyo_act0_3ciwl8_mllm_pcc5.json"
)
FRAMES_REL = (
    "captures/mllm_baseline/frames_baseline/"
    "20230608_s0_shelby_arroyo_act0_3ciwl8"
)
N_FRAMES = 1133
SESSION_DUR = 1207.273  # seconds, from H5 timestamps[-1]
SAMPLE_PERIOD = SESSION_DUR / (N_FRAMES - 1)  # ~1.0665 s


# Story selection: ids hand-picked from the candidate scan
# (see report comment block at the end of this docstring for full audit).
# R1: q96  - clean PSM hit (only candidate 2 hits GT) + Gemini picked it.
# R2: q31  - "C closes the cabinet" -- 2 candidates hit GT (idx 2 and 4)
#            but Gemini picked candidate 3 (a non-hit) -> Hit@1 miss.
# R3: q1   - revisit-heavy living-room narration with all 5 PSM
#            candidates same-cell, none hitting GT 298-303s window.
SELECTED_IDS = ["q96", "q31", "q1"]
SELECTED_TAGS = [
    "PSM hit, Gemini hit",
    "PSM hit, Gemini missed Hit@1",
    "PSM miss (revisit-heavy)",
]


def t_to_frame_idx(t: float) -> int:
    idx = int(round(t / SAMPLE_PERIOD))
    return max(0, min(N_FRAMES - 1, idx))


def _truncate(s: str, n: int = 110) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "..."


def _wrap(s: str, width: int = 28) -> str:
    """Basic word-wrap for the query text panel."""
    out: list[str] = []
    line = ""
    for word in s.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        out.append(line)
    return "\n".join(out)


def build_figure(
    eval_path: Path,
    frames_dir: Path,
    selected_ids: list[str],
    selected_tags: list[str],
    out_path: Path,
) -> dict:
    data = json.loads(eval_path.read_text())
    qs = data["questions_out"]
    by_id = {q["id"]: q for q in qs}
    missing = [qid for qid in selected_ids if qid not in by_id]
    if missing:
        raise SystemExit(f"missing question ids in eval JSON: {missing}")

    n_rows = len(selected_ids)
    n_cols = 6  # 1 text + 5 frames

    # Single-column LNCS width ~4.8". Tight grid.
    fig_w = 4.8
    fig_h = 1.05 * n_rows + 0.25
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1, 1]},
    )
    if n_rows == 1:
        axes = [axes]

    summary_rows: list[dict] = []

    for ri, (qid, tag) in enumerate(zip(selected_ids, selected_tags)):
        q = by_id[qid]
        preds = q["preds"][:5]
        pick_1idx = q.get("mllm_pick_idx")
        gt_intervals = q.get("intervals_gt") or []
        # Text panel.
        ax_txt = axes[ri][0]
        ax_txt.set_xticks([])
        ax_txt.set_yticks([])
        for s in ax_txt.spines.values():
            s.set_visible(False)
        q_text = _wrap(_truncate(q["query"], 110), width=24)
        gt_str = ", ".join(f"[{a:.1f},{b:.1f}]s" for a, b in gt_intervals)
        ax_txt.text(
            0.0, 0.97,
            f"{qid}",
            transform=ax_txt.transAxes,
            va="top", ha="left",
            fontsize=6.5, fontweight="bold", color="#222",
        )
        ax_txt.text(
            0.0, 0.86,
            q_text,
            transform=ax_txt.transAxes,
            va="top", ha="left",
            fontsize=5.0, color="#222",
            linespacing=1.15,
        )
        ax_txt.text(
            0.0, 0.18,
            f"GT: {gt_str}",
            transform=ax_txt.transAxes,
            va="top", ha="left",
            fontsize=4.6, color="#444", fontstyle="italic",
        )
        ax_txt.text(
            0.0, 0.08,
            tag,
            transform=ax_txt.transAxes,
            va="top", ha="left",
            fontsize=4.8, color="#0a4d99", fontweight="bold",
        )

        # Frame panels.
        n_hit = 0
        gemini_hit = False
        for ci, pred in enumerate(preds):
            ax = axes[ri][ci + 1]
            t = float(pred["exemplar_t"])
            sim = float(pred["similarity"])
            hits = bool(pred.get("exemplar_hits_gt"))
            if hits:
                n_hit += 1
            frame_idx = t_to_frame_idx(t)
            frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
            if not frame_path.exists():
                raise SystemExit(
                    f"frame file missing for {qid} cand {ci+1}: {frame_path}"
                )
            img = Image.open(frame_path)
            # Downscale aggressively to keep PDF small.
            img.thumbnail((220, 220))
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            # Hit/miss border.
            border_color = "#1aaa55" if hits else "#d62728"
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_edgecolor(border_color)
                s.set_linewidth(1.6)
            # Top-left hit/miss glyph.
            glyph = "✓" if hits else "✗"  # checkmark / cross
            ax.text(
                0.04, 0.96, glyph,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=7.0, fontweight="bold", color="white",
                bbox=dict(
                    boxstyle="round,pad=0.10",
                    facecolor=border_color, edgecolor="none",
                ),
            )
            # Bottom caption: time + similarity.
            ax.text(
                0.5, -0.04, f"t={t:.0f}s | sim={sim:.3f}",
                transform=ax.transAxes,
                va="top", ha="center",
                fontsize=4.6, color="#222",
            )
            # Gemini pick badge.
            if pick_1idx is not None and pick_1idx == ci + 1:
                ax.text(
                    0.96, 0.96, "* Gemini",
                    transform=ax.transAxes,
                    va="top", ha="right",
                    fontsize=4.6, fontweight="bold", color="#222",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="#ffd84a", edgecolor="#222",
                        linewidth=0.5,
                    ),
                )
                if hits:
                    gemini_hit = True

        summary_rows.append({
            "id": qid,
            "tag": tag,
            "n_hit": n_hit,
            "gemini_pick": pick_1idx,
            "gemini_hit": gemini_hit,
            "query": q["query"],
        })

    # Column header row above the frame columns (only on row 0).
    for ci in range(1, n_cols):
        axes[0][ci].set_title(f"k={ci}", fontsize=5.5, pad=2)

    fig.subplots_adjust(
        left=0.005, right=0.995, top=0.965, bottom=0.025,
        wspace=0.06, hspace=0.32,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return {
        "out_pdf": str(out_path.with_suffix(".pdf")),
        "out_svg": str(out_path.with_suffix(".svg")),
        "rows": summary_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    repo_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--repo-root", type=Path, default=repo_root)
    ap.add_argument(
        "--out", type=Path,
        default=repo_root / "journal" / "figures" / "f7_qualitative.pdf",
    )
    args = ap.parse_args()

    eval_path = args.repo_root / EVAL_REL
    frames_dir = args.repo_root / FRAMES_REL
    if not eval_path.exists():
        raise SystemExit(f"eval JSON not found: {eval_path}")
    if not frames_dir.is_dir():
        raise SystemExit(f"frames dir not found: {frames_dir}")

    result = build_figure(
        eval_path=eval_path,
        frames_dir=frames_dir,
        selected_ids=SELECTED_IDS,
        selected_tags=SELECTED_TAGS,
        out_path=args.out,
    )

    print()
    print(f"{'id':5s}  {'tag':36s}  {'hits/5':>6s}  {'pick':>4s}  hit?")
    print("-" * 75)
    for r in result["rows"]:
        hit_mark = "yes" if r["gemini_hit"] else "no"
        print(
            f"{r['id']:5s}  {r['tag']:36s}  "
            f"{r['n_hit']:>6d}  {str(r['gemini_pick']):>4s}  {hit_mark}"
        )
    print()
    for r in result["rows"]:
        print(f"  {r['id']}: {r['query'][:110]}")
    print()
    print(f"[f7] wrote {result['out_pdf']}")
    print(f"[f7] wrote {result['out_svg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
