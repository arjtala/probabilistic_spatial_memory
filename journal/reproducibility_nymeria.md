# Reproducing PSM v1 Nymeria results

> This is the parallel of [`reproducibility.md`](reproducibility.md)
> for the Nymeria-based numbers in [`results_v1.md`](results_v1.md)
> and the v1 paper draft (`paper_drafts/`). The Aria-internal
> reproducibility runbook (still useful for engine-level
> validation) lives in `reproducibility.md`.

## Data layout assumptions

The runbook below assumes the cluster layout we use:

```
/datasets/nymeria_partial/                 # READ-ONLY source
└── <session_id>/
    ├── recording_head/data/data.vrs       # 8.7G, the head-camera VRS
    ├── recording_head/mps/slam/
    │   └── closed_loop_trajectory.csv     # SLAM trajectory used for both
    │                                       # lat/lng projection + narration
    │                                       # clock-rebase (see commit e98f88b)
    └── narration/atomic_action.csv         # ground-truth narration windows

/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic/   # WRITEABLE
└── <session_id>/
    ├── clip_l_features.h5                  # CLIP-L feature extraction
    └── questions.yaml                      # auto-derived from atomic_action.csv

captures/                                   # WRITEABLE
└── eval_<session_id>_pcc<cap>.json        # one per (session, cap) run
```

If your cluster uses different roots, override via the `ROOT` /
`--features` / `--questions` flags described per-step below.

## Step 1: extract CLIP-L features for the 30 Nymeria sessions

```bash
# Cluster (SLURM, H200 GPUs):
sbatch scripts/slurm/extract_nymeria.sbatch
```

`scripts/slurm/extract_nymeria.sbatch` is a SLURM array job that runs
`psm_extraction.extract` over the 30 sessions in parallel. Each task
takes ~5 min wall (CLIP-L embed + h5 write), 16-way concurrent by
default. The orchestrator detects the `recording_head/data/data.vrs`
layout automatically via `_locate_vrs_file` and reads the SLAM
trajectory from `recording_head/mps/slam/closed_loop_trajectory.csv`
for per-frame lat/lng projection.

Verify all 30 land with non-collapsed lat/lng before proceeding:

```bash
python scripts/verify_nymeria_extractions.py
# Expect: "OK 30/30 sessions verified"
```

## Step 2: generate per-session questions.yaml

```bash
python scripts/nymeria_atomic_to_questions.py \
    /datasets/nymeria_partial \
    --out-root /checkpoint/.../nymeria_atomic
```

Parses each `narration/atomic_action.csv`, drops degenerate
intervals, and rebases narration timestamps by subtracting the
trajectory CSV's first `tracking_timestamp_us` so they land on the
extractor's 0-relative timeline (see commit `e98f88b`). Without
this rebase narrations would be unphysical (e.g. `t=33446s` for a
1207s-long recording).

## Step 3: PSM-only operating-point sweep

Single session, four cap values (the headline table):

```bash
python scripts/sweep_per_cell_cap.py \
    --take 20230608_s0_shelby_arroyo_act0_3ciwl8 \
    --caps 1 2 3 5
```

Outputs: `captures/eval_<sid>_pcc{1,2,3,5}.json`. Total wall: ~12 min
on CPU (CLIP-L text-embed dominates).

Multi-session generalization (4 sessions, same caps):

```bash
python scripts/multisession_per_cell_cap_sweep.py \
    --sessions \
        20230608_s0_shelby_arroyo_act0_3ciwl8 \
        20230607_s0_james_johnson_act0_e72nhq \
        20230609_s0_angela_harrell_act4_egucf6 \
        20230612_s0_jason_smith_act3_c6na21 \
    --caps 1 2 3 5
```

Total wall: ~50 min on CPU. Outputs land under
`captures/multisession_pcc_sweep/<sid>/`.

## Step 4: brute-force CLIP baseline (oracle)

```bash
TAKE=20230608_s0_shelby_arroyo_act0_3ciwl8
python scripts/eval_brute_force_clip.py \
    /checkpoint/.../nymeria_atomic/$TAKE/clip_l_features.h5 \
    /checkpoint/.../nymeria_atomic/$TAKE/questions.yaml \
    --top 5 --clip-device cpu \
    --clip-checkpoint laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
    --out /tmp/smoke_brute_$TAKE.json
```

Must pass `--clip-checkpoint` explicitly — the brute-force script's
default is bigG (1280-dim), which mismatches the CLIP-L features
in the h5.

## Step 5: PSM → Gemini MLLM rerank sweep

Requires `GEMINI_API_KEY` in env (`export GEMINI_API_KEY="..."`).

Single session, both cap endpoints:

```bash
tmux new -s mllm
python scripts/eval_psm_mllm.py \
    /checkpoint/.../nymeria_atomic/$TAKE/clip_l_features.h5 \
    /checkpoint/.../nymeria_atomic/$TAKE/questions.yaml \
    --mllm gemini --top 5 --per-cell-cap 5 \
    --h3-resolution 12 --time-window 30 --capacity 60 \
    --exemplars 1024 --exemplar-tolerance 5.0 \
    --clip-device cpu \
    --out captures/eval_${TAKE}_mllm_pcc5.json --verbose
# Detach with Ctrl-B then D; resume with `tmux attach -t mllm`.
```

Wall: ~30 min per (session, cap). The VrsFrameSource caches all
frame timestamps at provider-open time (commit `e41fed6`) and uses
`np.searchsorted` for O(log N) frame lookup, which is the
load-bearing perf optimization — without it, the linear-scan
baseline took ~3h per session.

Multi-session:

```bash
tmux new -s mllm-multi
python scripts/multisession_psm_mllm_sweep.py
# Default: 4 sessions × {cap=1, cap=5}, ~4 hours wall.
```

## Step 6: read the numbers

```bash
# Per-cap summary for one session:
for f in captures/eval_${TAKE}_pcc*.json; do
    cap=$(basename "$f" .json | sed 's/.*pcc//')
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
s = d['summary']
print(f\"pcc=$cap  hit@5={s['exemplar_hit_rate_at_5']*100:5.1f}%  bucket_mIoU@5={s['bucket_miou_at_5']:.3f}  exemplar_mIoU@5={s['exemplar_miou_at_5']:.3f}\")
" "$f"
done

# Multi-session summary (PSM-only):
python scripts/multisession_per_cell_cap_sweep.py --help
# (the script also re-emits the summary table at the end if you re-run it)
```

Expected (Nymeria `shelby_arroyo_act0`, single session):

| `per_cell_cap` | Hit@5 (ex=128) | Hit@5 (ex=1024) | exemplar mIoU@5 (ex=1024) |
|---|---|---|---|
| 1   | 9.1%  | 8.0%   | 0.044 |
| 2   | 9.6%  | 10.2%  | 0.055 |
| 3   | 10.2% | 10.7%  | 0.060 |
| 5   | 11.2% | 13.4%  | 0.074 |

`brute_force_clip` reference: 13.4% Hit@5 / 0.074 exemplar mIoU.

## Caveats reproducing on a different cluster

- **Disk space:** the 30 Nymeria sessions are ~805 GB raw VRS. The
  features.h5 outputs total ~150 MB.
- **CLIP-L model download:** first ingest pulls the LAION CLIP-L
  weights (~1.6 GB) from HuggingFace. Set `HF_HOME` to a fast
  shared cache.
- **VRS dependency:** `pip install projectaria-tools` is required;
  the engine itself does not depend on it (only the
  feature-extractor entry point does).
- **MLLM dependency:** `pip install requests` for the
  `scripts/_mllm_client.py` HTTP client. No SDK lock-in.
