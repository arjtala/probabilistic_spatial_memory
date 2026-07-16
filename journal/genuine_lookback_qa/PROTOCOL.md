# Genuine look-back QA — annotation & evaluation protocol

Addresses the reviewers' proxy-benchmark concern (WearablesAI ECCV 2026 round):
Nymeria narrations are third-person action descriptions, not genuine look-back
questions. This protocol produces a small evaluation on human-authored,
video-verified look-back questions.

## Why this is cluster-only
The CLIP features (`clip_l_features.h5`) and session footage live under the
cluster dataset ROOT (`$PSM_DATA_ROOT/video_retrieval/nymeria_atomic`),
not in the paper clone. Authoring genuine (query, ground-truth-interval) pairs
requires watching the footage, and scoring requires running `targets/psm`. Both
must happen on the cluster. **No numbers should be written into the paper until
this run produces all five `eval_genuine_<session>_s*.json` files.**

## Steps
1. **Sessions**: pick 1-3 revisit-rich sessions (start with `shelby_arroyo_act0`).
2. **Annotate**: copy `template_questions.yaml` to `<ROOT>/<session>/genuine_questions.yaml`
   and author 15-20 genuine look-back questions with **verified** intervals
   (seconds from session start). Never invent an interval.
3. **Verify and freeze**: have a second annotator independently check at least
   a random 20% of questions, resolve disagreements against the footage, then
   record `sha256sum genuine_questions.yaml`. Do not edit after evaluation.
4. **Build** the engine if needed: `make` → `targets/psm`.
5. **Run** five reservoir seeds per session:
   ```bash
   for seed in 0 1 2 3 4; do
     python scripts/eval_lookback.py \
       <ROOT>/<session>/clip_l_features.h5 \
       <ROOT>/<session>/genuine_questions.yaml \
       --psm-binary targets/psm \
       --clip-checkpoint laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
       --top 5 --time-window 30 --capacity 60 \
       --h3-resolution 12 --exemplars 128 --per-cell-cap 5 \
       --seed "$seed" --out "eval_genuine_<session>_s${seed}.json"
   done
   ```
6. **Report**: average seeds within each session and retain all five JSON files
   (Hit@k, exemplar/bucket mIoU, per-category breakdown). Add a
   "Human-authored look-back QA" paragraph with an explicit sample-size caveat.

## Fixed-budget companion

Run the matched fixed-M policies on the same frozen questions. The corpus
driver accepts an alternate question filename:

```bash
ROOT="$PSM_DATA_ROOT/video_retrieval/nymeria_atomic" \
QUESTIONS_NAME=genuine_questions.yaml \
OUT_ROOT=captures/genuine_fixed_budget \
BUDGETS=128 H3_RESOLUTIONS=12 CLIP_DEVICE=cuda \
bash scripts/run_wearables_budget_suite.sh <session> [...]
```

Only `similarity_search` questions enter the matched policy comparison.
Evaluate `query_mode: last_seen` separately as a PSM-only spatial task.
If one YAML contains both modes, do not quote its mixed overall summary; split
or aggregate records by `query_mode` first.

## Companion: 30-session confidence intervals (reviewers' other ask)
`scripts/aggregate_cap_sweep_30.py` now emits mean±std and 95% CI per
`per_cell_cap`, plus a ready-to-paste supp §A.1 sentence. Regenerate on the
cluster:
```
bash scripts/multisession_cap_sweep_30.sh   # repopulates captures/multisession_pcc_sweep/
python scripts/aggregate_cap_sweep_30.py    # prints mean±95%CI + paste-ready sentence
```
Send me the printed CI line and I will insert it into supp §A.1 (and optionally
the abstract). These CIs cannot be computed in the clone — the per-session
sweep outputs are not committed.
