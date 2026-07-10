# Genuine look-back QA — annotation & evaluation protocol

Addresses the reviewers' proxy-benchmark concern (WearablesAI ECCV 2026 round):
Nymeria narrations are third-person action descriptions, not genuine look-back
questions. This protocol produces a small evaluation on *real* look-back queries.

## Why this is cluster-only
The CLIP features (`clip_l_features.h5`) and session footage live under the
cluster dataset ROOT (`/checkpoint/dream/arjangt/video_retrieval/nymeria_atomic`),
not in the paper clone. Authoring genuine (query, ground-truth-interval) pairs
requires watching the footage, and scoring requires running `targets/psm`. Both
must happen on the cluster. **No numbers should be written into the paper until
this run produces `eval_genuine_<session>.json`.**

## Steps
1. **Sessions**: pick 1-3 revisit-rich sessions (start with `shelby_arroyo_act0`).
2. **Annotate**: copy `template_questions.yaml` to `<ROOT>/<session>/genuine_questions.yaml`
   and author 15-20 genuine look-back questions with **verified** intervals
   (seconds from session start). Never invent an interval.
3. **Build** the engine if needed: `make` → `targets/psm`.
4. **Run** per session:
   ```
   python scripts/eval_lookback.py \
     --session <ROOT>/<session> \
     --questions <ROOT>/<session>/genuine_questions.yaml \
     --psm-binary targets/psm --out eval_genuine_<session>.json
   ```
5. **Report**: send back `eval_genuine_<session>.json` (Hit@k, exemplar/bucket
   mIoU, per-category breakdown). I will add a "Genuine look-back QA" paragraph
   to §4 (results) with the real numbers and a one-line honesty note on sample size.

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
