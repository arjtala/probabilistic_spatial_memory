# Preregistration — Does spatial allocation improve bounded-memory retention in the long multi-revisit regime?

**Status:** venue-neutral. Written 2026-08-08, after Wearable AI @ ECCV 2026 reject
(ratings 6/5/4). Frozen *before* the run so either outcome is publishable and we
cannot fish for the operating point that flips a null.

## 1. Motivation
Submitted result (tag `wearableai2026-submission`, §5 `tab:fixed-budget`): on **14
short, single-visit** street-scale sessions, budget-matched `spatial_priority` is
flat vs `global_reservoir` on aggregate Hit@5 (−0.3 pp [−2.3, 2.0]); the rare-place
gain is +4.4 pp **[−1.3, 10.3] — CI crosses zero**; the only surviving effect is
common-place recall getting *worse* (−2.7 pp [−5.1, −0.4]); and a non-spatial
coreset `semantic_kcenter` is **best overall** (+4.1 pp [1.5, 6.7] over spatial).
On this corpus the spatial machinery does not earn its complexity.

The untested variable is the **corpus, not the method**: 26/30 Nymeria sessions have
≤9.4 m bounding boxes (a temporal-localization regime), and all 14 street-scale
sessions are short and single-visit. A spatial prior can only be exercised where the
same place recurs across time. This preregisters the decisive test of whether the
**long multi-revisit** regime changes the verdict.

## 2. Hypothesis (falsifiable, directional)
- **H1:** On long multi-revisit sessions, budget-matched `spatial_priority` improves
  **rare-place Hit@5** over `global_reservoir` with a 95% paired session-bootstrap CI
  **excluding zero**, **without** aggregate Hit@5 regression (aggregate CI lower
  bound ≥ −1 pp), AND the rare-place effect **beats the coordinate-permutation nulls**
  and **survives** grid translation/rotation.
- **H0 (null):** any clause of H1 fails.

## 3. Corpus & annotation — the gating prerequisite (cluster-side)
"Long multi-revisit session" = **≥30 min ingested** AND **cells revisited on ≥2
temporally-separated passes** (revisit *structure*, not just duration).

Neither route exists yet: PROTOCOL.md's hard gate was never passed (no
`eval_genuine_*.json`; only `template_questions.yaml`), and the GPS-grounded
`last_seen` generator is unbuilt (TODO). This annotation is the **weeks-scope cost**,
and it is required for **any** venue — RcVB's proxy-benchmark objection does not go
away by moving venues.

### 3a. IMPORTANT — cohort, not one session
`compare_fixed_budget.py` computes the CI by **resampling sessions** ("never
questions or seeds"). A **single** session gives n=1, so the pre-registered CI is
**undefined**. The decisive test therefore needs a **cohort of long multi-revisit
sessions** — target **≥5, spanning ≥2 capture/localization substrates** — so the CI
exists and yhV1's "may not generalize" confound is answered. Each session's
`genuine_questions.yaml` is frozen per PROTOCOL.md (sha256, 20% second-annotator
check).
*(Fallback if only 1–2 long sessions are feasible: add a within-session
question-bootstrap mode to the harness and preregister that instead — a weaker,
single-session inference. Flagged, not preferred.)*

## 4. Design (frozen)
- **Policies:** `global_reservoir` (budget-matched baseline), `spatial_priority`
  (probe), `semantic_kcenter` (the actual best-overall competitor — included so any
  "spatial wins" claim must contend with it).
- **Budget:** M=128 primary; M=64 secondary (submitted budget-interaction hint).
- H3 `r=12`; `K=5`; CLIP-L; 5 seeds; global cosine, **no per-cell cap** (identical
  retrieval across policies, so differences reflect only *which* frames are retained).
- Only `similarity_search` questions enter the matched comparison (per PROTOCOL).
- **Unit of analysis:** sessions (seeds averaged within session, then paired session
  bootstrap).

## 5. Strata (frozen; identical under all controls)
Target place = modal canonical H3 cell in the GT interval. `rare`/`common` = bottom/
top exposure quartiles of that cell's frame count within the session.

## 6. Controls (place-specificity guards)
Coordinate-permutation nulls (seeds 101/202/303), grid translation (+4.5 m E and N),
rotation (30°). The rare-place effect must **exceed** the perm-nulls and be **~0**
under translation/rotation to count as place-specific (not partition-balancing or an
H3-boundary artifact).

## 7. Exact commands (cluster-side; data under `$PSM_DATA_ROOT`)
```bash
ROOT="$PSM_DATA_ROOT/video_retrieval/<corpus>" \
QUESTIONS_NAME=genuine_questions.yaml \
OUT_ROOT=captures/decisive_longrevisit \
METHODS=global_reservoir,spatial_priority,semantic_kcenter \
BUDGETS="128 64" SEEDS=0,1,2,3,4 H3_RESOLUTIONS=12 CLIP_DEVICE=cuda \
EXTRA_ARGS="--translate-east-m 4.5 --translate-north-m 4.5 --rotation-deg 30 \
--coord-permutation-seed 101 --coord-permutation-seed 202 --coord-permutation-seed 303" \
bash scripts/run_wearables_budget_suite.sh <session_ids...>
```
Inference (paired session bootstrap):
```bash
python scripts/compare_fixed_budget.py captures/decisive_longrevisit/**/*.json \
  --method-a spatial_priority --method-b global_reservoir
python scripts/compare_fixed_budget.py captures/decisive_longrevisit/**/*.json \
  --method-a spatial_priority --method-b semantic_kcenter
```
SLURM: `scripts/slurm/fixed_budget_suite.sbatch`.

## 8. Preregistered decision rule (set BEFORE the run)
- **GO / positive regime** — H1 holds (rare-place CI > 0, no aggregate regression,
  beats perm-nulls, survives boundary controls). Claim = *"spatial allocation
  improves rare-place recall in the long multi-revisit regime"* (scoped; if
  `semantic_kcenter` still wins aggregate, the claim stays rare-place-specific, not
  "beats all baselines"). Positive-claim venue viable.
- **NO-GO / null** — any H1 clause fails. Report the negative result: *"budget-matched
  across substrates, spatial structure does not improve bounded-memory retention for
  wearable look-back QA; generic semantic diversity does."* The methodology
  (budget-matched design, permutation nulls, translation/rotation controls, paired
  session bootstrap, pre-registered strata) is the contribution. Negative/evaluation
  venue.
- **No post-hoc operating-point search.** M, r, strata, controls, and this rule are
  frozen; report the M=128 / r=12 point regardless of any other point.

## 9. Prior
Author's prior on the null: **> 0.5** even in the long regime. Preregistering the
rule is what makes both outcomes credible.
