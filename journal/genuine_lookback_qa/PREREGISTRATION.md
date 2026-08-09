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

PROTOCOL.md's hard gate has never been passed (no `eval_genuine_*.json`; only
`template_questions.yaml`). The GPS-grounded `last_seen` generator now exists
(`scripts/generate_revisit_questions.py`), which makes the *proxy* cohort
affordable — but it does **not** discharge the annotation requirement. Human
authoring is still required for **any** venue: RcVB's proxy-benchmark objection
does not go away by moving venues, and it does not go away by automating the
proxy. See §3b — the two banks do different jobs and both are preregistered.

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

### 3b. Two banks, both preregistered — and why construction must be H3-independent
The test runs on **two question banks**, and the reporting rule in §8 spans both.

| Bank | Source | Job |
|---|---|---|
| **Proxy** | `generate_revisit_questions.py`, ≥5-session cohort | Powers the session-bootstrap CI. Makes the cohort affordable. |
| **Validity** | Hand-authored per PROTOCOL.md, 1–2 sessions | Independent check that a proxy result is not a construction artifact. Answers RcVB. |

**Construction independence (load-bearing).** The proxy generator must select
places by **metric distance on the raw track**, never by H3 cell identity. The
first cut of the generator reused the evaluator's `h3_cells` / `group_indices`
to guarantee no drift; that is the wrong trade. It would make the bank a
function of the same H3 partition and cell-exposure statistics that (a) define
the rare/common strata in §5 and (b) *are* the mechanism of `spatial_priority`.
H1 would then reduce to "does a policy that retains frames from low-exposure H3
cells score well on questions selected to sit in low-exposure H3 cells?" — which
can come out positive by construction. It would also blunt §6's coordinate-null:
the permuted arm scrambles the policy's spatial view while the GT intervals still
encode true H3-derived structure, so the control no longer isolates
place-specificity.
Scoring is unaffected — `eval_fixed_budget.py` still computes cells and strata
with `h3_cells` exactly as before. Only *construction* is decoupled. Any H3
figure emitted by the generator is a diagnostic and must never feed selection.

**The validity bank is the reason this still is not sufficient on its own.**
Even metric-clustered proxy questions are generated from the trajectory, so they
inherit *some* spatial structure. Human-authored questions are the only ones
constructed independently of the track, which is why §8 makes them the
tiebreaker rather than an appendix.

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

**Step 0 — build the proxy bank per candidate session, then select the cohort.**
Keep only sessions reporting `long_multirevisit=true`; assemble ≥5 across ≥2
substrates. Record each `questions_sha256` before running any policy.
```bash
python scripts/generate_revisit_questions.py \
  --features "$PSM_DATA_ROOT/video_retrieval/<corpus>/<session>/clip_l_features.h5" \
  --out      "$PSM_DATA_ROOT/video_retrieval/<corpus>/<session>/revisit_questions.yaml" \
  --metadata-out captures/revisit_meta/<session>.json \
  --place-radius-m 15 --visit-gap-sec 30 --min-separation-sec 120 \
  --n-questions 20 --seed 0
```
Places are metric clusters on the raw track (§3b); `--h3-resolution` is a
sidecar diagnostic only and must not be used to select sessions or questions.

**Step 1 — the suite.** Run once per bank: `QUESTIONS_NAME=revisit_questions.yaml`
for the proxy cohort, and `QUESTIONS_NAME=genuine_questions.yaml` for the
validity sessions (§8a). Use a distinct `OUT_ROOT` per bank so captures never
pool.
```bash
ROOT="$PSM_DATA_ROOT/video_retrieval/<corpus>" \
QUESTIONS_NAME=revisit_questions.yaml \
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

### 8a. Two-bank reporting rule (frozen — deciding this after the run is the exact
### post-hoc move this document exists to prevent)
H1 is evaluated on the **proxy cohort** (the CI-bearing test) and **reported
alongside** the validity bank on every occasion it is stated. The validity bank
is the **tiebreaker**, not a robustness footnote:

| Proxy cohort | Validity bank | Verdict |
|---|---|---|
| H1 holds | effect same direction | **GO.** Positive claim, scoped as in §8. |
| H1 holds | effect absent / opposite | **NO-GO for the positive claim.** Report as a construction-sensitive proxy effect that does not replicate on human questions. This is a *finding*, not a failed experiment. |
| H1 fails | either | **NO-GO / null**, per §8. |

The validity bank is small (1–2 sessions, 15–20 questions each), so it will not
carry its own session-bootstrap CI and **must not be given one**. It is read as
direction-of-effect plus per-question detail only. Underpowering is not licence
to discount it when it disagrees — a disagreement is the informative outcome and
is reported as such.

Both banks' sha256 hashes are recorded in the results table. Neither bank may be
regenerated, reseeded, or re-authored after any policy has been run against it.

## 9. Prior
Author's prior on the null: **> 0.5** even in the long regime. Preregistering the
rule is what makes both outcomes credible.

## Amendment A — 2026-08-09 (corpus reality: HDD-first two-phase; coordinate freeze)

Records decisions forced by the Step-0 survey. It does **not** relax H1, the §8
go/no-go rule, or §8a; §1–9 above are unchanged.

**A1. The ≥2-substrate cohort is unreachable from existing extractions.** The
Step-0 sweep (all 8 CLIP-L corpora, frozen params) found the gate blocked on
**duration, not revisit structure**: no wearable session clears the ≥30-min clause
(max minutes — Nymeria 23.5, aria 15.1, LookOut 10.7, SLOPER4D 7.3,
aria_gen2_pilot 5.7; the lone 42-min EgoExo take has zero separated revisits).
Only **HDD** qualifies — 60 drives, a single driving/GPS substrate. The §3a
"≥5 across ≥2 substrates upfront" target cannot be met now; the ≥2-substrate
requirement moves to Phase 2.

**A2. Two-phase sequencing.**
- **Phase 1 — HDD existence gate (runs now).** Cohort = the 60-drive
  `hdd_longrevisit_v1` (manifest `journal/genuine_lookback_qa/hdd_cohort_v1_manifest.json`;
  1101 questions; gate `duration_sec≥1800 AND n_revisited_places≥5`; generator
  `e78adf8`; per-session `questions_sha256` + `gps_coords_sha256` frozen). n=60
  ≫ the `--min-sessions=2` floor, so the paired session-bootstrap CI is valid and
  well-powered on HDD alone. Phase 1 tests **existence** — does spatial allocation
  help *at all* in a long-multi-revisit regime — not generalization.
- **Phase 2 — wearable generalization (conditional).** Runs **only if Phase 1 is
  GO.** Requires ≥1 long-form wearable substrate (capture/extraction — weeks) for
  yhV1's confound, plus the §8a human-authored validity bank (native to wearable
  look-back QA, not to driving). A Phase-1 GO is the *trigger* for that investment;
  a Phase-1 null needs no Phase 2.

**A3. Null scope (amends §8 wording for Phase 1).** HDD is 1-fps windshield video
with **road-corridor** revisits. A Phase-1 null is strong — "no benefit even in the
most revisit-rich data in hand" — but is scoped to **driving-corridor revisits**,
not wearable-universal. The §8 NO-GO phrase "across substrates" applies only after
Phase 2; a Phase-1-only null is reported as single-substrate (HDD) and must not
imply wearable coverage.

**A4. Validity bank in Phase 1.** Human-authored look-back QA is not native to HDD
driving footage, so the §8a validity-bank tiebreaker cannot be exercised in
Phase 1. Frozen consequence: a Phase-1 **positive** on the proxy cohort **cannot
clear §8a on its own** — it only licenses Phase 2 (where the validity bank lives).
Phase 1's decisive outcomes are therefore (i) **null → hypothesis dead**, or
(ii) **positive → run Phase 2**; Phase 1 alone never yields a standalone positive
claim.

**A5. Coordinate source is part of the frozen bank definition.** Banks were
regenerated on **at-rest-corrected** coordinates after the `astimezone()` clock bug
(`io/hdd.py`, fixed in `7480050`; originals preserved under
`provenance/gps_20260705/`, guard `assert_track_coverage` added). Each bank + sidecar
carries `coord_provenance = io/hdd.py@7480050` and `n_frames_without_coordinate`.
The pre-fix (skewed) banks and the 94+7 survey YAMLs are **void**; no policy may run
against skewed-coord banks. Freeze key = (place-radius, visit-gap, min-separation,
n-questions, seed, **coordinate provenance**).

**A6. Cohort membership + exclusions (frozen selection).** 60/132 drives pass (not
the 59 first reported off skewed coords; the boundary flip is `201706081335`, exactly
5 revisited places). Handled explicitly: **2 stationary drives** (RTK bbox extent
~0 m — genuinely parked, correctly not counted as revisits, distinct from a collapsed
track) are excluded; **`201710040938`** carries a real 527-s RTK-acquisition lag, its
uncovered frames written NaN (never fabricated) and barred from anchoring queries/GT.
These are selection decisions, frozen here.

**A7. §6 factual correction (for any resubmission).** The submitted §6 sentence
"corpus-scale CLIP features … not available (only a sanity drive was extracted)" was
**inaccurate at submission**: the full 132-drive corpus was extracted 2026-07-05,
twelve days before c20094d. HDD is engine-runnable. The withdrawn HDD cardinality/AUC
numbers stay withdrawn until **rerun through the engine** with the k-nearest-cell +
VPR controls (the offline-Python-substitute half of the withdrawal stands); correct
the availability sentence, and keep docs/HDD.md:264 "REAL RESULTS" void pending that
rerun.

## Amendment B — 2026-08-09 (proxy metric: captioned Hit@5, not retention; viability gate)

The GPS-`last_seen` proxy (§3b/§7) cannot score Phase 1: `last_seen` carries no text
query, so `eval_fixed_budget.py:708` keeps zero of its questions, and H1 is stated in
**Hit@5** (a text→frame retrieval metric). Any `last_seen` score is circular:
`oracle_retained` **rewards `spatial_priority`'s own objective** (rare stratum =
low-exposure cells = exactly what the policy protects → positive by construction, the
§3b hazard moved into the metric), and a spatial retrieval rule favours the spatial
policy in scoring. Retention is therefore **demoted to a labelled
allocation-retention diagnostic, never a headline**.

**B1. Metric stays Hit@5.** Phase 1 proxy = **captioned `similarity_search`** on the
HDD cohort, retrieved by CLIP text→frame cosine (**non-spatial**, so no policy is
favoured by construction), scored Hit@5 — continuous with §5 `tab:fixed-budget`; the
only changed variable is session length/revisit-richness (the preregistered variable).

**B2. Frame selection is H3-independent (§3b).** Caption frames are sampled inside
**metric-revisited places** (`metric_places`/`revisited_places`), ≥`min-separation-sec`
apart (`--place-selection metric`); H3 stays a diagnostic, never a selector.

**B3. GT interval = ALL visits of the place (frozen).** A caption of visit B also
matches near-duplicate visit A; scoring only the captioned visit penalises policies
that retain *more* frames of the place (more competing near-duplicates) — a
construction bias running **against** `spatial_priority`, the mirror of the retention
tautology. GT = union of the place's visit intervals → measures **place recall**
("where was I last at this place"), which is H1. (Captioned-visit-only measures
episode discrimination, which on repetitive driving is encoder-limited, not
policy-limited — not our question.)

**B4. Policy-free viability gate (frozen, pre-run).** Caption viability is tested with
**no policy**: caption N frames, CLIP-text-embed, retrieve over the **full unbudgeted
bank**; this is the retrieval *ceiling* (M=128 retention can only degrade it). **Pass
requires both:** (i) **≥50%** of captions retrieve their GT interval at **top-5** on
the full bank; (ii) median pairwise caption CLIP-text cosine **≤ 0.80** (guards against
"every caption says road-with-power-lines"). (i) is decisive and governs; (ii) is a
secondary degeneracy guard. Failing (i) means no budget or policy can produce a
measurable `spatial_priority`-vs-`global_reservoir` difference → the corpus cannot
support B.

**B5. Caption model + prompts frozen by SHA.** Model `gemini-3-1-pro-preview-genai`.
Prompts are frozen artifacts: `caption_prompt_driving_v1.txt` (distinctiveness, text
allowed), `v2.txt` (appearance-only, text banned), `v3.txt` (brand-recognition). The
prompt behind any scored bank is recorded by SHA; **no prompt tuning after a bank is
scored**.

**B6. Pilot outcome (3 drives, policy-free full-bank ceiling).**
- v1: top-5 = **26%** (5/19). v2: top-5 = **12%** (4/34). v3: **pending — the last
  prompt attempt**.
- Mechanism = a **pincer**: v1's discriminative content is *text*, which CLIP-L cannot
  read ("Pine Street" sign → rank 1611/10165); v2's CLIP-encodable content is *generic
  architecture*, non-unique in Bay Area suburbia (tan two-storey building → rank 1006).
  Specific-but-unreadable, or readable-but-non-unique — prompt engineering moves along
  the tradeoff, not off it.

**B7. Pre-committed branch (frozen; no fishing).** **v3 is the last prompt (no v4+).**
If no prompt clears B4 on ≥5 qualifying drives → **B is not viable on HDD**, and the
decisive test moves to **Phase 2 (wearable capture/extraction)**. This is a reportable
finding with a mechanism — *"1-fps windshield video is text-discriminative and
appearance-generic, so CLIP-grounded look-back QA is not constructible on it"* — not a
failed experiment to bury. `oracle_retained` may accompany only as a labelled
allocation-retention diagnostic (B-preamble), never as the headline.
