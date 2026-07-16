# WearablesAI Novelty Experiments

This plan targets the current paper and title. A title change is not necessary.
The goal for the remaining submission window is to turn the paper's weak point,
novelty, into a controlled scientific question:

> Under an identical wearable exemplar budget, when does spatial allocation
> preserve useful experiences better than non-spatial streaming memory?

## Deadline and Scope

- Submission: **July 25, 2026 at 11:59 PM Anywhere on Earth**.
- Los Angeles equivalent: **July 26, 2026 at 4:59 AM PDT**.
- Notification: August 8. Camera ready: August 15.
- Format: ECCV 2026 format, 14 content pages plus unlimited references.
- Official sources: [workshop CFP](https://wearable-ai-workshop.github.io/),
  [OpenReview portal](https://openreview.net/group?id=thecvf.com/ECCV/2026/Workshop/WearableAI),
  and [ECCV policies](https://eccv.ecva.net/Conferences/2026/SubmissionPolicies).

The paper directly fits the workshop topic **Persistent Scene and Object Memory
from Video**. Long-context interaction and efficient wearable inference are
secondary fits.

Do not call the current system globally memory-bounded. It is bounded per H3
cell, while the cell table and HLL rings grow with explored area. The new
experiments enforce an exact **global exemplar count** and report HLL register
bytes separately.

## Experimental Contract

Freeze these choices before looking at results:

| Axis | Primary setting | Sweep |
|---|---:|---:|
| Encoder | CLIP-L, 768-d | none before submission |
| Retrieval | global cosine top-5 | no per-cell result cap |
| Exemplar budget M | 128 | 64, 128, 256, 512 |
| H3 resolution | 12 | 10, 11, 12, 13 |
| Seeds | 0 through 4 | five common seeds |
| Visit boundary | 30 seconds outside a cell | fixed |
| Primary metric | exemplar Hit@5 | mIoU@5 secondary |

For CLIP-L, M={64,128,256,512} corresponds to approximately
{0.19,0.38,0.75,1.50} MiB of embedding-plus-timestamp state. Report M/N for
every session. M=128 is the primary low-memory point and retains continuity
with the paper's existing R=128 operating point, but it is now global rather
than per cell.

Use global top-k for this experiment. Mixing `per_cell_cap=1` into the primary
comparison would confound memory allocation with result diversification. Keep
the existing cap sweep as a separate ablation.

### Retention policies

1. `global_reservoir`: streaming Algorithm R over every frame.
2. `fifo`: the M most recent frames.
3. `hybrid`: half lifetime reservoir and half recent frames.
4. `uniform_time`: M evenly spaced frames.
5. `semantic_kcenter`: offline farthest-first cosine coreset. This tests whether
   generic visual diversity, rather than physical space, explains a gain. It
   uses the full embedding bank and is a diagnostic, not a streaming-memory
   competitor.
6. `spatial_priority`: causal exact-M allocation. A frame from an
   underrepresented cell evicts a random exemplar from a most-represented cell;
   otherwise the cell uses Algorithm-R replacement. Once visited cells exceed
   M, stable random cell priorities maintain a uniform priority sample of M
   cells instead of a recent-cell cache.
7. `spatial_balanced`: water-fill M slots across H3 cells, then reservoir
   sample within each cell.
8. `visit_balanced`: water-fill across cells, then across distinct visits to
   each cell, then sample within each visit.

`spatial_priority` is the new causal policy to test as the primary contribution.
It is still a research heuristic: quota expansion cannot recover discarded old
frames, an update may inspect O(M) retained cell banks, and its observation
counters remain O(number of visited cells). Test old versus recent strata and
do not call the whole policy globally memory-bounded.
The last two policies use the completed sequence to compute quotas. They are
**non-causal diagnostics**, not upper bounds and not deployed PSM algorithms.
Even a positive Python result must be described as an evaluated retention
policy until the same policy is integrated into the C engine.

## P0: Fixed-Budget Retention

### Hypotheses

- H1: Causal spatial-priority increases oracle retention coverage for rare places.
- H2: Higher oracle coverage translates into higher Hit@5.
- H3: Visit balancing helps old events at repeatedly visited places.
- H4: Under a fixed M, H3 accuracy has an interior optimum instead of improving
  mechanically as finer cells create more total storage.

### Metrics

- **Hit@5**: any retrieved exemplar timestamp falls in a GT interval.
- **Exemplar mIoU@5**: existing `+/-1.5 s` scorer.
- **Oracle retention coverage**: any retained timestamp falls in a GT interval,
  before semantic ranking. This separates retention from encoder failure.
- **Actual retained slots**: must equal `min(M, N)`.
- **Logical selector state**: raw float32 embedding plus float64 timestamp,
  identical across exact-M methods. Per-cell counters and containers are
  excluded and must be disclosed.
- **Modeled deployment overhead**: report
  `cells * ring_capacity * 2^precision` HLL register bytes separately. The
  diagnostic selectors do not actually allocate these HLLs.

### Fast local smoke test

Install the evaluation dependencies in the environment that has the encoder:

```bash
python -m pip install -r requirements-paper.txt
python -m pip install -e 'extraction[clip]'
```

Inspect the run matrix without loading CLIP:

```bash
python scripts/eval_fixed_budget.py \
  datasets/hdd/201703061033/clip_l_features.h5 \
  datasets/hdd/201703061033/questions.yaml \
  --budgets 64,128 \
  --seeds 0,1 \
  --h3-resolutions 10,12 \
  --methods global_reservoir,fifo,hybrid,uniform_time,semantic_kcenter,spatial_priority,spatial_balanced,visit_balanced \
  --dry-run
```

Then remove `--dry-run` and add `--clip-device cuda` or the appropriate local
device. The checked-in HDD question intervals explicitly say they were seeded
from PSM output and are not video-verified. Use this only to test plumbing, not
as a paper result.

### Corpus run order

The driver is resumable. Existing JSON files are skipped unless `FORCE=1`.
It validates input hashes, checkpoint, scoring settings, transforms, and
evaluation-code hash before reusing a capture.
Each capture records resolved model revision and core package versions; each
manifest also records `pip freeze`. Keep manifests with the reported results.

Use the paper's existing **14 street-scale sessions** as the primary corpus:
10 LookOut, 3 SLOPER4D, and the street-scale Nymeria session. Run the driver
once per dataset root with the explicit session IDs already used by Table 2.
Most of Nymeria-30 is room-scale and occupies only one or two r12 cells; use
that set as a negative/control regime, not the primary spatial test.
All three corpus question streams remain proxy evidence: Nymeria uses atomic
action narrations, while the LookOut and SLOPER4D prompts were generated from
target frames. Do not present the 14-session result as natural look-back QA.

Primary run for any one corpus root:

```bash
ROOT=/path/to/corpus/session_root \
PY=python \
CLIP_DEVICE=cuda \
BUDGETS=128 \
SEEDS=0,1,2,3,4 \
H3_RESOLUTIONS=12 \
METHODS=global_reservoir,fifo,hybrid,uniform_time,semantic_kcenter,spatial_priority,spatial_balanced,visit_balanced \
bash scripts/run_wearables_budget_suite.sh SESSION_ID [...]
```

Run one representative session first:

```bash
ROOT="$PSM_DATA_ROOT/video_retrieval/nymeria_atomic" \
BUDGETS=128 \
SEEDS=0,1 \
H3_RESOLUTIONS=12 \
METHODS=global_reservoir,fifo,hybrid,uniform_time,semantic_kcenter,spatial_priority,spatial_balanced,visit_balanced \
CLIP_DEVICE=cuda \
bash scripts/run_wearables_budget_suite.sh \
  20230608_s0_shelby_arroyo_act0_3ciwl8
```

Summarize captures at any time:

```bash
python scripts/summarize_fixed_budget.py \
  captures/wearables_fixed_budget \
  --recursive --strict \
  --out captures/wearables_fixed_budget/summary.md
```

The summary macro-averages session files within each seed. For the paper,
average seeds within each session first, then paired-bootstrap sessions. Do not
treat questions from the same session or five random seeds as independent
samples.

The paired script implements that order directly:

```bash
python scripts/compare_fixed_budget.py \
  captures/wearables_fixed_budget \
  --recursive \
  --method-a spatial_priority \
  --method-b global_reservoir \
  --budget 128 --h3-resolution 12 \
  --out captures/wearables_fixed_budget/paired_primary.md
```

## P0: Rare Places, Age, and Revisits

`eval_fixed_budget.py` attaches the following pre-registered fields to every
scored question:

- `rare_place`: bottom quartile of target-cell exposure within that session.
- `common_place`: top quartile of target-cell exposure.
- `old_event`: all GT intervals end in the first half of the session.
- `recent_event`: all GT intervals start in the final quarter.
- `revisited_place`: target cell has at least two visits.
- `heavily_revisited`: target cell has at least four visits.

A visit is split when consecutive observations in the cell are more than 30
seconds apart. The target cell is the modal H3 cell across GT-support frames;
if no sampled frame lands inside the interval, the nearest frame supplies the
cell and `gt_support_frames=0` records that limitation.

### Interpretation

| Outcome | Supported conclusion |
|---|---|
| Spatial-priority oracle coverage and Hit@5 both improve | The causal spatial policy protects low-exposure places under a fixed exemplar budget. |
| Oracle improves, Hit@5 does not | Allocation works, but CLIP ranking is the bottleneck. |
| Semantic k-center matches the spatial gain | Generic visual diversity, not physical place, may explain the result. |
| Spatial-priority beats semantic k-center and coordinate null | Evidence favors a place-specific allocation effect. |
| Visit-balanced beats spatial-balanced on revisits | Visit stratification is promising, but still needs a causal implementation. |
| No oracle improvement | Do not claim an allocation advantage; retain the systems-integration framing. |

## P0: Spatial Controls

Run these at the primary M=128 setting. Grid translation and rotation preserve
the local trajectory geometry while changing H3 boundaries. The coordinate
permutation null preserves the coordinate multiset but breaks its alignment
with the visual stream.

### Grid-boundary robustness

```bash
python scripts/eval_fixed_budget.py FEATURES QUESTIONS \
  --budgets 128 --seeds 0,1,2,3,4 --h3-resolutions 12 \
  --methods spatial_priority,spatial_balanced,visit_balanced \
  --translate-east-m 4.5 --out-dir captures/grid_controls/SESSION

python scripts/eval_fixed_budget.py FEATURES QUESTIONS \
  --budgets 128 --seeds 0,1,2,3,4 --h3-resolutions 12 \
  --methods spatial_priority,spatial_balanced,visit_balanced \
  --translate-north-m 4.5 --out-dir captures/grid_controls/SESSION

python scripts/eval_fixed_budget.py FEATURES QUESTIONS \
  --budgets 128 --seeds 0,1,2,3,4 --h3-resolutions 12 \
  --methods spatial_priority,spatial_balanced,visit_balanced \
  --rotation-deg 30 --out-dir captures/grid_controls/SESSION
```

Report the mean and range over the base, east, north, and rotation variants.
Do not select the best alignment.

### Broken-alignment null

```bash
for permutation_seed in 101 202 303; do
  python scripts/eval_fixed_budget.py FEATURES QUESTIONS \
    --budgets 128 --seeds 0,1,2,3,4 --h3-resolutions 12 \
    --methods spatial_priority,spatial_balanced,visit_balanced \
    --coord-permutation-seed "$permutation_seed" \
    --out-dir captures/coordinate_null/SESSION
done
```

Correct alignment should beat the null. If it does not, the observed effect is
consistent with generic partition balancing rather than physical place.
`--coord-shift-fraction` remains available as a weaker, occupancy-preserving
circular-shift control for comparison with earlier runs.

## P1: Human-Authored, Video-Verified Look-Back Questions

Nymeria atomic narrations are a useful temporal-grounding proxy, but they are
not questions a wearer asked. This is the highest-value annotation task left.

1. Select two revisit-rich Nymeria sessions.
2. Author 15 to 20 questions per session using
   `journal/genuine_lookback_qa/template_questions.yaml`.
3. Verify every interval against the video. Never derive intervals from a
   method's retrieved frames.
4. Have a second person verify at least a random 20 percent and report the
   agreement protocol.
5. Run the primary M=128 comparison and full PSM on exactly this frozen set.

The current command is documented in
`journal/genuine_lookback_qa/PROTOCOL.md`. Keep similarity-search questions in
the matched baseline set. `query_mode: last_seen` is a separate PSM-only task
and must not be scored as a zero for non-spatial baselines.

Run fixed-budget evaluation on the frozen genuine questions separately:

```bash
ROOT="$PSM_DATA_ROOT/video_retrieval/nymeria_atomic" \
QUESTIONS_NAME=genuine_questions.yaml \
OUT_ROOT=captures/genuine_fixed_budget \
BUDGETS=128 H3_RESOLUTIONS=12 CLIP_DEVICE=cuda \
bash scripts/run_wearables_budget_suite.sh SESSION_A SESSION_B
```

## P1: End-to-End PSM Check

The fixed-budget harness isolates allocation using identical global cosine
ranking. It does not replace the C engine evaluation. On the primary session,
also retain the existing comparisons:

- C PSM at R=128 and `per_cell_cap=K`.
- Brute-force CLIP.
- Sliding-window CLIP.
- The fixed-M global and spatial allocation policies.
- Galaxy S22 engine latency and RSS already measured in the paper.

Never use `eval_lookback.py` wall time as query latency: it starts and ingests a
fresh PSM process once per question. Use the in-process C benchmark for latency.

## Statistics and Reporting

1. Freeze methods, M, H3 resolution, visit gap, seeds, and question set.
2. Compute each metric per session and seed.
3. Average the five seeds within a session.
4. Report the mean across sessions with a paired 95 percent bootstrap CI on
   `spatial_priority - global_reservoir` using `compare_fixed_budget.py`.
5. Make M=128, r=12 the only primary comparison. Treat other budgets and
   resolutions as sensitivity analysis.
6. Report negative controls and failures, including the broken-coordinate null.
7. Report oracle coverage next to Hit@5 so retention and encoder effects cannot
   be conflated.

Do not promote a small pooled-question p-value. The independent unit is the
session, not the narration and not the reservoir seed.

## Nine-Day Execution Order

| Date | Deliverable |
|---|---|
| Jul 16 | Smoke-test scripts; freeze protocol and configs. |
| Jul 17 | Annotate first genuine-QA session; run one-session M sweep. |
| Jul 18 | Verify annotations; run the 14 street-scale sessions at M=128, r=12. |
| Jul 19 | Finish five seeds and budget sweep. |
| Jul 20 | Run grid transforms and coordinate-permutation null. |
| Jul 21 | Analyze rare/old/revisit strata and paired session statistics. |
| Jul 22 | Update claims, tables, limitations, and abstract. |
| Jul 23 | Independent paper and reproducibility review. |
| Jul 24 | Freeze PDF, source archive, commands, and captured JSON manifests. |
| Jul 25 | Submit with several hours of AoE buffer. |

## Claim Guardrails

- Say **fixed global exemplar budget**, not globally bounded total memory.
- Call `spatial_balanced` and `visit_balanced` **non-causal diagnostics**, not
  upper bounds or deployed methods.
- Call `spatial_priority` a causal Python retention policy until it is integrated
  into and benchmarked in the C engine.
- Say **proxy benchmark** for Nymeria narrations and target-frame-generated
  LookOut/SLOPER4D prompts.
- Say **human-authored, video-verified look-back QA**, not prospectively asked
  wearer queries.
- Report dataset versions, download instructions, and licenses rather than
  redistributing source videos.
- Keep the current title. Change it only if a new allocator is integrated,
  evaluated end to end, and becomes the actual central method.
