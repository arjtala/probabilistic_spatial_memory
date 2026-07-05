# Honda HDD — candidate dataset for the OpenSUN3D / SpatialMem submission

**Status:** scoping brief. Decision pending — see [Open decisions](#open-decisions).
**Why this file exists:** data lives on the H200 cluster; this brief travels
with the repo so the evaluation can be set up cluster-side.

> Note on placement: repo policy keeps longer writeups in `journal/`. This lives
> in `docs/` by explicit request so it syncs to the H200 cluster alongside the data.

---

## Why HDD is on the table

The abstract/intro were reframed for the SpatialMem track (branch
`opensun3d-reframe`) to lead on **dynamic spatial representation**,
**test-time memory updating**, and **memory-budgeted systems** — state
bounded by *area explored*, not sequence length. That reframe makes two
claims the current corpus (14 single-pass walks: Nymeria, SLOPER4D,
LookOut/Aria) does **not** yet demonstrate:

1. **multi-session persistent memory** (`section_1_intro.tex:15`)
2. **unique-event cardinality across revisits** (`section_1_intro.tex:18`)

A dataset with genuine same-place revisits and wide geospatial coverage
closes that gap. Honda HDD is the leading candidate.

### Is OpenSUN3D wearable-only? No.
Unlike the Wearable AI CFP, OpenSUN3D / SpatialMem is about spatial
intelligence and world models — "how machines reason about and engage with
their environments." An instrumented vehicle is an embodied agent; driving
is on-topic. Driving *strengthens* the "state scales by area, not time"
thesis because a drive sweeps far more H3 cells per minute than a walk.

---

## What HDD gives us

| Property | HDD | Notes |
|---|---|---|
| Scale | 104 h real driving, SF Bay Area | Dramatic area coverage → starkest bounded-vs-linear memory contrast |
| GPS | ✅ real lat/lon traces | Plugs straight into existing GPS→H3 path (no SLAM-frame projection, unlike AEA) |
| RGB video | ✅ front-facing camera | CLIP/SigLIP-embeddable; extraction pipeline is encoder-agnostic |
| Other sensors | CAN bus, GPS | Not needed for PSM |
| Embodiment | vehicle (egocentric-adjacent) | On-topic for SpatialMem, off-topic for Wearable AI |

## What HDD does *not* give us (the catches)

1. **Revisit structure is undocumented.** HDD was built for driver
   behavior/intention modeling, not place revisitation. 104 h in one metro
   almost certainly re-drives arterials, but the dataset doesn't advertise
   it — we must **measure and demonstrate** the revisit structure ourselves
   (novel finding if it holds; risk if overlap is thin).
2. **No look-back QA labels.** Our Hit@5 / mIoU harness depends on annotated
   query→interval pairs. HDD has none. Either hand-annotate a small query
   set, or run HDD as a **pure systems** result (memory-vs-area curves,
   cross-drive HLL cardinality) with no retrieval-accuracy metric.
3. **Access gate.** Non-commercial; requires a university email request via
   the HDD download form. Confirm eligibility before planning.
4. **Indoor/outdoor & CLIP suitability** confirmed outdoor RGB; CLIP
   suitability is expected (plain RGB frames) but unverified until we embed.

---

## Options for what HDD should carry

Pick one; it sets the effort and whether QA annotation is needed.

### Option A — Verify overlap first (recommended sequencing)
Request access, ingest GPS, H3-bin the traces, and **measure how many cells
are driven by ≥2 distinct drive IDs**. Decide A→B or A→C once revisit
density is known. De-risks catch #1 before any annotation spend.

### Option B — Systems demo only
Ingest HDD GPS+video; show bounded per-cell state vs. linearly-growing dense
bank across a large area, plus cross-drive HLL cardinality accrual + decay
where revisits exist. **No QA labels needed.** Low–medium effort, on-topic,
but yields **no accuracy metric** on HDD.

### Option C — Full retrieval eval
Hand-annotate a look-back QA query set so Hit@5/mIoU carry over from Nymeria.
Strongest, most comparable result. Significant annotation effort; requires
confirmed revisits (do Option A first regardless).

---

## Open decisions

- [x] **Which option (A / B / C)?** → **Option A** (verify overlap first).
      B-vs-C decided after the revisit measurement (see decision rule below).
- [x] **HDD access gate?** → Moot. Data is already on the cluster at
      `/checkpoint/dream/arjangt/video_retrieval/hdd/release_2019_07_08/`
      (21 drive-days, 132 drives with RTK GPS + center camera).
- [ ] **Archival (8-page) or non-archival (4-page)?** A new HDD result is
      heavier; a 4-page non-archival abstract may not have room and keeps
      Wearable AI open as a fallback (dual-submission).
- [x] **Does HDD replace or supplement the existing corpus?** → **Supplement**
      the frozen 14-session street-scale corpus (adds the multi-session /
      revisit-cardinality dimension the single-pass corpus lacks).
- [ ] **Fallback if revisit density is thin:** keep HDD as a pure
      area-vs-memory systems figure, or soften the intro's multi-session
      claims to match the existing single-pass corpus.

## Revisit-density decision rule (pre-registered 2026-07-05, before running)

Recorded *before* the measurement to avoid a post-hoc criterion.

- **Primary metric:** coverage-weighted fraction of driving over
  **≥ 2-distinct-*day*** H3 cells at **resolution r10** (~66 m edge).
  - *Day-count, not drive-count* — same-day drives may be one continuous trip;
    cross-day revisits are the actual "multi-session persistent memory" signal.
  - *r10, not finer* — RTK meter-level error cannot leak across a 66 m cell
    boundary, so r10 revisits are noise-robust. r11/r12 reported but flagged
    noise-sensitive (no fix-quality column ships with HDD).
  - *Coverage-weighted, not raw cell count* — a revisited arterial dominates
    driven time; raw "N revisited cells among thousands" understates the story.
    Weighting is by distinct drives/days, never by raw GPS point count
    (stationary idling would inflate it — points are speed-gated via `vel.csv`).
- **Go/no-go bar: ≥ 30%.** At or above → a meaningful third of driving is over
  cross-session re-seen ground; proceed to Option B/C. Below → fall back
  (area-vs-memory systems figure only, or soften §1 multi-session claims).
- **Conservative bias:** lane changes + GPS noise map true revisits to
  *adjacent* cells, so the strict count **under**-reports. Clearing the bar
  despite this undercount is therefore robust. A k-ring-1 "soft revisit"
  sensitivity variant is held in reserve for a borderline result only.
- **B-vs-C is a narrative call, not just a number:** HDD is driving; the paper's
  headline task is wearable look-back QA. A healthy revisit number makes the
  systems claims (Option B, no annotation) sound; Option C (retrieval eval)
  additionally needs annotation and is only worth the spend if a driving
  retrieval result *strengthens* rather than *fragments* the wearable narrative.

## Revisit-density result (measured 2026-07-05) — **GO** ✅

`scripts/hdd_revisit_density.py` over 130/132 drives (2 dropped: unreadable /
out-of-region GPS). Full stats in `captures/hdd/revisit_density.json`;
r10 cell map + histogram in `journal/figures/hdd_revisit_{density,hist}_r10.svg`.

**Decision metric (pre-registered): coverage-weighted fraction of driving over
≥2-distinct-day cells at r10 = 47.0% ≥ 30% bar → GO.** Precisely: 47.0% of
*(drive, cell) traversal events* land in ≥2-distinct-day cells (this is an
event fraction, not a fraction of driven time or distance).

| res | cells | %cells ≥2-drive | %cells ≥2-day | cov ≥2-day (events) |
|---|---|---|---|---|
| r9  | 8,984 | 36.2% | 21.8% | 48.7% |
| **r10** | **23,668** | **31.0%** | **21.8%** | **47.0%** |
| r11 | 58,628 | 25.8% | 20.2% | 44.4% |
| r12 | 144,200 | 22.6% | 19.4% | 41.2% |

**Persistence (the reframe's load-bearing claim):** of 5,163 ≥2-day r10 cells,
the gap between first and last visit is **median 105.9 d, p90 174.3 d, max
221.2 d** — half the revisited ground is re-driven >3.5 months apart. This is a
genuine multi-session-persistence signal, not same-week repetition.

**Depot-exclusion sensitivity (defuses "your revisits are just the Honda
garage"):** recomputing coverage with the top-K most-driven (depot-origin)
cells removed from both numerator and denominator barely moves it —
**47.0% → 46.9% (K=5) → 46.7% (K=10) → 46.5% (K=20)**. Cross-day re-driving is
distributed across thousands of arterial cells, not concentrated at the
Mountain View origin. The claim is measured, not argued.

**Honesty notes:**
- Two preprocessing bugs in the first run biased the number *down* to a
  contaminated 22.6%: (a) a too-tight lat floor (37.0) dropped 4 real Santa
  Cruz drives; (b) outlier GPS fixes created ~42k spurious singleton r10 cells.
  Fixing both (widen box to include Santa Cruz + per-drive 80 km outlier
  reject) gave the 47.0% above. Both fixes only *raise* the metric, so 47% is
  itself conservative w.r.t. the adjacent-cell undercount noted in the rule.
- The most-revisited cells cluster near the Honda facility (Mountain View), so
  some revisit is depot-driven (drives originate there). But the depot-exclusion
  sensitivity above (47.0% → 46.5% removing the top-20 cells) shows the coverage
  is carried by distributed arterial revisits, not the origin. The top-50 cells
  already span ~7×6 km along the El Camino corridor (lat 37.387–37.448).
- `vel.csv` speed units are unverified (max ~47), but the stopped-vs-moving
  gate at >1.0 is unit-robust (0.0 = idling).

**Next: B-vs-C is now the open call.** Revisits + months-long persistence make
Option B (systems: bounded per-cell state vs. linear dense bank; cross-drive
HLL cardinality accrual + decay over the 5,163 revisited cells) solid and
annotation-free. Option C (retrieval eval) needs a hand-annotated look-back QA
set and only pays off if a *driving* retrieval result strengthens the wearable
narrative rather than reading as bolted-on.

## Decision: Option 3 — B now, C gated behind reviewer demand (2026-07-05)

HDD carries the **systems/persistence** thesis; the frozen 14-session corpus
carries **retrieval fidelity**. Two corpora, two complementary properties
(stated as division of labor so a missing HDD Hit@5 reads as scope, not
omission). Run Option C only if (a) a reviewer explicitly demands an HDD
accuracy number, or (b) the drafted B figures read thin without one.

### B figure plan + status

- **F-HDD-1 memory-vs-area — ✅ built** (`scripts/hdd_memory_vs_area.py`,
  `captures/hdd/memory_vs_area.json`, `journal/figures/hdd_memory_vs_area.svg`).
  Honest model: PSM state = Σ_cells min(frames_in_cell, R=128) exemplars;
  dense bank = all frames. On 102.9 h / 23,736 r10 cells, PSM is smaller at
  every ingest rate, and the gap grows with fps as the per-cell reservoir caps
  redundant frames:

  | sample fps | PSM | dense bank | bank/PSM | PSM frame saving |
  |---|---|---|---|---|
  | 1  | 0.89 GB | 1.14 GB | 1.3× | 26.3% |
  | 5  | 2.51 GB | 5.69 GB | 2.3× | 56.7% |
  | 15 | 3.95 GB | 17.07 GB | 4.3× | 77.1% |
  | 30 | 4.79 GB | 34.14 GB | **7.1×** | **86.1%** |

  **Honesty caveat (caught by the pre-GPU curve check) + agreed §1/§5 framing.**
  The win comes from per-cell reservoir *capping* (dwell + revisits discard
  frames beyond R), NOT from a global area plateau — area keeps growing on this
  corpus (only 20% of cells seen by the halfway mark; the fleet explores new
  routes over 8 months, so the 47% revisit fraction doesn't dominate exploration
  enough to flatten the area curve). An earlier model (n_cells × R, assuming
  every reservoir full) wrongly inflated PSM 10× above the bank at 1 fps; the
  min() is the real reservoir. So pitch it precisely:
  - **F-HDD-1 claim:** "PSM state grows *sublinearly in observation count* —
    it is O(distinct cells visited), not O(frames ingested). At 15 fps it is
    4.3× smaller than a dense bank; at 30 fps, 7.1×." The area bound is an
    **asymptotic architectural property** (state = Σ cells × capped reservoir),
    stated as such — NOT an observed plateau, which HDD does not show.
  - **§1 edit:** soften "state bounded by area, not time" → "state is
    O(distinct cells visited), not O(frames ingested)"; drop any plateau/
    saturation wording.
  - **Report all three fps rows, including 1 fps (1.3×).** 1 fps is the
    wearable corpus's rate and the margin is thin there; frame it as "the
    advantage grows with ingest rate, and always-on wearable/vehicle streaming
    is 15–30 fps," turning the weak row into a motivated design point.
  - **Division of labour:** F-HDD-1 carries "state is sublinear / area-bounded
    across the whole corpus"; the *persistence* claim (memory accrues + decays
    across sessions on revisited cells) is **F-HDD-2's** job, not F-HDD-1's.
- **F-HDD-2 cross-drive HLL cardinality accrual + decay — ✅ built, ⏳ runs
  post-extraction** (`scripts/hdd_hll_cardinality.py`). This is where revisits
  ARE the whole story: a fixed 1 KiB/cell HLL sketch whose cardinality accrues
  across drives (persistent) and decays via the ring buffer over the months-long
  gaps (bounded). Python HLL **validated to match `targets/psm` exactly** (0.0%
  median relerr on the sanity drive); windowed decay is time-grid-sampled so it
  actually drops to ~0 mid-gap and jumps on revisit (verified 5→0→1 across a
  100-day synthetic gap). Seeded by `top_cells_r10` in the revisit JSON;
  recomputes per-cell days/gap from *loaded* drives and warns on partial
  feature coverage.
- **F-HDD-3 self-supervised cross-session retrieval — ✅ built, ⏳ runs
  post-extraction** (`scripts/hdd_cross_session_retrieval.py`). Query a
  revisited cell with a drive-A exemplar; AUC that different-day drive-B frames
  in that cell outrank other-cell frames (+ hit-rate@k over the full pool incl.
  same-drive near-duplicate distractors). No hand-labels. Controls: same-drive
  AUC (upper bound), shuffled-cell AUC (~0.5 floor). Tie-corrected AUC + AUC
  math unit-tested. **Degenerate contingency:** if the embed-sanity verdict were
  DEGENERATE, F-HDD-3 reports AUC *with a low-separability caveat* or pivots to a
  GPS-only consistency check. (Moot — the CLIP-L gate PASSED, cos 0.41–0.94.)
- **Verification:** a 15-agent adversarial review→verify workflow over F-HDD-2/3
  raised 13 findings, confirmed 6 (2 major F-HDD-2, 1 major + 3 minor F-HDD-3),
  all fixed (commit `3292571`). Both scripts pass `pyrefly` (0 errors) and were
  smoke-tested on the real 50-frame sanity H5.
- **Second independent audit (2026-07-05):** a fresh 4-lens pass (statistics /
  numerics / data-contract / reproducibility) over all HDD code raised 17,
  **confirmed 0** — the code held; verifiers caught the raisers' own errors (a
  non-existent float32→float64 round-trip; a "decay window < median gap" that
  misread first→last *span* as inter-visit *gap*; a proposed "defensive
  normalize" that would have broken the validated engine match). Banked non-bug
  improvements applied: run-config (seed, max-queries, top-cells) echoed into
  the JSON outputs for provenance; F-HDD-3 documents that its AUC blends
  coarse-geography with visual place identity (→ add a k-NN-cell hard-negative
  control if it's ever written up) and is a per-query (not per-cell) mean.

### Embedding pipeline (prereq for F-HDD-2/3) — ✅ built, embed-sanity ✅ PASSED

**Embed-sanity gate PASSED (CLIP-L, drive 201702271017, SLURM job 9270044 on
h200-137, 2026-07-05):** 50 windshield frames, `cos_mean=0.776, std=0.085,
p05=0.60, p95=0.88, min=0.41, max=0.94` → **OK (usable spread, not degenerate)**.
`track_mode=real_gps`, sidecar consumed as `json_sidecar:hdd-rtk`. Also confirmed
the Bug-2 fix live: `sample_fps=50/3180.2` used the *video* duration, not the GPS
span. F-HDD-3's encoder-collapse risk is cleared → full array + F-HDD-2/3 greenlit.

- `extraction/psm_extraction/io/hdd.py` — reads real RTK GPS (lat/lng swap +
  SF-Bay guard), no fake-origin projection. Verified: discovers 132 drives.
  Carries `first_iso` so the sidecar can correct the video↔GPS clock skew.
- `scripts/extract_hdd_sessions.py` — writes an Aria-style `gps.json` sidecar
  and drives the standard `python -m psm_extraction extract` pipeline (mirrors
  `extract_sloper4d_sessions.py`). Sidecar timestamps are placed on the **video
  clock with the GPS-warmup skew corrected** (measured +3.44 s on drive
  201703061033: video starts 10:33:53, first RTK fix 10:33:56) so per-frame
  lat/lng don't lead the true position by tens of metres — which would bias r10
  binning for F-HDD-2/3. Includes a `--sanity-only` **embed-sanity gate**.
- `scripts/slurm/hdd_embed_sanity.sbatch` — **single-task GPU job for the gate.**
  Run this FIRST: `sbatch scripts/slurm/hdd_embed_sanity.sbatch` (or
  `--export=ALL,MODEL=siglip2_l` / `,DRIVE=<id>`). Extracts ~50 frames from one
  drive, embeds, prints the pairwise-cosine spread + OK/DEGENERATE/FAIL verdict,
  and writes `captures/hdd/embed_sanity_<model>.json`. Non-zero exit on
  FAIL/DEGENERATE so it can gate the array via a dependency.
- `scripts/slurm/extract_hdd.sbatch` — full 132-drive array, MODEL=clip_l|
  clip_bigg|siglip2_l. Launch only after the sanity verdict is OK. Needs a GPU
  node (no torch/GPU/open_clip on the login node used for the GPS-only analyses).

**Hardening (static verification, 2026-07-05).** Since the embed path can't be
runtime-tested off-GPU, a 3-agent audit (CLI-contract / sbatch-conventions /
adversarial code review) checked it before the sbatch shipped. It caught and
fixed 3 GPU-only failures that would each have wasted a job: the sanity group
selector could pick the always-present `gps` group and `KeyError`; the ~50-frame
fps was computed from the GPS span (8–11 s off the video) instead of the video
duration; and a ≤1-frame extraction would have returned a false "OK". Plus the
clock-skew fix above. All files pass `pyrefly` (0 errors).

## First cluster-side steps (once access + option chosen)
1. ⏳ Confirm HDD RGB frames embed cleanly (CLIP-L / SigLIP 2) — `--sanity-only`
   gate in `extract_hdd_sessions.py`, run on a GPU node before the full array.
2. ✅ **Done** — Ingest GPS → H3; histogram cells by distinct-drive **and
   distinct-day** count + inter-visit temporal-gap distribution (catch #1;
   result above). → `scripts/hdd_revisit_density.py`,
   `captures/hdd/revisit_density.json`.
3. F-HDD-2: cross-drive HLL cardinality accrual + time-decay plot (seeded by
   the top-N revisited-cell → drive-list mapping from step 2's JSON).
4. ✅ **Done (model)** — F-HDD-1 memory-vs-area: PSM bounded per-cell state vs.
   dense-bank linear growth (`scripts/hdd_memory_vs_area.py`).
