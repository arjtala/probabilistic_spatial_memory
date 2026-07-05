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
≥2-distinct-day cells at r10 = 47.0% ≥ 30% bar → GO.**

| res | cells | %cells ≥2-drive | %cells ≥2-day | cov ≥2-day |
|---|---|---|---|---|
| r9  | 8,984 | 36.2% | 21.8% | 48.7% |
| **r10** | **23,668** | **31.0%** | **21.8%** | **47.0%** |
| r11 | 58,628 | 25.8% | 20.2% | 44.4% |
| r12 | 144,200 | 22.6% | 19.4% | 41.2% |

**Persistence (the reframe's load-bearing claim):** of 5,163 ≥2-day r10 cells,
the gap between first and last visit is **median 105.9 d, p90 174.3 d, max
221.2 d** — half the revisited ground is re-driven >3.5 months apart. This is a
genuine multi-session-persistence signal, not same-week repetition.

**Honesty notes:**
- Two preprocessing bugs in the first run biased the number *down* to a
  contaminated 22.6%: (a) a too-tight lat floor (37.0) dropped 4 real Santa
  Cruz drives; (b) outlier GPS fixes created ~42k spurious singleton r10 cells.
  Fixing both (widen box to include Santa Cruz + per-drive 80 km outlier
  reject) gave the 47.0% above. Both fixes only *raise* the metric, so 47% is
  itself conservative w.r.t. the adjacent-cell undercount noted in the rule.
- The most-revisited cells cluster near the Honda facility (Mountain View), so
  some revisit is depot-driven (drives originate there). But the top-50 cells
  span ~7×6 km along the El Camino corridor (lat 37.387–37.448), and 47%
  coverage across 23,668 cells cannot come from the depot alone — arterial
  revisits carry it. Depot cells could be excluded as a sensitivity check;
  the margin over the bar makes it unnecessary.
- `vel.csv` speed units are unverified (max ~47), but the stopped-vs-moving
  gate at >1.0 is unit-robust (0.0 = idling).

**Next: B-vs-C is now the open call.** Revisits + months-long persistence make
Option B (systems: bounded per-cell state vs. linear dense bank; cross-drive
HLL cardinality accrual + decay over the 5,163 revisited cells) solid and
annotation-free. Option C (retrieval eval) needs a hand-annotated look-back QA
set and only pays off if a *driving* retrieval result strengthens the wearable
narrative rather than reading as bolted-on.

## First cluster-side steps (once access + option chosen)
1. Confirm HDD RGB frames embed cleanly (CLIP-L / SigLIP 2) — sanity batch.
2. ✅ **Done** — Ingest GPS → H3; histogram cells by distinct-drive **and
   distinct-day** count + inter-visit temporal-gap distribution (catch #1;
   result above). → `scripts/hdd_revisit_density.py`,
   `captures/hdd/revisit_density.json`.
3. If revisits exist: cross-drive HLL cardinality accrual + time-decay plot
   (seeded by the top-N revisited-cell → drive-list mapping from step 2's JSON).
4. Memory-vs-area curve: PSM bounded per-cell state vs. dense-bank linear growth.
