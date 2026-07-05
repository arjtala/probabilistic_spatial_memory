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

- [ ] **Which option (A / B / C)?** — determines all downstream work.
- [ ] **Do we have a university affiliation / email to clear the HDD access gate?**
- [ ] **Archival (8-page) or non-archival (4-page)?** A new HDD result is
      heavier; a 4-page non-archival abstract may not have room and keeps
      Wearable AI open as a fallback (dual-submission).
- [ ] **Does HDD replace or supplement the AEA idea?** AEA has revisits but is
      indoor / no GPS / low geospatial diversity — user ruled it weak on the
      "spatial" axis. HDD is the stronger spatial fit if revisits check out.
- [ ] **Fallback if revisit density is thin:** keep HDD as a pure
      area-vs-memory systems figure, or soften the intro's multi-session
      claims to match the existing single-pass corpus.

## First cluster-side steps (once access + option chosen)
1. Confirm HDD RGB frames embed cleanly (CLIP-L / SigLIP 2) — sanity batch.
2. Ingest GPS → H3; histogram cells by distinct-drive count (answers catch #1).
3. If revisits exist: cross-drive HLL cardinality accrual + time-decay plot.
4. Memory-vs-area curve: PSM bounded per-cell state vs. dense-bank linear growth.
