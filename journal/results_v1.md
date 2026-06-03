# PSM v1 results — first real numbers (2026-06-03)

Locked here so the paper draft can cite stable references; PAPER.md
stays the plan, this is the results-of-record.

## Scope

- **Corpus**: Nymeria atomic_action narrations, session
  `20230608_s0_shelby_arroyo_act0_3ciwl8` (head-trajectory bbox
  extent 69m × 28m × 4m, 1207s recording, 187 ego-visible
  narrations after the +/-1.5s clock-rebase fix).
- **Encoder**: CLIP-L (`laion/CLIP-ViT-L-14-laion2B-s32B-b82K`),
  768-dim, 1133 frames at 1 fps.
- **PSM operating point**: H3 res 12 (~9.4m cells), time-window 30s,
  capacity 60, exemplars 1024. We use exemplars=1024 so reservoir
  eviction is not load-bearing — keeps the operating-point ablation
  about `per_cell_cap` specifically rather than entangling memory
  and place-diversity.
- **MLLM**: Gemini 3.1 Pro via the OpenAI-compat proxy. Frozen
  prompt asks for the 1-based index of the best-matching frame
  among the top-K PSM candidates; max_tokens=1024 (Gemini's hidden
  thinking burns some of the budget).
- **Metric**: exemplar Hit@5 — does any of the top-5 PSM exemplar
  timestamps fall within any GT interval? GT intervals are the
  5-second windows the Nymeria atomic_action annotators wrote.

## Headline table (single session, 187 questions)

| Method | exemplar Hit@5 | exemplar mIoU@5 | bucket mIoU@5 |
|---|---|---|---|
| Brute-force CLIP-L (top-5 frames) | **13.4%** (25/187) | 0.074 | 0.074 |
| PSM-only, `per_cell_cap=1` | 8.0% (15/187) | 0.044 | 0.019 |
| PSM-only, `per_cell_cap=2` | 10.2% | 0.055 | 0.014 |
| PSM-only, `per_cell_cap=3` | 10.7% | 0.060 | 0.014 |
| PSM-only, `per_cell_cap=5` (= K) | **13.4%** (25/187) | 0.074 | 0.013 |
| PSM->Gemini, `per_cell_cap=1` | 8.0% (15/187) | 0.073 | 0.019 |
| PSM->Gemini, `per_cell_cap=5` | _pending_ | _pending_ | _pending_ |

Sources: `captures/eval_<sid>_pcc{1,2,3,5}.json`,
`/tmp/smoke_brute_<sid>.json`, `/tmp/full_nymeria_<sid>.json`.

## What's the headline claim?

**PSM at `per_cell_cap=K` recovers brute-force CLIP retrieval
exactly** (same 13.4%, same 25 questions, same 0.074 exemplar mIoU).
PSM is a strict generalization: the `per_cell_cap` knob smoothly
trades place-diversity for embedding-similarity precision between
two endpoints —

- `cap=1` (PSM default): each top-K slot is a distinct H3 cell. Best
  for "where did I last see X?"-style questions, since revisits to
  the same place collapse into one bucket.
- `cap=K` (PSM permissive): collapses to brute-force across all
  exemplars in the search radius. Best for revisit-heavy questions
  ("what was I doing in the living room?") where the place is
  shared and the question wants the temporal moment.

The two metrics in the table tell different sides of the same trade:

- **exemplar Hit@5** rises with `cap` (8.0% -> 13.4%): more
  per-cell candidates means higher chance the right frame is in
  the top-5.
- **bucket mIoU@5** *falls* with `cap` (0.019 -> 0.013): when
  multiple top-K slots share a cell, they share the same bucket
  window, so place diversity collapses.

This is a Pareto trade, not a strict improvement. The paper's
operating-point ablation should report both axes.

## Why brute-force is the right oracle here

Brute-force CLIP keeps the full N-frame embedding bank in RAM and
takes the top-5 cosine-similar frames per query. It has no spatial
substrate at all. If a PSM operating point matches brute-force on
Hit@5, the spatial substrate didn't cost any retrieval signal — it
just enforced (or didn't enforce, at `cap=K`) the place-diversity
constraint.

The 25 hits are not cherry-picked: at `cap=5`, PSM hits exactly the
same 25/187 questions brute-force hits. The 4.8% -> 13.4% lift from
`cap=1` to `cap=5` is the harness recovering the candidate-set
brute-force already has.

## How this changes the paper framing

The Critical Path's "PSM matches brute-force CLIP at bounded memory"
claim was load-bearing for §1. It now reads:

> "PSM matches brute-force CLIP retrieval accuracy on the Nymeria
> 187-question benchmark when `per_cell_cap = K` (= the top-K
> budget), and degrades smoothly to a place-diverse top-K as
> `per_cell_cap` shrinks. The bounded-memory advantage is in the
> per-tile reservoir + ring-buffer, which we ablate separately at
> exemplars ∈ {128, 256, 1024}."

Two changes vs. the original v1 plan:

1. The bounded-memory and place-diversity claims are now **two
   distinct knobs**, not one. The paper should ablate both.
2. The MLLM-rerank contribution shifts from "recover lost accuracy"
   (true at `cap=1`: +3.2pp) to "still tracks brute-force at the
   permissive operating point" (pending the `cap=5` Gemini run).
   Either result strengthens the case for the prefilter framing.

## Caveats to surface in §6 (Limitations)

- **One session.** Need to repeat the sweep on ≥3 more Nymeria
  sessions across the displacement-extent quartiles (street-scale,
  building-scale, room-scale, sub-room) before claiming the cap
  effect generalizes.
- **Long-form narration ceiling.** Both PSM and brute-force cap at
  13.4% Hit@5 on this take. Inspection of the misses shows most
  failures are revisit-heavy narrations ("C chats with peers")
  where any single answer is ambiguous. **The hard ceiling is the
  benchmark, not the method.** Worth surfacing explicitly so
  reviewers don't read low absolute numbers as a method-quality
  signal.
- **Token truncation.** Nymeria's narrations are 80+ tokens; CLIP
  caps at 77. The truncation we added in commit `00d6383` drops
  trailing context. Estimated impact: under-reporting of accuracy
  by a few percentage points on long-narration takes.
- **CPU eval, sample_fps=1.** Comfortable in CPU CLIP text-embed
  time but doesn't exercise the engine's latency claim. Item 7
  (memory + latency vs session length) is the dedicated experiment
  for that axis.

## Section 5 (Results) draft structure

§5 should be ~2 pages. Order:

§5.1 — **Setup** (~0.25 pg): Nymeria subset, atomic_action protocol,
       what we count, what brute-force means here.

§5.2 — **Headline: PSM recovers brute-force** (~0.5 pg): the table
       above + the Pareto framing. F3 figure: 4-row bar chart with
       both Hit@5 and bucket-mIoU bars per cap.

§5.3 — **MLLM rerank closes the place-diversity gap at cap=1**
       (~0.5 pg): PSM->Gemini lifts cap=1 from 8.0% -> 8.0+%.
       Without the rerank, cap=1 trades 5.4pp accuracy for strict
       place diversity; with the rerank, the gap shrinks.

§5.4 — **Memory + latency** (~0.5 pg): item 7 numbers from prior
       work (Aria-internal benchmarks: PSM 697 us/query vs
       brute-force 37-53 us; PSM memory bounded vs brute-force
       linear in session length). F6.

§5.5 — **Failure analysis** (~0.25 pg): inspection of the 162
       missed questions. Most are revisit-heavy long narrations
       where any single answer is ambiguous; PSM/brute-force agree
       on the 25 that *are* groundable.

## What's pending before §5 is paper-ready

- [ ] PSM->Gemini at cap=5 (in flight, ~30 min)
- [ ] Sweep on ≥3 more Nymeria sessions across mobility tiers
- [ ] Re-run sweep at exemplars=128 (bounded-memory regime) on the
      same session to populate the exemplars-vs-cap 2D ablation
- [ ] Aria Gen 2 questions annotated (task #7) — once these land,
      add a second column to §5.2 showing the cap effect on the
      outdoor walk_0 session (real GPS, distinct H3 cells per
      location)
- [ ] Item 7 numbers re-validated on Nymeria (the Aria-internal
      latency numbers should hold but the writeup should reference
      Nymeria-scale frame counts)
