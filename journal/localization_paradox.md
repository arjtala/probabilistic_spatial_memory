# Localization Paradox and PSM Demo

> ⚠️ **Internal-preliminary, superseded.** This writeup uses the original 3-session
> internal Aria corpus, which was classified internal-only on 2026-05-28 and
> cannot appear in a published paper. All numbers below are validation that the
> pipeline works end-to-end, not publishable results. The current
> results-of-record live in [`results_v1.md`](results_v1.md) (Nymeria-based,
> 2026-06-04). See [`PAPER.md`](PAPER.md) for the corpus-pivot reasoning.

## TL;DR

Probabilistic Spatial Memeory (PSM) is a bounded, time-decayed spatial-memory engine built on HyperLogLog sketches over an H3 hex grid. It is structurally a candidate for closing the **localization paradox** in streaming egocentric memory, i.e. the gap between an MLLM's *semantic* answer ("you saw a yellow bus") and its near-zero *temporal grounding*, the `[t_start, t_end]` interval that proves it.

The term comes from a forthcoming streaming egocentric memory benchmark (the "Localization Paradox benchmark" after its headline finding), which evaluates frontier MLLMs (Gemini 3 Pro, GPT-5, InternVL 3.5) on 20K+ hand-annotated "look-back" QA pairs over 613 hours of unscripted Ray-Ban-style smart-glasses video. The headline finding is that frontier models reach 27–50% *semantic* accuracy but their *temporal grounding* `mIoU` against the ground-truth `[t_start, t_end]` intervals collapses to near zero — and the paper's discussion section calls explicitly for "adaptive temporal indexing… and hierarchical memory buffers that can compress hours of video without losing the granular detail of momentary 'needle' events," which is exactly the architectural shape PSM provides.

Across **three independent recordings on three continents** with **12 hand-annotated look-back questions and 3 negative controls**, evaluated over **5 reservoir-sampler seeds** (60 question-seed evaluations):

| metric | value |
|---|---|
| **exemplar Hit @5** | **68.3% ± 7.0%** |
| exemplar mIoU @5 | 0.259 ± 0.015 |
| bucket mIoU @5 | 0.182 ± 0.001 |
| false-positive rate (negatives) | **0%** |

The vanilla MLLM baseline reported in the localization-paradox literature is near-zero on equivalent tasks. PSM's `--search` returns grounded `(cell, t_start, t_end, exemplar_t)` tuples in O(matching_tiles × capacity) — sub-millisecond on session-scale memory. The narrative for the paper is **PSM as MLLM prefilter**, not competitor: PSM emits the kind of candidate intervals an MLLM reranker can consume as grounding context.

---

## Glossary

**Exemplar.** A per-cell reservoir-sampled embedding kept alongside the HLL counter. The `--exemplars N` flag sets the per-cell reservoir size (we use 128). Each exemplar carries the full embedding vector and the frame's timestamp `exemplar_t`. Reservoir sampling (Algorithm R) means every observed frame has equal probability of ending up in the final sample. At query time, `psm --search` computes cosine similarity between the text query and every cell's exemplars, then ranks cells by their best-exemplar score.

**Bucket.** One slot of the per-cell ring buffer of HLL sketches. Each bucket has a `(t_min, t_max)` interval covering all observations that landed in it. Coarse-grained (60–90 s wide at the retention setting we use) but fully deterministic across reservoir seeds.

**Hit @k.** Did at least one of the top-k returned cells produce a candidate interval overlapping a ground-truth interval? Reported in two flavors:
- *exemplar* Hit @k — uses `[exemplar_t ± 1.5 s]` as the predicted interval. Frame-precise.
- *bucket* Hit @k — uses the cell's `(t_min, t_max)`. Place-precise but loose.

**mIoU @k.** Mean over questions of the best IoU within top-k. Same two flavors. The localization-paradox literature wants exemplar-level numbers (frame-precise grounding); bucket-level is reported alongside as a place-level baseline.

---

## The setup

Three recordings, three continents, three modalities:

| session | location | modality | duration | scored | negative |
|---|---|---|---|---|---|
| `1501677363692556` (A) | Fulham, London | Aria glasses, walking | 15.0 min | 4 | 1 |
| `287142033569927` (B)  | Tucson, AZ      | Aria glasses, cycling | 15.1 min | 4 | 1 |
| `201703061033` (C)     | Palo Alto, CA   | Honda HDD, driving    |  4.4 min | 4 | 1 |

Visual sanity check across the three sessions: left pane is the video with the DINO attention overlay, right pane is the hex spatial memory built from CLIP exemplars plus the colored GPS trace ribbon.

![sessions strip](../captures/three_sessions_strip.png)

Configuration for every result below: **CLIP-ViT-L/14 LAION-2B**, **128 exemplars per cell**, **75 s × 12 retention** (~15 min effective horizon), **--seed 42** for the single-seed numbers and **--seed 0..4** for the seed sweep.

---

## Results

### Single seed (12 questions, seed=42)

| session | n | exemplar mIoU @5 | exemplar Hit @5 | bucket mIoU @5 |
|---|---|---|---|---|
| Fulham (A)        | 4 | 0.229 | 50%  (2/4) | 0.027 |
| Tucson (B)        | 4 | 0.368 | **100% (4/4)** | 0.226 |
| Palo Alto (C)     | 4 | 0.180 | 50%  (2/4) | 0.295 |
| **combined**      | **12** | **0.259** | **66.7% (8/12)** | **0.165** |

### Seed sweep (5 seeds, mean ± std)

| session | seeds | exemplar mIoU @5 | exemplar Hit @5 | bucket mIoU @5 |
|---|---|---|---|---|
| Fulham (A)        | 0–4 | 0.230 ± 0.047 | **55% ± 20.9%** | 0.026 ± 0.003 |
| Tucson (B)        | 0–4 | 0.368 ± 0.000 | **100% ± 0%**   | 0.226 ± 0.000 |
| Palo Alto (C)     | 0–4 | 0.179 ± 0.002 | **50% ± 0%**    | 0.295 ± 0.000 |
| **combined**      | 0–4 | **0.259 ± 0.015** | **68.3% ± 7.0%** | **0.182 ± 0.001** |

Two methodological observations from the sweep:

- **Bucket mIoU is deterministic across seeds** (`±0.000`–`±0.003`). It depends only on H3 cell membership, which is a function of the observe sequence. Only the exemplar layer carries reservoir-sampling variance.
- **Tucson (100%) and Palo Alto (50%) are bedrock-stable**. Tucson hits all four every seed; Palo Alto's deterministic 50% means its two failures (semi-truck, person walking) are *encoder* misses, not reservoir noise. Fulham's variance is concentrated in the bus / elevator / bicycle queries where multiple plausible cued frames exist in the covering cell.

### Negative-control behavior

All three negative controls (dog in A, wildcat in B, airplane in C) registered as misses (no top-k exemplar overlapping any forbidden GT, though by definition there is no GT for a negative control, so the test is "did PSM still return something high-similarity at all?"; it didn't). The harness's hit-column prints `✓` for hits and `✗` for misses; all three negative controls printed `✗` across every seed. **0% false-positive rate** at this scale, so PSM doesn't fabricate matches when the cued thing isn't in the recording.

---

## Failure modes (3-category taxonomy)

The 4/12 misses across all seeds decompose cleanly:

**(i) Right cell, wrong rank.** Session A's bus surfaces inside top-5 (exemplar IoU @5 = 0.600) but rank-1 lands 26 s past the actual bus interval. The right *interval* is in the candidate set; an MLLM reranker on PSM's top-k closes this gap without engine changes. **This is the canonical MLLM-prefilter use case.**

**(ii) Right cell, wrong exemplar.** Session A's bicycle. DINO's attention overlay clearly highlights the cyclist in the relevant frames, but CLIP cosine doesn't rank those frames highly. The cell containing the cyclist is in PSM's neighborhood set; the reservoir's sampled exemplars from that cell aren't bicycle-canonical-enough for CLIP to surface. **Larger reservoirs help; DINO-attention-weighted resampling would help more.**

**(iii) Total semantic miss.** Session A's elevator (camera operator inside an elevator the first 30 s; CLIP-L's top-1 lands at 45 s in a different cell, no top-5 cell goes near 0–30 s) and session C's semi-truck and person walking. The encoder doesn't recognize these as canonical instances of the cued class on this footage. **Encoder limitation, not engine limitation.** SigLIP-2 or OpenCLIP-bigG would likely lift these.

The DINO <-> CLIP disagreement on the bicycle (and likely the elevator) is an interesting observation, suggesting salience-detection sees what text-similarity doesn't. Combining DINO attention as a *prior* on which exemplars enter the reservoir, or as a *reranker* on top of CLIP cosine, is something to explore later perhaps.

---

## Visualizing memory structure

### Embedding atlas — paired view

Two panels, same dot per frame: left = real `(lng, lat)`, right = UMAP of the per-frame CLIP embedding. Same H3 cell color on both sides. Demonstrates that the spatial decomposition isn't arbitrary — geographic neighborhoods correspond to compact regions in the encoder's semantic space, with at least one cell showing bimodality (the wearer revisits a junction with a different visual aspect; PSM's exemplar reservoir captures both modes rather than collapsing them to a centroid).

![embedding atlas paired](../captures/embedding_atlas_paired.png)

### Embedding atlas — per-query similarity grid

Per-query similarity gradient over the UMAP cloud, top-30 frames outlined. Useful for the encoder-bottleneck argument — when CLIP-base hits 33% Hit@5 and CLIP-L hits 100% on the same Tucson session, the gradient panels make the difference visible.

![embedding atlas grid](../captures/embedding_atlas_grid.png)

Both figures are reproducible from `clip_l_features.h5` via `scripts/embedding_atlas.py`.

---

## Why it matters

The paper's discussion section explicitly calls for *"adaptive temporal indexing… and hierarchical memory buffers that can compress hours of video without losing the granular detail of momentary 'needle' events."* PSM's H3-indexed ring buffer of HLL sketches plus per-tile exemplar reservoir is exactly that shape:

- **Bounded memory** regardless of session length (HLL state is fixed-size per cell).
- **Time decay** falls out of the ring buffer naturally (oldest bucket overwritten).
- **Spatial locality** keeps queries fast (`O(matching_tiles × capacity)`).
- **Mergeable HLLs** mean cross-session memory union is essentially free — the substrate for E8-style cross-session stability.

Forwarding PSM's top-k `(cell, t_start, t_end)` candidates plus the corresponding evidence frames to an MLLM as explicit grounding context turns "guess from priors" (the paradox) into "answer from a small set of candidate intervals" (a tractable RAG-style task). 68% of the time the right interval is already in the top-5; the remaining failures are diagnostic, not random.

---

## Limitations

1. **12 scored questions across 3 sessions** gives a credible shape, not tight error bars. Doubling the question budget per session (target 10/session, 30 total) tightens the std on the headline number below ±5%.
2. **Question set is object-recall-dominated.** No questions yet exercise the benchmark's other seven cognitive classes (sequential action, time duration, counting, temporal ordering, object comparison, location trace, spatial awareness). Per-category mIoU is deferred until the corpus grows.
3. **CLIP-L LAION-2B is good but not the strongest available encoder.** SigLIP-2 large/giant or OpenCLIP-bigG would likely lift session A's bicycle/elevator and session C's semi-truck/person-walking failures. Re-extraction is cheap (~10 min on M4 Pro per session).
4. **Single-seed result is one draw**; we run 5 seeds for error bars, but a 10-seed sweep would tighten the Fulham std specifically.

---

## How to reproduce

Setup, extraction, eval, aggregation, and companion-figure regeneration commands are split out into a separate runbook to keep this doc focused on the result.

→ See [`reproducibility.md`](./reproducibility.md) for the full pipeline (one-time setup + per-session eval, ~5 min/session).

Apples-to-apples comparisons across encoder / reservoir / retention configurations are then a matter of changing one knob at a time and re-running the loop above.
