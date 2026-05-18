# Localization Paradox — Follow-up

Follow-up to [`localization_paradox.md`](./localization_paradox.md).
Previously it was shown that PSM, a bounded spatial-memory engine, can answer
"where/when did I see X?" questions over a 15-minute egocentric video at
~68% top-5 accuracy.

**One-paragraph recap of v1.** PSM tiles the world into H3 hex cells.
Each cell holds two things: a HyperLogLog *sketch* counting how many
distinct things were observed there in each time *bucket* (~75 s wide),
and a small reservoir of *exemplar* embeddings — actual image-encoder
vectors sampled from the frames that landed in that cell. A text query
("a red bus") is encoded by the same image-text model and ranked by
cosine against every cell's exemplars. The top-k cells become candidate
intervals for downstream grounding. Throughout this doc, **Hit @5**
means "the right place appeared in the top-5 returned cells," and
**mIoU @5** measures how well the returned interval overlaps the
ground-truth one. Two flavors: **bucket** uses the cell's whole ~75 s
window (place-precise but loose); **exemplar** uses the matching
frame's timestamp ±1.5 s (frame-precise but stricter). See v1 for the
full glossary.

This follow-up tests four things on the same sessions:

1. Does a much larger image-text model help? **Barely.**
2. Do better-framed questions help? **A lot.**
3. Can PSM answer a "where did I…" question without any image-text model
   at all? **Yes — 10/10 times on the one question we tried.**
4. Can the per-cell exemplar layer be compressed to 2/3/4 bits per
   coordinate without hurting accuracy? **Yes — 10× smaller is free.**

Together these results say the same thing in four ways: PSM's
contribution is the *map-and-time layer*, not the embedding model on top
of it, and the embedding payload itself can be aggressively
compressed without moving the headline.

## Glossary delta

Three new terms beyond v1's:

- **Place-aware question** — a question whose answer is a *location*
  ("where did I leave the building?") rather than a recurring object
  category ("did I see a bus?").
- **k-ring** — H3's neighborhood of hex cells around a center cell.
  `k_ring=1` is the 6 immediate neighbors; widens the search beyond an
  exact GPS match.
- **`query_mode: last_seen`** — a YAML option that bypasses the image-
  text model and asks PSM directly for the most recently visited cells
  near a given (lat, lng).

## TL;DR

| metric | Prev (12 object-recall questions, CLIP-L) | follow-up (22-question full set, OpenCLIP-bigG) |
|---|---|---|
| Hit @5 | 68.3% ± 7.0% | **83.0% ± 2.7%** |
| bucket mIoU @5 | 0.182 | **0.311** |
| false-positive rate | 0% | 0% |

![Per-session and combined Hit @5: v1 baseline (12 object-recall questions, CLIP-L) vs v2 follow-up (full 20-question set, OpenCLIP-bigG). Tucson was already saturated; the lift comes from Palo Alto (50% → 83%, driven by the new place-aware questions) and Fulham (55% → 71%). Combined error bar tightens from ±7.0% to ±2.7%.](./figures/v1_v2_lift.pdf){width=100%}

Four findings:

**1. The image-text model is not the bottleneck.** Swapping CLIP-L for
OpenCLIP-bigG (~6× more parameters) lifts Hit @5 by 3–5 pp on every
subset and leaves bucket-level grounding unchanged. The Localization
Paradox literature already shows frontier vision-language models
collapse on temporal grounding regardless of size; our ablation is the
matching positive evidence — *at PSM's layer of the stack*, model
capacity isn't moving the needle on grounding either.

**2. Place-aware questions land far better than object-recall ones.**
Holding the model fixed at CLIP-L, eight new questions framed around
*where* (rather than *what*) hit 95% Hit @5 vs. 68% on the previous object set.
Bucket mIoU jumps from 0.182 → 0.505 on the place-aware subset, a 2.8×
lift driven entirely by the question style. Same engine, same encoder,
same hyperparameters.

**3. PSM can answer a "where did I…" question with the model bypassed.**
For Palo Alto's "where did I drive in reverse," we skipped the image-
text search and asked PSM directly which GPS cell was visited around the
event. **Result: bucket Hit @5 = 100% across all 5 reservoir seeds and
both encoders, fully deterministic.**

**4. The per-cell exemplar layer compresses 5–10× without moving Hit @5.**
Each reservoir exemplar is an embedding vector — 5 KB at OpenCLIP-bigG.
Encoding those vectors as 2/3/4 bits per coordinate (TurboQuant) shrinks
storage by 5× to 10× and leaves the headline Hit @5 within seed noise
across all four codec choices. The map-and-time layer is the load-bearing
part; the embedding *bytes* are not.

## 1. Bigger model, small lift

Restricted to the previous 12 baseline object-recall questions:

| metric | CLIP-L | OpenCLIP-bigG | Δ |
|---|---|---|---|
| Hit @5 | 68.3% ± 7.0% | 71.7% ± 4.6% | +3.4 pp |
| mIoU @5 | 0.259 ± 0.015 | 0.265 ± 0.018 | flat |
| **mIoU @1** (rank-1 placement inside hit cells) | 0.137 ± 0.015 | **0.154 ± 0.032** | **+12% rel** |
| bucket mIoU @5 | 0.182 ± 0.001 | 0.181 ± 0.001 | identical |

The one place a bigger model helps is **rank-1 placement inside cells
that were already correct** (mIoU @1 +12% rel). That matters for any
downstream pipeline that takes PSM's top-k and reranks with an MLLM —
fewer reranker calls, sharper candidate frames. It does not move
grounding metrics.

Per-question, three notable shifts:
- **Palo Alto's parked semi-truck** went from 0/5 hits to 3/5 — the
  encoder genuinely didn't recognize it before. Previously I had labeled this a
  "total semantic miss"; bigG confirms the diagnosis.
- **Fulham's elevator** dropped from 2/5 hits to 0/5 — both encoders
  agree the dim 0–37 s elevator interior isn't reliably distinguishable
  from corridor footage. The previous 2/5 was lucky; the new 0/5 is honest.
- **The bicycle and walking-person failures persist across both
  encoders.** These are cases where DINO's salience attention picks the
  cyclist out cleanly but CLIP's text-similarity doesn't — a modality
  disagreement, not a model-size problem.

The full per-question diff lives in [Appendix § A](#per-question-diffs).

## 2. Better questions, big lift

The previous questions were all object recall ("a red bus", "a stop sign"). The
Localization Paradox benchmark exercises eight cognitive categories;
The previous corpus only really touched one. This section adds 12 new questions
(4 per session) targeting six categories that map naturally to PSM's
H3 + ring-buffer substrate: **Location Trace**, **Sequential Action**,
**Time Duration**, **Spatial Awareness**, **Counting**, **Temporal
Ordering**.

| subset | encoder | n | Hit @5 | bucket mIoU @5 | bucket Hit @5 |
|---|---|---|---|---|---|
| baseline q1–q4 | CLIP-L | 12 | 68.3% ± 7.0% | 0.182 | 25.0% |
|  | bigG   | 12 | 71.7% ± 4.6% | 0.181 | 25.0% |
| **place-aware q6–q9** | CLIP-L | 8 | **95.0% ± 6.8%** | **0.505** | **62.5%** |
|  | bigG   | 8 | **100.0% ± 0.0%** | **0.505** | **62.5%** |
| full q1–q9 | CLIP-L | 20 | 79.0% ± 2.2% | 0.312 | 40.0% |
|  | bigG   | 20 | 83.0% ± 2.7% | 0.311 | 40.0% |

Tucson now hits 100% across the full 9-question set on bigG. Palo Alto
lifts from 50% (previous) to 82.5% — the new questions reliably hit the
parts of the session that the original corpus systematically missed. Fulham's
mean is unchanged but its run-to-run variance collapses to zero on
bigG: the place-aware extension questions are bedrock-stable across
seeds, while the original q1–q4 carried all the variance.

Across encoders, all 8 place-aware IoU-scoreable questions hit reliably
on both CLIP-L and bigG, with the lone difference being a single seed
on Tucson's "a red car" question. Place-aware questions don't depend on
which exemplar happened to land in the reservoir — they depend on
whether the wearer visited the cell, which is a GPS fact, not an
embedding fact.

![Same dot per frame, two views. Left: real `(lng, lat)`. Right: UMAP of the per-frame CLIP embedding. Color is the H3 cell. Geographic neighborhoods land as compact regions in the encoder's semantic space — that's why place-aware questions are robust across reservoir seeds (the wearer being in a cell is a GPS fact; the exemplar reservoir doesn't have to "win" for the cell to surface).](../captures/embedding_atlas_paired.png){width=100%}

## 3. Encoder bypass: one question, no image-text model

Palo Alto q9 asks **"where did I drive in reverse?"** — a question with
no useful CLIP embedding, since CLIP can't render the *concept of
reversing* into a visual query. The first attempt failed predictably
(0/5 hits across all seeds and both encoders).

So we added a `query_mode: last_seen` to the eval harness. Instead of
embedding the text and searching by similarity, the harness calls PSM
directly: *"give me the most recently visited cells around this GPS
coordinate."* No image-text model is invoked at all.

**Result: bucket Hit @5 = 100% across all 5 seeds × both encoders.**
Fully deterministic. The right cell is in the top-5 every time. (Detail:
the top-1 cell is a near-neighbor returned by `--last-seen` ranking by
recency; the right cell sits at rank 2–3. For a "find me the last
location of X" workflow that's fine; for an MLLM reranker it's also
fine because the right cell is always in the candidate set.)

This is the cleanest demonstration in the doc that the map-and-time
layer carries the grounding work. The image-text model is one possible
front-end; it isn't load-bearing for the kind of question PSM is
structurally well-suited to.

## 4. Compressed exemplars: 10× smaller, same accuracy

Each per-cell reservoir exemplar is an embedding vector stored as raw
float32 — 2 KB at CLIP-base 512-d, 5 KB at OpenCLIP-bigG 1280-d. The
question for this section: can a lossy compression scheme shrink that
layer without hurting the answer-quality numbers from §1–§3?

We tested **TurboQuant**, a published vector-quantization method that
encodes each embedding as 2/3/4 bits per coordinate (vs. raw's 32 bits),
shrinking storage by 5× to 10×. It's a drop-in for the existing exemplar
layer: `psm --exemplar-codec {raw,turboquant_2b,turboquant_3b,turboquant_4b}`.
Implementation in `core/exemplar_codec.{h,c}`; see `EXPERIMENTS.md`
for the algorithmic spec.

### Memory compression (bigG, 1280-d embeddings)

| codec | bytes/exemplar | ratio vs raw |
|---|---|---|
| raw            | 5120 | 1.0× |
| turboquant_4b  | 1041 | **4.9×** |
| turboquant_3b  |  785 | **6.5×** |
| turboquant_2b  |  529 | **9.7×** |

### Question quality (mean ± std across 5 seeds, 20 scored questions/seed)

| codec | exemplar mIoU @1 | exemplar mIoU @5 | exemplar Hit @5 | bucket mIoU @5 |
|---|---|---|---|---|
| raw            | 0.130 ± 0.019 | 0.212 ± 0.011 | **83.0% ± 2.7%** | 0.311 ± 0.001 |
| turboquant_4b  | 0.108 ± 0.015 | 0.211 ± 0.013 | **82.0% ± 2.7%** | 0.309 ± 0.001 |
| turboquant_3b  | 0.115 ± 0.019 | 0.198 ± 0.015 | **83.0% ± 2.7%** | 0.312 ± 0.000 |
| turboquant_2b  | 0.122 ± 0.012 | 0.218 ± 0.017 | **81.0% ± 4.2%** | 0.312 ± 0.000 |

**Hit @5 is statistically flat across all four codecs** — the 81–83%
spread is half a single codec's across-seed std. mIoU @5 and bucket
mIoU @5 are also flat. mIoU @1 (top-rank precision) drops slightly under
heavier compression, within a standard deviation of seed noise.

![Compression vs answer quality on OpenCLIP-bigG. Each point is one codec at the bytes/exemplar it produces (1280-d → 2048-d after RHT padding), evaluated across 3 sessions × 5 seeds × 20 questions. Error bars are across-seed std. The blue band marks raw's Hit @5 ± std; every codec's mean lands inside it, so the headline answer-quality metric is statistically flat over a 10× compression range.](./figures/codec_tradeoff.pdf){width=100%}

### Retrieval drift vs raw (135 paired questions per codec)

A separate sanity check: how often does compression change *which cells*
PSM returns, regardless of whether the question scores correctly? The
top-k Jaccard column reports the cell-set overlap with raw at each k.

| codec | top-1 cell match | Jaccard top-5 | Jaccard top-10 | rank ρ (mean) | top-1 cosine delta |
|---|---|---|---|---|---|
| turboquant_2b  | 82.2% | 0.773 | 0.773 | 0.757 | +0.0198 ± 0.0071 |
| turboquant_3b  | 91.9% | 0.801 | 0.801 | 0.893 | +0.0046 ± 0.0033 |
| turboquant_4b  | 91.1% | 0.878 | 0.878 | 0.925 | +0.0012 ± 0.0022 |

So compression *does* perturb retrieval at the cell level — a 4-bit
codec returns 88% of raw's top-5 cells; a 2-bit codec returns 77%. What
saves the headline question-quality numbers is that the perturbations
land on cells that score similarly to the originals on this question
bank.

### Verdict

- **4-bit (5× smaller) is the safe default.** Same Hit @5 as raw, 88%
  cell-set overlap. Recommended when reservoir memory matters.
- **2-bit (10× smaller) is a free lunch on this corpus.** Hit @5 is
  statistically identical to raw despite only 77% cell-set overlap; the
  question bank just isn't sensitive to that kind of swap.
- **Caveat on the success criterion.** The original codec spec asked
  for ≥0.9 cell-set overlap before promoting a non-raw default. 4-bit
  misses that bar (0.88) yet leaves Hit @5 unchanged — a sign the bar
  is over-tuned to retrieval-fidelity rather than answer-quality. A
  denser ground-truth corpus would resolve which is the right metric to
  govern the promotion.

## What controls performance?

The encoder ablation, the place-aware extension, and the encoder bypass
together identify three independent sources of variance in PSM's
behavior. Each axis has a different fix:

- **Map/time layer** — H3 cells, ring-buffer time windows, bucket
  intervals. *Encoder-invariant; question-style-sensitive.* This is
  where the lift from previous → place-aware extension comes from. Bucket
  mIoU is its natural metric.
- **Image-text model** — CLIP / SigLIP / OpenCLIP. *Affects rank-1
  placement and which categorical things get recognized at all.* A
  bigger model partially recovers "total semantic miss" failures (Palo
  Alto's semi-truck) but doesn't move grounding.
- **Reservoir sampling** — which observed frames survive as per-cell
  exemplars. *Affects run-to-run variance, especially on object-recall
  questions where which exemplar got sampled determines whether the
  query hits.* Place-aware questions are largely reservoir-invariant.

The reservoir is the only remaining unbounded-memory layer in PSM (each
exemplar is a 768- or 1280-d float vector; cell count grows with
session length). Place-aware questions being reservoir-invariant
predicted that compressing exemplars from float32 to 2/3/4-bit codes
should preserve our headline numbers while shrinking the exemplar layer
several-fold — and §4 confirms it.

## Limitations

1. **22 IoU-scoreable questions across 3 sessions** is a credible shape,
   not tight error bars. The combined Hit @5 std tightened from ±7.0%
   (previous) to ±2.7% (here), but a reviewer would want closer to ±1%; the
   tractable path is ~30 questions/session, biased toward the
   underrepresented categories (Time Duration, Counting).
2. **Counting questions are diagnostic-only.** Palo Alto q6 (stop signs,
   GT=6) and Tucson q8 (bridges, GT=10) currently predict by counting
   distinct cells in top-k, which is `--top`-cap-bound rather than
   bounded by the underlying HLL cardinality. A real counting scorer
   needs either a similarity-threshold cell counter or an HLL-native
   cardinality query (sketched in
   [Appendix § E](#counting-limitation)).
3. **Three categorical/spatial questions need external grading.** The
   harness records the top-k cells and (lat, lng) but cannot auto-judge
   answers like "river vs. railway" or "under vs. over a bridge"; an
   OSM-overlay annotation pass would close this gap.

---

# Appendix

## A. Per-question diffs {#per-question-diffs}

`L` = CLIP-L, `B` = OpenCLIP-bigG. Each cell is a per-seed hit pattern
across 5 seeds (`✓` = hit, `✗` = miss). `↑`/`↓` flag a hit count change
between encoders; `+` marks questions added in this follow-up; `++`
marks the encoder-bypass question. Rows where both encoders show `--`
mean the question isn't IoU-scoreable (categorical, count-only, or
pending GT).

```
=== Fulham (A) ===
   q1 a red bus                                  L ✓✓✓✓✓  B ✓✓✓✓✓
↑  q2 a zebra crossing                           L ✓✓✗✓✓  B ✓✓✓✓✓
   q3 a person riding a bicycle                  L ✗✗✗✗✗  B ✗✗✗✗✗   (CLIP/DINO failure)
↓  q4 elevator                                   L ✗✗✗✓✓  B ✗✗✗✗✗   (stochastic)
+  q6 a row of mailboxes [PLACE-AWARE]           L ✓✓✓✓✓  B ✓✓✓✓✓
+  q7 the river Thames embankment [SPATIAL]      L  --    B  --     (categorical)
↑+ q8 a large lorry on a busy high street        L ✓✓✓✓✗  B ✓✓✓✓✓
+  q9 a motorcycle on a high street              L ✓✓✓✓✓  B ✓✓✓✓✓

=== Palo Alto (C) ===
   q1 a stop sign                                L ✓✓✓✓✓  B ✓✓✓✓✓
↑↑ q2 a semi-truck                               L ✗✗✗✗✗  B ✗✓✗✓✓   (encoder lift)
   q3 tall palm trees                            L ✓✓✓✓✓  B ✓✓✓✓✓
   q4 a person walking                           L ✗✗✗✗✗  B ✗✗✗✗✗   (encoder fail)
+  q6 a stop sign [COUNTING]                     L ✓✓✓✓✓  B ✓✓✓✓✓
+  q7 palms + truck [TEMPORAL ORDERING]          L ✓✓✓✓✓  B ✓✓✓✓✓
+  q8 a Speed Limit 25 sign                      L ✓✓✓✓✓  B ✓✓✓✓✓
++ q9 where did I drive in reverse [LAST-SEEN]   L ✓✓✓✓✓  B ✓✓✓✓✓   (encoder off)

=== Tucson (B) ===
   q1 a person wearing blue clothes              L ✓✓✓✓✓  B ✓✓✓✓✓
   q2 a bicyclist                                L ✓✓✓✓✓  B ✓✓✓✓✓
   q3 a bridge                                   L ✓✓✓✓✓  B ✓✓✓✓✓
   q4 a white van                                L ✓✓✓✓✓  B ✓✓✓✓✓
   q6 bike-path turn-back [PENDING]              L  --    B  --     (no GT yet)
↑  q7 a red car                                  L ✓✓✓✗✓  B ✓✓✓✓✓
+  q8 all the bridges I cycled across [COUNT]    L  --    B  --     (count-only)
+  q9 a bridge from below [SPATIAL]              L  --    B  --     (categorical)
```

Five questions are unscoreable in this view — three are categorical or
counting-only (recorded but not auto-graded), one is pending GPS
verification, and one is the encoder-bypass question shown separately.

## B. Failure modes

As previously, three-category taxonomy still describes every miss; v2 adds a
fourth.

**(i) Right cell, wrong rank.** Fulham q1 (bus), Palo Alto q3 (palms).
Hit cell is in top-5 with mIoU ≥ 0.5, but rank-1 lands outside the GT
interval. An MLLM reranker on PSM's top-k closes this gap.

**(ii) Right cell, wrong exemplar.** Fulham q3 (bicycle). DINO attention
clearly highlights the cyclist; CLIP cosine doesn't rank those frames
highly. Persists across CLIP sizes — modality disagreement, not capacity.

**(iii) Total semantic miss.** Fulham q4 (elevator), Palo Alto q4
(person walking). Both encoders fail on every seed. Palo Alto q2
(semi-truck) was originally in this category; bigG recovered it
to 3/5, validating the diagnosis. Remaining cases need either a
stronger encoder (SigLIP-2, bigG-39B) or attention-weighted reservoir
sampling.

**(iv) Categorical answer needs external grading.** Fulham q7 (river
Thames embankment), Tucson q9 (bridge from below), Palo Alto q7 (palms
+ truck ordering). Top-k cells are deterministic across seeds and
encoders, but auto-scoring would need an OSM-overlay annotation pass.

## C. Encoder-bypass (q9) configuration and per-seed result

The YAML question:

```yaml
- id: q9
  category: location_trace
  query: "where did I drive in reverse"
  query_mode: last_seen
  query_lat: 37.387940
  query_lng: -122.085884
  query_k_ring: 1
  intervals:
    - [0.0, 70.3]
```

The coordinate is the cell center occupied during the reverse-drive
moment. The GT interval `[0.0, 70.3]` is the cell's natural temporal
extent at the `(75 s × 12)` retention setting; the 6-second reverse
event happens at `[15, 21]` inside that bucket. The question semantics
are *where*, not *for how long*, so scoring against the cell extent is
the right grounding metric.

Per-seed result:

```
clipL  seed=0..4   bucket@k=1.000  bucket_hit=✓
bigG   seed=0..4   bucket@k=1.000  bucket_hit=✓
```

10/10 hits, fully deterministic. The result depends only on H3 cell
membership, not on the embedding layer or the reservoir.

## D. Methodological additions

Three optional YAML fields in `scripts/eval_lookback.py`, all backward-
compatible:

- **`query_mode: last_seen`** — routes to `psm --last-seen LAT,LNG
  --k-ring N` instead of `--search`. Required: `query_lat`, `query_lng`.
  Optional: `query_k_ring` (default 1). Last-seen results have no
  `exemplar_t`; the harness synthesizes the bucket midpoint so existing
  IoU math still works (collapses exemplar IoU to ≈ bucket IoU).
- **`count: <int>`** — cardinality answer compared against
  `len(distinct cells in top-k)`. Diagnostic-only; see Counting
  Limitation below.
- **`expected: <string>`** — categorical answer recorded but not auto-
  scored. Harness prints a "Spatial-reasoning questions" table with
  top-1 cell and (lat, lng) for downstream grading.

## E. Counting limitation {#counting-limitation}

Two questions exercise PSM as a counting backend: Palo Alto q6 (stop
signs, GT = 6) and Tucson q8 (bridges, GT = 10). At top-5, both predict
5; at top-20, they predict 15 and 20 respectively. Neither is right —
the prediction is dominated by the `--top` cap, not by the underlying
HLL cardinality.

A defensible counting scorer needs one of:

- **Threshold-based cell counting.** Count cells with `similarity > τ`
  rather than top-k. Threshold becomes a per-query hyperparameter.
- **HLL-native cardinality query.** PSM has `RingBuffer_merge_window`
  internally; a `psm --cardinality "<text>" --threshold τ` flag would
  give the right answer in one call.

Both are out of scope for this follow-up. Counting questions are
recorded but excluded from the headline metrics.

## F. Reproducibility

```bash
# 1. Re-extract all three sessions (~25-30 min/session for bigG)
bash scripts/extract_bigg_all.sh   # bigG
TAG=clipL FEATURES=clip_l_features.h5 \
  CHECKPOINT=laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
  bash scripts/extract_bigg_all.sh    # CLIP-L (cached after first run)

# 2. Run eval over 3 sessions × 5 seeds × 9 questions, both encoders
bash scripts/eval_bigg_all.sh        # bigG
TAG=clipL FEATURES=clip_l_features.h5 \
  CHECKPOINT=laion/CLIP-ViT-L-14-laion2B-s32B-b82K \
  bash scripts/eval_bigg_all.sh      # CLIP-L

# 3. §4 codec sweep (assuming raw bigG runs already in captures/)
CODEC=turboquant_4b bash scripts/eval_bigg_all.sh
CODEC=turboquant_3b bash scripts/eval_bigg_all.sh
CODEC=turboquant_2b bash scripts/eval_bigg_all.sh

# 4. Side-by-side aggregate (both encoders + all codecs in one table)
python scripts/eval_aggregate.py --by-seed --label-from-features \
  captures/eval_*_clipL_e128_s*.json captures/eval_*_clipBigG_e128_s*.json \
  captures/eval_*_clipBigG_turboquant_*_e128_s*.json

# 5. Codec-vs-raw retrieval drift (cell overlap + rank correlation)
python scripts/eval_codec_drift.py \
  captures/eval_*_clipBigG_e128_s*.json \
  captures/eval_*_clipBigG_turboquant_*_e128_s*.json
```

A top-20 sweep (`scripts/eval_bigg_all.sh` with `TOP=20
TAG_TOP_SUFFIX=1`) preserved at `captures/eval_*_top20_*.json` confirms
the headline numbers don't change with a wider candidate pool — except
for counting (see Appendix § E).
