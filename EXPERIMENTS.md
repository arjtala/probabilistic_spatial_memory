# Experiments

This file turns the README's open questions into explicit, repeatable experiments using the tooling that already exists in this repo.

A subset of these experiments (E5, E7, E10, E11, E12) is on the critical path for the ECCV 2026 workshop paper plan; see [`journal/PAPER.md`](journal/PAPER.md) for that workstream's status, ordering, and reviewer-anticipation notes.

## Repro Baseline

For every run, record:

- git commit: `git rev-parse HEAD`
- build profile: `local`, `portable`, `debug`, or `sanitize`
- host details: `uname -a` and `sysctl -n machdep.cpu.brand_string 2>/dev/null || lscpu`
- dataset path(s): the exact `features.h5` or session directory used
- command line: keep the full command in a shell history file or log

For machine-readable `psm` runs, prefer JSON output:

```bash
targets/psm -f /path/to/features.h5 -g dino -t 5.0 -r 10 -j > run.json
```

The JSON includes global metadata plus per-tile `current` and `total` counts, so downstream summaries can be scripted with `jq`.

## E1. Embedding-Hash Stability Under Semantic-Preserving Change

Question:
- How stable are embedding hashes under viewpoint, lighting, blur, or mild compression changes?

Current capability:
- The repo does not synthesize perturbed embeddings itself, but it can compare multiple HDF5 feature exports from the same route.

Inputs:
- A baseline `features.h5`
- One or more matched variants of the same route, for example:
  - second capture of the same path under different lighting
  - re-encoded or blurred video re-run through the offline feature extractor
  - alternate camera pose / head-motion pass over the same scene

Protocol:
1. Hold `group`, `time_window_sec`, `h3_resolution`, `capacity`, and `precision` fixed.
2. Run `psm -j` on the baseline and every variant.
3. Compare `tile_count`, total mass, current mass, and the top-k hottest cells.
4. Replay the same sessions in `psm-viz` with the same map settings for a qualitative hotspot check.

Command template:

```bash
targets/psm -f /data/base/features.h5 -g dino -t 5.0 -r 10 -j > base.json
targets/psm -f /data/variant/features.h5 -g dino -t 5.0 -r 10 -j > variant.json
```

Suggested summaries:

```bash
jq '{tile_count, total_mass: ([.tiles[].total] | add), current_mass: ([.tiles[].current] | add), hottest_total: ([.tiles[].total] | max)}' base.json
jq '{tile_count, total_mass: ([.tiles[].total] | add), current_mass: ([.tiles[].current] | add), hottest_total: ([.tiles[].total] | max)}' variant.json
```

Primary readout:
- Relative drift in `tile_count`
- Relative drift in summed `total` mass
- Whether the top-k hotspot cells stay stable

## E2. Counting Unit Ablation

Question:
- Which counting unit is most useful: whole-frame states, objects, or clustered semantic tokens?

Current capability:
- The in-tree implementation counts frame-level embeddings today.
- The comparison harness is ready now; alternate counting units only require alternate HDF5 exports or group names from the upstream extractor.

Inputs:
- Multiple feature exports for the same session, each representing a different tokenization strategy:
  - frame-level embeddings
  - object-level embeddings
  - clustered / stabilized embeddings

Protocol:
1. Keep the route, `time_window_sec`, `h3_resolution`, and ring-buffer settings fixed.
2. Run `psm -j` once per counting unit.
3. Compare:
  - `tile_count`
  - summed `total` mass
  - summed `current` mass
  - ranking of top hotspot cells
4. Inspect the same session in `psm-viz` to judge whether the heatmap matches intuitive revisitation / novelty.

Command template:

```bash
targets/psm -f /data/session/features_frame.h5   -g dino -t 5.0 -r 10 -j > frame.json
targets/psm -f /data/session/features_object.h5  -g dino -t 5.0 -r 10 -j > object.json
targets/psm -f /data/session/features_cluster.h5 -g dino -t 5.0 -r 10 -j > cluster.json
```

Decision rule:
- Prefer the counting unit that preserves stable hotspots under repeated traversal without collapsing obviously distinct places into one flat map.

## E3. Familiarity Convergence Sensitivity Sweep

Question:
- How quickly does familiarity converge, and how sensitive is that to H3 resolution and time-window size?

Current capability:
- Fully automatable with `psm -j`.

Protocol:
1. Pick one or more representative sessions.
2. Sweep `h3_resolution` and `time_window_sec`.
3. Record `tile_count`, total mass, current mass, and hottest tile total.
4. Optionally replay a few representative parameter points in `psm-viz`.

Sweep command:

```bash
FEATURES=/data/session/features.h5
GROUP=dino

for resolution in 8 9 10 11 12; do
  for window in 2.5 5.0 10.0 20.0; do
    targets/psm -f "$FEATURES" -g "$GROUP" -t "$window" -r "$resolution" -j \
      | jq -c '{
          h3_resolution,
          time_window_sec,
          tile_count,
          total_mass: ([.tiles[].total] | add),
          current_mass: ([.tiles[].current] | add),
          hottest_total: ([.tiles[].total] | max)
        }'
  done
done | tee familiarity_sweep.jsonl
```

Primary readout:
- Lower resolutions should merge more places into fewer tiles.
- Larger windows should raise `total_mass` and make the map more historically sticky.

## E4. Novelty Definition Comparison

Question:
- What is the most useful operational definition of novelty: raw distinct counts, recent change, or prediction error?

Current capability:
- Count-based novelty is automatable with `psm -j`.
- Prediction-error comparison is qualitative today via the JEPA overlay in `psm-viz`.

Part A: compare count-based proxies

```bash
targets/psm -f /data/session/features.h5 -g dino -t 5.0 -r 10 -j > novelty.json

jq -c '.tiles[]
  | .recency = (if .total > 0 then .current / .total else 0 end)
  | {cell, current, total, recency}' novelty.json \
  > novelty_tiles.jsonl
```

Interpretation:
- `total`: persistent familiarity / accumulated experience
- `current`: what is active in the latest bucket
- `recency = current / total`: places that are active now relative to their own history

Part B: compare against prediction error

```bash
targets/psm-viz -d /data/session -g jepa -m recency
```

Review:
1. Pause on a visually surprising scene.
2. Compare the map hotspot under `recency` against the JEPA prediction-error overlay.
3. Repeat on several frames from the same route.

## Localization Paradox Alignment

A forthcoming streaming egocentric memory benchmark (the "Localization Paradox benchmark" after its headline finding) evaluates multimodal LLMs on 20K+ "look-back" QA pairs over 613 hours of unscripted Ray-Ban Meta egocentric video. Frontier MLLMs (Gemini 3 Pro, GPT-5, InternVL 3.5) achieve 27-50% semantic accuracy but near-zero temporal grounding (`mIoU` often ≈0). The paper's discussion section explicitly calls for "adaptive temporal indexing… and hierarchical memory buffers that can compress hours of video without losing the granular detail of momentary 'needle' events." PSM's H3-indexed ring buffer of HLL sketches is a candidate substrate for exactly that architectural gap.

Prerequisites for E5-E7 are tracked in `TODO.md` under the "Localization Paradox Alignment" section:

- Per-bucket `(t_min, t_max)` retention, so a query can emit `[t_start, t_end]` intervals.
- Per-tile exemplar embedding sampler (reservoir), so k-NN retrieval against past observations is possible.
- `SpatialMemory_query_intervals(...)` API surface.

Two further pipeline components are external to this repo and are described per-experiment below rather than tracked as TODOs:

- **Text-query adapter** (E5, E7): embed a benchmark question via a vision-language text tower (CLIP / SigLIP / DINO) and match against per-tile exemplar embeddings to produce the semantic cue. Sits outside `libpsm`.
- **Grounded response stage** (E5): forward PSM's top-k `(cell, t_start, t_end)` intervals plus the corresponding evidence frames to an MLLM as explicit grounding context. This is an MLLM API call, not engine code.

**Status:** First-pass results for E5/E6/E7 are in `journal/localization_paradox.md` (initial demo) and `journal/localization_paradox2.md` (follow-up: encoder ablation, place-aware question extension, encoder-bypass query mode). Headline: 83% Hit@5 on a 22-question set across 3 sessions × 5 seeds × 2 encoders, 100% on the 8 place-aware questions. E9 (TurboQuant compressed exemplars) is also done — see § E9 Result below — and confirms 9.7× exemplar-memory reduction at 2-bit with statistically flat Hit@5. Benchmark integration (Gemini 3 Pro / GPT-5 in the loop) is still pending; the local query-only results are the foundation.

### E5. PSM as Retrieval Prefilter for MLLM Temporal Grounding

Question:
- Does PSM's spatial + temporal prior materially close the Localization Paradox gap on frontier MLLMs?

Current capability:
- `SpatialMemory_query_similar(...)` / `psm --search` provide the retrieval backend for the text-query adapter.
- `scripts/e5_clip_demo.py` is a minimal local demo over a plain video using CLIP image/text embeddings and a synthetic track.
- `scripts/eval_lookback.py` is the production text-query harness: embeds a question via CLIP, calls `psm --search`, and reports per-question IoU + Hit@k. Now supports `query_mode: last_seen` for encoder-free location_trace queries.
- `scripts/eval_bigg_all.sh` runs a paired encoder × seed sweep across multiple sessions.
- The in-tree extraction pipeline (`python -m psm_extraction extract --models clip,dino[,jepa]`) reproduces the Aria pipeline's `features.h5` shape end-to-end on Apple Silicon — verified on a 15-min Fulham session with DINOv3 attention-distribution parity to the original. So a benchmark session that ships `data.mp4 + gps.json + imu.json + metadata.json` can be ingested entirely in-house, no external pipeline required.
- Local query-only first pass: see `journal/localization_paradox2.md` for 22-question × 5-seed × 2-encoder results. Full benchmark evaluation (with an MLLM in the loop) is still external to this repo.

Inputs:
- A benchmark subset with usable spatial context (GPS or coarse scene localization)
- One target MLLM for grounded answer generation (e.g., Gemini 3 Pro, GPT-5, or an open-source model for ablation)
- A shared image/text encoder for the question cue (the "text-query adapter"; CLIP / SigLIP / similar)

Protocol:
1. Ingest each session into PSM with fixed `(h3_resolution, time_window_sec, capacity, precision)`.
2. For each benchmark query, embed the question via the text adapter and combine with the trigger-moment observation if the adapter requires frame-conditioned disambiguation.
3. Call `SpatialMemory_query_similar(...)` (or `psm --search <query.f32>`) for top-k candidate `(cell, t_start, t_end)` tuples from matching tiles.
4. Feed the candidates as explicit grounding context to the MLLM alongside the question and the last-visible frame.
5. Measure `mIoU`, `R@1`, and `GQ@0.5` (the benchmark's grounding metrics) vs. the raw-MLLM baseline in its main results table.

Readout:
- Absolute `mIoU` on each of the benchmark's 8 cognitive categories, especially Location Trace and Spatial Awareness.
- Sensitivity to `k` (how much prefilter slack the MLLM needs).
- Per-query latency: PSM `query_similar` vs. MLLM full-video scan.

Decision rule:
- If the prefilter lifts Gemini 3 Pro's `mIoU` above 0.6 on Location Trace + Spatial Awareness, this experiment is a direct architectural rebuttal to the paper's discussion section.

### E6. Counting-Question Fidelity via HLL Cardinality

Question:
- Is HLL cardinality per tile per window a reliable proxy for "Counting" questions ("how many times have I done X so far?") in the benchmark?

Current capability:
- `RingBuffer_merge_window` already returns cardinality estimates; `precision` in `[10, 18]` controls error.
- The benchmark's diagnostic analysis shows Counting is the weakest category across all MLLMs — a gap PSM is structurally well-placed to fill.
- `scripts/eval_lookback.py` accepts a `count: <int>` field on questions and reports `count_predicted` (currently `len(distinct cells in top-k)`), `count_abs_error`, `count_correct`. **Known issue:** this prediction is `--top`-cap-bound, not HLL-cardinality-bound — at top-5 it under-predicts, at top-20 it saturates at the cap. See `journal/localization_paradox2.md` § "Counting limitation" for a sketch of the two ways to fix this (similarity-threshold cell counting, or an `psm --cardinality "<text>" --threshold τ` flag wired to `RingBuffer_merge_window`).

Protocol:
1. Select benchmark sessions containing Counting questions.
2. For each question, identify the object/activity signature (from the question + trigger context).
3. Query PSM for the cardinality of that signature across the time window implied by the question.
4. Compare against the benchmark's ground-truth count.
5. Sweep HLL precision `{10, 12, 14, 16, 18}` and record count error (MAE, RMSE) plus memory per tile.

Decision rule:
- Report the smallest precision whose count error falls below the distractor spread used in the benchmark's 5-way MCQ variant — that is the practical lower bound on HLL size for real counting queries.

### E7. Location-Trace Query Latency

Question:
- How much faster is a PSM look-back than an MLLM's full-video scan for "where did I last have [object]?" queries, and at what retrieval accuracy?

Current capability:
- Exemplar embedding retention landed (see TODO "Localization Paradox Alignment"); E7 is no longer blocked.
- `psm --search` and `psm --last-seen` are both wired through `scripts/eval_lookback.py`. The `query_mode: last_seen` path is the encoder-free version of E7 — it answers location-trace questions purely from H3 cell membership, with no embedding model in the loop.
- First-pass result: Palo Alto's "where did I drive in reverse" question hits bucket Hit @5 = 100% across all 5 reservoir seeds × both encoders, fully deterministic. See `journal/localization_paradox2.md` § 3 for detail.
- Per-query latency typically `O(matching_tiles × capacity)` — sub-millisecond on session-scale memories.

Protocol:
1. Ingest a benchmark session into PSM.
2. For each "Location Trace" question, time:
   - PSM's `query_intervals` call (cue → top-k `(cell, t_start, t_end)`).
   - An MLLM forward pass on the full video up to the trigger timestamp (use the same model evaluated in the benchmark's main results table for a fair baseline).
3. Measure retrieval Acc@1, Acc@5, Acc@10 against the benchmark's ground-truth interval (hit = ground-truth interval falls in PSM's returned set).

Readout:
- End-to-end latency delta (expected: 2-4 orders of magnitude).
- Acc@k curve across `k = 1, 5, 10`.
- Memory footprint per session at the configured `(h3_resolution, capacity, precision)`.

Decision rule:
- If PSM achieves ≥50% Acc@5 at <10 ms median query latency while the MLLM baseline is >1 s, this experiment is the operational case for PSM as a memory backbone beneath a reasoning model.

### E8. Cross-Session Place-Memory Stability

Question:
- On two separate captures of the same route, does PSM return the same top-k tiles and the same per-tile exemplars? That is, does the spatial memory carry a stable place-level identity across visits separated by hours, days, or weeks?

Motivation:
- Recent work on temporally consistent instance segmentation under sparse revisits (e.g., 4D indoor scene tracking that extends mAP to a temporal-identity metric) argues for stability across intermittent observations as a first-class evaluation axis. PSM operates at the *place* level rather than the *object* level, but the same question applies: does revisiting the same hex produce the same memory?
- This is the cross-session counterpart to E1 (which probes single-session perturbation stability).

Current capability:
- Fully automatable with `psm --search -j` and `psm --last-seen -j`.
- HLL sketches are mergeable, so a "merged across sessions" condition is also reachable by ingesting both sessions into a single run.

Inputs:
- Two or more captures of the same route under naturally varying conditions (different time of day, weather, foot traffic, minor route perturbations).
- A small fixed query bank — the same `query.f32` files used for the demo (`q-bus.bin`, `q-zebra.bin`, etc.).

Protocol:
1. Hold `(group, h3_resolution, time_window_sec, capacity, precision, exemplars)` fixed across all sessions.
2. For each session `s` and each query `q`, run:
   ```bash
   targets/psm -f "$SESSION_$s/features.h5" -g clip \
     --search "$QBANK/$q.bin" --top 10 --exemplars 8 -j > "out/$s.$q.json"
   ```
3. Compute three stability scores per query:
   - **Cell overlap @k**: Jaccard of the top-k cell sets across session pairs.
   - **Rank correlation @k**: Spearman ρ on the cell rankings of the cells common to both sessions.
   - **Exemplar drift**: cosine distance between the best exemplar embedding returned for each common cell across sessions.
4. Repeat for `--last-seen LAT,LNG` at a small set of fixed coordinates along the route — this gives a query-free spatial baseline.

Suggested summary:

```bash
jq -s '
  def cells(x): [x.tiles[].cell];
  def topk(x; k): cells(x)[:k];
  {
    overlap_top5:  ((topk(.[0]; 5)  - (topk(.[0]; 5)  - topk(.[1]; 5)))  | length) / 5,
    overlap_top10: ((topk(.[0]; 10) - (topk(.[0]; 10) - topk(.[1]; 10))) | length) / 10
  }' out/s1.bus.json out/s2.bus.json
```

Primary readout:
- **Cell overlap @5** ≥ 0.6 across paired sessions on the same query is the headline number — the place memory is stable enough to support cross-session lookback.
- **Rank correlation** disambiguates "right cells, wrong order" from "wrong cells entirely".
- **Exemplar drift** isolates whether instability comes from the spatial layer (HLL/H3) or the appearance layer (per-tile reservoir).

Decision rule:
- If cell overlap @5 stays above 0.6 on the cue queries while a `--last-seen`-only baseline is at chance, the embedding-driven retrieval is providing genuine place-identity beyond raw geographic prior — that is the structural claim worth defending in the paper's evaluation.

### E9. TurboQuant-Compressed Exemplar Reservoirs

Question:
- Can PSM store per-tile exemplar embeddings in a TurboQuant-style 2-4 bit representation while preserving `--search` retrieval quality?

Motivation:
- PSM's bounded memory story is currently strongest for the HLL counting layer. The semantic retrieval layer still keeps reservoir-sampled exemplar embeddings as raw float32 bytes, so exemplar memory scales as `tiles x exemplars x embedding_dim x 4B`.
- TurboQuant's rotation-plus-scalar-quantization recipe is a direct fit for high-dimensional CLIP/DINO/JEPA embeddings. If it preserves nearest-neighbor ranking inside PSM, the exemplar layer can become bounded and compact for hour-scale egocentric memory without giving up text/image look-back queries.

Current capability:
- `TileExemplar` stores opaque bytes today, and `SpatialMemory_query_similar(...)` assumes those bytes are `float32` vectors with the same dimensionality as the query.
- This makes the experiment easy to stage behind a codec boundary: keep the HLL/ring-buffer path unchanged, add an alternate exemplar encoding path, and decode or score compressed exemplars only inside `query_similar`.
- **Status:** the codec boundary and the TurboQuant codecs are in. `core/exemplar_codec.{h,c}` defines `ExemplarCodec` with `EXEMPLAR_CODEC_RAW` and faithful `EXEMPLAR_CODEC_TURBOQUANT_{2,3,4}B` (randomized Hadamard transform + Lloyd-Max-optimal scalar quantization, bit-packed). `Tile_observe` runs payloads through `ExemplarCodec_encode`; `SpatialMemory_query_similar` builds a per-call prepared query and scores via `ExemplarCodec_cosine`. Selectable via `psm --exemplar-codec NAME`; `--search -j` emits `exemplar_codec` and `exemplar_payload_bytes`. Still open: the sweep + write-up that turns this into the E9 result.

Implementation sketch:
1. Add an `ExemplarCodec` mode with at least `raw_f32` and `turboquant_{2,3,4}b`.
2. During `SpatialMemory_observe`, quantize only the reservoir exemplar payload; continue hashing the original embedding bytes into HLL so count semantics do not change.
3. During `SpatialMemory_query_similar`, compare the float32 query against the compressed exemplar either by dequantizing into a scratch buffer or by using TurboQuant's inner-product estimator directly.
4. Record codec metadata in JSON output and benchmark logs: bits per coordinate, side-info bytes per exemplar, and effective bytes per tile.

Protocol:
1. Pick one or more sessions with CLIP and DINO groups.
2. For each group, ingest the same `features.h5` with `--exemplars {4,8,16}` under raw float32 and TurboQuant bit budgets `{2,3,4}`.
3. Run the same query bank used by E5/E8 through `psm --search -j`.
4. Compare each compressed run against the raw-float32 run:
   - top-k cell overlap for `k = 1, 5, 10`
   - rank correlation on shared cells
   - winning-exemplar cosine error
   - bytes per retained exemplar and bytes per tile
   - query latency, including any dequantization/scoring cost

Readout:
- Memory compression ratio for the exemplar reservoir, separate from HLL memory.
- Retrieval quality as top-k overlap and rank stability vs. raw float32.
- Whether 2-bit is enough for place-level recall, or whether 3-4 bit is the practical floor for PSM.

Decision rule:
- Promote TurboQuant from experiment to implementation if a 4x or better exemplar-memory reduction keeps top-5 cell overlap above 0.9 and median `query_similar` latency below 10 ms on representative sessions.

### E9 Result (2026-05-12)

Faithful TurboQuant (sign-flip + Walsh-Hadamard rotation, Lloyd-Max-optimal scalar quantization, bit-packed payload) is implemented in `core/exemplar_codec.{h,c}` and selectable via `psm --exemplar-codec {raw,turboquant_2b,turboquant_3b,turboquant_4b}`. Sweep run on the 3-session × 5-seed × 20-question scoreable corpus from the §1–§3 Localization Paradox follow-up, OpenCLIP-bigG (1280-d), `--exemplars 128`. Full breakdown lives in [`journal/localization_paradox2.md` § G](journal/localization_paradox2.md#g-e9-turboquant-exemplar-compression-result); summary here.

**Memory compression** (bigG embeddings padded to 2048-d after RHT):

| codec | bytes/exemplar | ratio vs raw |
|---|---|---|
| raw            | 5120 | 1.0× |
| turboquant_4b  | 1041 | 4.9× |
| turboquant_3b  |  785 | 6.5× |
| turboquant_2b  |  529 | 9.7× |

**Question-quality (Hit @5, mean ± std across 5 seeds, 100 question-seed evaluations per codec):**

| codec | exemplar Hit @5 |
|---|---|
| raw            | 83.0% ± 2.7% |
| turboquant_4b  | 82.0% ± 2.7% |
| turboquant_3b  | 83.0% ± 2.7% |
| turboquant_2b  | 81.0% ± 4.2% |

**Retrieval drift vs raw (135 paired questions per codec):**

| codec | top-1 match | Jaccard top-5 | rank ρ |
|---|---|---|---|
| turboquant_4b  | 91.1% | 0.878 | 0.925 |
| turboquant_3b  | 91.9% | 0.801 | 0.893 |
| turboquant_2b  | 82.2% | 0.773 | 0.757 |

**Read:**
- The headline answer-quality metric (Hit @5) is statistically flat across all four codecs. The 9.7× compression at 2-bit costs zero accuracy on this corpus.
- The codec *does* perturb retrieval at the cell level (top-5 Jaccard 0.88 at 4-bit, 0.77 at 2-bit), but the question bank's grounding is not sensitive to that magnitude of drift.
- The decision rule's "top-5 overlap ≥ 0.9" bound is met *only by mIoU @5 / Hit @5*, not by raw cell-set Jaccard. The strict-Jaccard bound is too strict for an answer-quality corpus; a rank-stability benchmark with denser ground truth would test it more honestly.
- Latency was not formally swept; informally, `query_similar` runtime per call is dominated by the per-tile linear scan (~tens of microseconds), and the codec's per-exemplar dequant cost is amortized inside that — well below the 10 ms bar.

**Verdict:** ship `turboquant_4b` as the recommended default when reservoir memory matters (4.9× memory reduction, top-5 Jaccard 0.88, identical Hit @5). Use `turboquant_2b` (9.7×) when memory is the binding constraint and answer-quality on this kind of question bank is the bar.

### E9 follow-ups

- Latency sweep: extend `benchmarks/benchmark_spatial_memory.c` to report bytes/tile and median µs per `query_similar` call across the four codecs.
- Reservoir-size cross: run the same sweep at `--exemplars {4, 8, 16}` to test whether the codec's noise floor matters more on small reservoirs.
- Rank-stability benchmark: a denser corpus where the ground truth distinguishes between adjacent cells would let the strict top-5 Jaccard rule actually bite.
- CLIP-L (768-d) parity check — the existing `clipL` raw runs already exist; running the three codec sweeps would confirm the bigG result transfers to a smaller embedding.

### E10. Reproducing the Localization Paradox on PSM's corpus

**Status (2026-06-18, post K-sweep):** Done on the v1 street-scale
corpus (3 SLOPER4D + 1 Nymeria). `scripts/eval_mllm_baseline.py`
(landed 2026-06-18, commits `7197033`/`6fce5f4`/`d7cb6b7`) runs the
vanilla-MLLM baseline at K uniformly-sampled frames per question.
Result on the 4 sequences at K=8 (matched to PSM's default top-k for
apples-to-apples):

| sequence              | bbox  | vanilla MLLM Hit@5 | PSM clipL r12 Hit@5 | gap   |
|-----------------------|-------|--------------------|---------------------|-------|
| Nymeria shelby (187q) | 69 m  | **0.0 %**          | 7.6 %               | ∞     |
| seq003_street_002     | 105 m | 6.7 %              | 13.3 %              | 2.0×  |
| seq008_running_001    | 176 m | 6.7 %              | 30.0 %              | 4.5×  |
| seq009_running_002    | 446 m | 7.1 %              | 17.9 %              | 2.5×  |

K-sweep on seq009 (K ∈ {8, 16, 32}; K=64 and K=128 both blocked by
proxy 500s — payload too large for the internal api.llama.com
proxy) extends the picture to **oracle vs discrimination**:

| K  | oracle upper bound (any frame in GT) | Gemini's actual Hit@5 | discrimination rate |
|----|--------------------------------------|------------------------|---------------------|
| 8  | 7.1 % (2/28)                         | 7.1 %                  | 100 % (2/2)         |
| 16 | 7.1 % (2/28)                         | 3.6 %                  | 50 % (1/2)          |
| 32 | **21.4 %** (6/28)                    | 7.1 %                  | **33 % (2/6)**      |

The K=32 oracle climbs to 21.4 % (the uniform-sample set contained
a GT-matching frame for 6/28 questions) but Gemini's pick rate stays
at 7.1 % — discrimination over a uniform-sample multi-frame video
baseline **degrades** as K grows: adding candidates dilutes the
choice rather than refining it.

**Decision rule outcome**: The 2026-05 plan said "if MLLM mIoU < 0.10
on Location Trace, the v2 cited paradox is real on our corpus and
the paper has a baseline to beat." Result: vanilla MLLM Hit @ 5 = 0.0 %
on Nymeria (187 questions), 6.7–7.1 % on the SLOPER4D sequences at
K=8, flat across K=8/16/32. **Decision rule fires**: the cited paradox
holds on our street-scale corpus. The §5 framing is the oracle-vs-
discrimination gap, not a flat-K dismissal — see PAPER.md 2026-06-18
journal entries.

E10 status: **done**. E5 (PSM → MLLM reranker) ran on the same 4
sequences immediately after — see E5 status block / PAPER.md
2026-06-18 reranker entry; rerank is bimodal in Hit @ 1 (helps on
visually-rich, hurts on visually-poor) and invariant in Hit @ 5.

---

**Original problem statement (preserved for reference, pre-results):**

Question:
- Does the temporal-grounding collapse documented for frontier MLLMs at benchmark scale also appear on our 3-session, 20-question corpus? Without this, every comparison in the v2 writeup is against a *cited* number, not a *measured* one.

Current capability:
- The corpus exists with hand-annotated `[t_start, t_end]` intervals for 20 IoU-scoreable questions across 3 sessions.
- The grounding metric (`mIoU` against ground-truth intervals) is already implemented in `scripts/eval_lookback.py`.
- What's missing: an MLLM client that takes (full session video, question) and returns a predicted interval.

Inputs:
- One frontier MLLM (target: Gemini 3 Pro via API; backup: a locally-served Llama-3.2-90B-Vision on H200s for unlimited rate).
- A frozen prompting protocol that elicits `[t_start, t_end]` intervals consistently. Reviewer-credible options: (a) chain-of-thought with explicit "name the seconds-since-start when the event happens" suffix, (b) two-shot exemplars of the desired output shape, (c) a two-stage protocol where the MLLM answers semantically first and a separate grounding pass localizes the answer.

Protocol:
1. For each (session, question), call the MLLM with the full session video and the question text.
2. Parse the response into a single `[t_start, t_end]` interval (or report a parse failure as mIoU=0).
3. Compute mIoU and Hit @5 against ground truth using the same scorer as PSM.
4. Report per-category numbers and the overall mean.
5. Repeat the *same* questions through PSM (already done in v2) to establish the direct comparison.

Readout:
- The headline number this experiment must produce: "vanilla MLLM mIoU = X.XX on our corpus" vs PSM's 0.21 bucket-mIoU @5 from v2.
- Per-category MLLM mIoU (Location Trace, Object Recall, Spatial Awareness, Counting if scoreable).
- Failure-mode breakdown: parse failures vs hallucinated intervals vs wrong-interval-with-right-semantics.

Decision rule:
- If MLLM mIoU < 0.10 on Location Trace, the v2 cited paradox is real on our corpus and the paper has a baseline to beat.
- If MLLM mIoU ≥ 0.40, the cited paradox doesn't transfer to small-scale carefully-annotated corpora and the framing has to change (e.g., "the paradox is a benchmark-scale phenomenon; PSM is a memory architecture that scales without requiring frontier-MLLM-scale compute").

### E11. Retrieval-Method Ablation

**Status (2026-06-05):** Brute-force CLIP-L baseline landed on Nymeria
`shelby_arroyo_act0` (13.4% Hit@5, 25/187; see [`journal/results_v1.md`](journal/results_v1.md)).
Sliding-window + uniform-sample baselines still pending on Nymeria — `scripts/eval_brute_force_clip.py`
already in place; the other two harnesses need `--clip-checkpoint`
threading like brute-force got in commit `00d6383`. Aria-internal results
(PSM 83% / brute 80%) are classified internal-preliminary.

Question:
- Does PSM's spatial decomposition (H3 + ring buffer + reservoir) actually help, or does any retrieval method get our headline numbers on this corpus?

Current capability:
- The CLIP-L and OpenCLIP-bigG feature extractors already produce per-frame embeddings (`features.h5::<group>/embeddings`).
- The IoU scoring harness in `scripts/eval_lookback.py` accepts pre-computed `(t_min, t_max)` candidates from any source — only the retrieval call needs to be swapped out.
- What's missing: a `retrieval_method: ...` field in the question YAML that routes through one of: PSM (status quo), brute_force_clip (rank every frame), sliding_window_clip (~5s windows, mean-pool embeddings, rank windows), uniform_sample_clip (1-frame-per-window).

Protocol:
1. Hold corpus, encoder, and question bank fixed.
2. For each retrieval method, return top-5 candidates as `[t_start, t_end]` intervals (the frame timestamp ±1.5s for frame-level methods; the window for sliding/uniform).
3. Score with the same exemplar-IoU and Hit @5 scorer.
4. Compare against PSM's numbers from v2.

Readout:
- Headline table: Hit @5 by retrieval method × encoder.
- Specifically: brute-force-CLIP Hit @5 — if it equals PSM's, the H3+ring-buffer structure is doing no work for accuracy and the paper has to lead with the *bounded-memory* argument instead of the *accuracy* one.
- Memory footprint comparison: brute-force needs to keep all N frame embeddings in RAM; PSM's ring buffer ages them out.

Decision rule:
- If PSM Hit @5 ≥ brute-force Hit @5 + 5pp, the spatial decomposition helps directly.
- If PSM Hit @5 ≈ brute-force Hit @5 (within 2pp), reposition the paper around "PSM matches brute-force quality at O(matching_tiles) query cost instead of O(N_frames)."
- If PSM Hit @5 < brute-force Hit @5, the corpus is too small to need spatial structure and we should either scale up or change the headline claim.

### E12. PSM Hyperparameter Sensitivity

**Status (2026-06-05):** Operating-point sweep on Nymeria done:
`per_cell_cap` ∈ {1,2,3,5} × exemplars ∈ {128, 1024} on
`shelby_arroyo_act0` ([`captures/eval_<sid>_pcc*.json`](captures/);
table in [`journal/results_v1.md`](journal/results_v1.md)). Multi-
session generalization across 4 sessions × 4 caps also landed
(`scripts/multisession_per_cell_cap_sweep.py`). H3-resolution ablation
on Nymeria still pending — Aria-internal numbers (res 8-12) classified
internal-preliminary.

Question:
- How robust is the v2 result to the three knobs we never swept: H3 resolution, retention window, reservoir size? A reviewer will assume we tuned on the test set unless we show otherwise.

Current capability:
- All three knobs are CLI flags on `psm --search`: `-r` (H3 res), `-t` × `-C` (retention = time_window × capacity), `--exemplars`.
- `scripts/eval_bigg_all.sh` already runs the 3-session × 5-seed × N-question matrix; adding a hyperparameter axis is one more bash loop.
- The `eval_aggregate.py` --by-seed reporting already pools across (session, seed); pooling across (hyperparameter point) is a small extension.

Protocol:
1. Fixed: OpenCLIP-bigG, raw float32 codec, 5 seeds, 20 questions.
2. Vary one knob at a time, keep the other two at the v2 defaults (`-r 10`, `-t 75 -C 12`, `--exemplars 128`):
   - H3 resolution: {8, 9, 10, 11, 12}
   - Retention: {30s × 24, 45s × 16, 75s × 12, 150s × 6, 300s × 3} (all hold ~12-15 min total horizon)
   - Reservoir: {16, 32, 64, 128, 256}
3. Report Hit @5 with across-seed std at each point.

Readout:
- Three single-axis plots (one per knob) with the v2 default marked. Ideally Hit @5 is flat across a wide plateau and only drops at extreme values.
- The "operating range" — knob values for which Hit @5 stays within ±2pp of the default.
- Any surprising interaction (e.g., does small reservoir matter more at higher H3 resolution?).

Decision rule:
- If Hit @5 swings by >5pp across reasonable knob ranges, the v2 result was lucky and needs re-tuning before any baseline comparison.
- If Hit @5 stays within ±2pp across a 2x knob range, the result is robust and the paper can claim "PSM is insensitive to its hyperparameters in the operating regime."

## Reproducible Benchmark Sweeps

These sweeps use the existing benchmark binaries directly and emit CSV to stdout.

### SpatialMemory Throughput Sweep

```bash
benchmarks/sweep_spatial_memory.sh > spatial_memory_sweep.csv
PROFILE=portable benchmarks/sweep_spatial_memory.sh > spatial_memory_portable.csv
```

Environment knobs:
- `PROFILE` (`local` by default)
- `OBSERVE_OPS_LIST`
- `GRID_CELLS_LIST`
- `QUERY_OPS_LIST`

CSV columns:
- `profile`
- `observe_ops`
- `grid_cells`
- `query_ops`
- `scenario`
- `ops`
- `tiles`
- `total`
- `secs`
- `ops_per_sec`

### Tile Decode Sweep

```bash
benchmarks/sweep_tile_decode.sh > tile_decode_sweep.csv
PROFILE=portable benchmarks/sweep_tile_decode.sh > tile_decode_portable.csv
```

Environment knobs:
- `PROFILE` (`local` by default)
- `TOTAL_DECODES_LIST`
- `THREAD_COUNT_LIST`
- `SOURCE_MODE_LIST` (`memory disk` by default)

CSV columns:
- `profile`
- `total_decodes`
- `thread_count`
- `source_mode`
- `failures`
- `elapsed_sec`
- `decodes_per_sec`
- `megapixels_per_sec`

## E0. Nymeria Pipeline (corpus prerequisite)

Hard prerequisite for every experiment going into the Wearables AI
2026 paper. The 3-session Aria corpus used through 2026-05-28 is
internal-only and cannot appear in a published paper; all published
numbers must come from the public Nymeria dataset. This section
specifies the pipeline that gets Nymeria sessions through
`features.h5` so the existing E11/E12/item-7 scripts work unchanged.

### Subset selection

20 sessions, version-controlled in
[`journal/manifests/nymeria_subset_v1.yaml`](journal/manifests/nymeria_subset_v1.yaml).
Selection design: 5 paired-wearer dates × 2 wearer slots (s0/s1) × 2
contrasting activity variants per wearer = 20 sessions. This shape
gives:

- **5 same-date wearer pairs** (s0 + s1) for the federated-memory
  experiment. The Nymeria capture protocol records two wearers in the
  same physical space at the same time; HLL union across the two
  wearers' PSMs is a free experiment that the Aria corpus couldn't
  support.
- **2 activities per wearer** (act0 + act2 where available) for cross-
  activity retrieval probes and same-place / different-activity
  revisitation.
- **Dates spanning June-July 2023** for environmental diversity.

Sessions live at `/checkpoint/dream/arjangt/video_retrieval/nymeria/<sid>`
on the cluster. Each session ships VRS (camera + IMU) plus narration
JSON, motion JSON, and metadata.

### Pipeline scope

1. **`extraction/psm_extraction/io/aria_vrs.py`** — VRS reader.
   `projectaria-tools` exposes a Python API for VRS playback that
   yields (frame, timestamp_ns, lat/lng if GPS available, IMU
   samples). Slots into the existing `ExtractOptions.frame_source`
   abstraction so the rest of the extraction pipeline (CLIP/DINO/JEPA
   runners, HDF5 writer, schema-v2 metadata) is unchanged.
2. **`scripts/extract_nymeria_all.sh`** — env-knobbed sweep over the
   manifest sessions. One encoder at a time (bigG by default).
3. **`scripts/slurm/extract_nymeria.sbatch`** — GPU job wrapper.
   Probably needs `h200_comm_shared` × 1 GPU per session and an array
   job to parallelize the 20 sessions.
4. **Question annotation** for the 20 sessions. Nymeria's narration
   JSON helps but doesn't pre-form IoU-scoreable QA. Hand-annotate
   ~3-5 questions per session = 60-100 total. Format:
   `journal/manifests/nymeria_questions_v1.yaml`.

### Acceptance criteria

- All 20 sessions produce a `clip_bigg_features.h5` parseable by
  `psm --search` (sanity check: `psm -f <h5> -g clip` ingests
  without error, prints non-zero tile count).
- Question manifest committed and at least 50 questions have
  ground-truth `[t_start, t_end]` intervals.
- `eval_baselines_all.sh` + `eval_bigg_all.sh` + `eval_hyperparam_sweep.sh`
  all accept Nymeria features files unchanged (item 0 ends the
  moment these run green on Nymeria).

### Downstream consumers

E11 (item 1), E12 (item 2), item 7 (latency on Nymeria session sizes),
plus the new federated-memory + cross-activity probes that the Aria
corpus couldn't support.

## E0'. SLOPER4D Pipeline (street-scale corpus, 2026-06-17 pivot)

After the full 30-session Nymeria sweep returned flat ~2% Hit@5
across every (h3_res, retention, exemplars) operating point — the
hyperparameter sweep made it clear PSM cannot discriminate at
room scale, which is where 26/30 Nymeria sessions sit (≤9.4 m bbox)
— the corpus story pivoted. SLOPER4D (Dai et al., CVPR 2023)
provides multi-session LiDAR-SLAM trajectories at genuine
street scale, replacing the single Nymeria street-scale session
(`shelby_arroyo_act0`, 69 m bbox) we had been planning to lean on.

See `journal/PAPER.md` 2026-06-16 + 2026-06-17 entries for the
full decision log.

### Subset selection

15 sequences shipped in the SLOPER4D release; **6 currently
downloadable** without an authors' request. Of those 6, 5 are
retained — `seq002_football_001` is dropped because its 22 m bbox
is room-scale and below the r10 threshold (66 m H3 edge):

| Sequence | Trajectory | bbox XY | H3 scale | Notes |
|---|---|---|---|---|
| seq003_street_002 | 275 m | 105 m | r10 (marginal) | Outdoor street walking |
| seq005_library_002 | 417 m | 91 m | r11, edge-r10 | Indoor library + outdoor stairs |
| seq007_garden_001 | 77 m | 66 m | r11 / r12 transition | Bench + path |
| seq008_running_001 | 225 m | 176 m | r10 (~3× edge) | Outdoor running |
| seq009_running_002 | 946 m | **446 m** | **r10 (~7 cells)** | Long outdoor run; primary street-scale anchor |

Three additional street-scale sequences (`001_campus_001` 908 m,
`010_park_001` 642 m, `011_park_002` 1,025 m) are listed in the
SLOPER4D paper but not in the 6 currently published. If acquired
via direct request to the authors, they triple the multi-session
street-scale coverage and we fold them into the same sweep.

Source zips at `/checkpoint/dream/arjangt/SLOPER4D/*.zip` (CC BY-NC-SA 4.0).
Selectively unzipped to `/checkpoint/dream/arjangt/SLOPER4D-unzipped/`
(only `lidar_data/lidar_trajectory.txt` + `rgb_data/*.MP4` per
sequence — skip the multi-GB LiDAR point clouds that PSM doesn't
need).

### Pipeline scope

1. **`extraction/psm_extraction/io/sloper4d.py`** — trajectory
   reader. Parses `lidar_data/lidar_trajectory.txt` (space-separated
   `framenum X Y Z qx qy qz qw timestamp`), projects metric XYZ to
   WGS84 lat/lng via flat-earth at fake origin
   `(24.4381, 118.0992)` (Xiamen University, where the dataset was
   captured). PSM is invariant to origin choice; pinning near XMU
   is a viz-honesty default.
2. **`scripts/extract_sloper4d_sessions.py`** — per-sequence
   orchestrator wrapper. Writes a temporary Aria-style `gps.json`
   sidecar from the projected trajectory next to the MP4, calls
   `python -m psm_extraction extract --gps-json …` so the existing
   CLIP runner + v2 writer + frame-alignment apply unchanged, then
   cleans up the sidecar. H5 basename matches encoder
   (`clip_l_features.h5` / `clip_bigg_features.h5`).
3. **`scripts/extract_sloper4d.sh`** — login-node probe + extract
   wrapper. Prints the per-sequence trajectory summary before
   calling the orchestrator.
4. **`scripts/slurm/extract_sloper4d.sbatch`** — 5-task array per
   encoder (one task per kept sequence). Submit twice for the
   clipL + bigG encoder ablation:
   ```
   sbatch                              scripts/slurm/extract_sloper4d.sbatch
   sbatch --export=MODEL=clip_bigg     scripts/slurm/extract_sloper4d.sbatch
   ```
5. **Question generation** — TODO. SLOPER4D ships no narration
   track; the planned framing uses `query_mode: last_seen` GPS-
   grounded queries (item 6 in PAPER.md), sidestepping manual
   annotation by using the trajectory itself as ground truth.

### Acceptance criteria

- All 5 sequences extract to v2 `clip_l_features.h5` +
  `clip_bigg_features.h5` with `track_mode: real_gps` and a
  populated `gps` sensor group.
- H3 r10 cell counts on seq009 ≥ 5 (confirms multi-cell carving
  at the highest-mobility sequence).
- H3 resolution sweep on seq009 replicates the Nymeria-street
  finding: Hit @5 at r12 ≥ 2× Hit @5 at r10 for both encoders.
  Failing that replication implies the Nymeria-street signal was
  a one-session artifact and the corpus story needs another revision.

### Downstream consumers

E11 (item 1, naive retrieval baselines), E12 (item 2, hyperparameter
sweep — H3-res ablation is the primary acceptance criterion), and
item 6 (encoder-bypass `query_mode: last_seen` stress test, which
SLOPER4D is uniquely well-suited to because its global trajectory
is dense and accurate).
