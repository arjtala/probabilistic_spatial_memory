# Experiments

This file turns the README's open questions into explicit, repeatable experiments using the tooling that already exists in this repo.

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

A forthcoming NeurIPS 2026 streaming egocentric memory benchmark — referred to below as the **Localization Paradox benchmark** after its headline finding — evaluates multimodal LLMs on 20K+ "look-back" QA pairs over 613 hours of unscripted Ray-Ban Meta egocentric video. Frontier MLLMs (Gemini 3 Pro, GPT-5, InternVL 3.5) achieve 27-50% semantic accuracy but near-zero temporal grounding (`mIoU` often ≈0). The paper's discussion section explicitly calls for "adaptive temporal indexing… and hierarchical memory buffers that can compress hours of video without losing the granular detail of momentary 'needle' events." PSM's H3-indexed ring buffer of HLL sketches is a candidate substrate for exactly that architectural gap.

Prerequisites for E5-E7 are tracked in `TODO.md` under the "Localization Paradox Alignment" section:

- Per-bucket `(t_min, t_max)` retention, so a query can emit `[t_start, t_end]` intervals.
- Per-tile exemplar embedding sampler (reservoir), so k-NN retrieval against past observations is possible.
- `SpatialMemory_query_intervals(...)` API surface.

Two further pipeline components are external to this repo and are described per-experiment below rather than tracked as TODOs:

- **Text-query adapter** (E5, E7): embed a benchmark question via a vision-language text tower (CLIP / SigLIP / DINO) and match against per-tile exemplar embeddings to produce the semantic cue. Sits outside `libpsm`.
- **Grounded response stage** (E5): forward PSM's top-k `(cell, t_start, t_end)` intervals plus the corresponding evidence frames to an MLLM as explicit grounding context. This is an MLLM API call, not engine code.

### E5. PSM as Retrieval Prefilter for MLLM Temporal Grounding

Question:
- Does PSM's spatial + temporal prior materially close the Localization Paradox gap on frontier MLLMs?

Current capability:
- `SpatialMemory_query_similar(...)` / `psm --search` provide the retrieval backend for the text-query adapter.
- `scripts/e5_clip_demo.py` is a minimal local demo over a plain video using CLIP image/text embeddings and a synthetic track.
- The in-tree extraction pipeline (`python -m psm_extraction extract --models clip,dino[,jepa]`) reproduces the Aria pipeline's `features.h5` shape end-to-end on Apple Silicon — verified on a 15-min Fulham session with DINOv3 attention-distribution parity to the original. So a benchmark session that ships `data.mp4 + gps.json + imu.json + metadata.json` can be ingested entirely in-house, no external pipeline required.
- Full benchmark evaluation is still external to this repo: it needs a benchmark subset and one MLLM inference endpoint.

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
- Blocked on exemplar embedding retention (see TODO "Localization Paradox Alignment").
- Once unblocked, answers should run in `O(matching_tiles × capacity)` — typically sub-millisecond on session-scale memories.

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
