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
