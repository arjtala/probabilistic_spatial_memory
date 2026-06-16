# Probabilistic Spatial Memory

A bounded-memory, time-decayed spatial memory system built on probabilistic data structures. Models *what has been seen*, *where it was seen*, and *how memory fades over time* — from egocentric video captured on [Project Aria](https://www.projectaria.com/) glasses.

## TL;DR

`psm` turns timestamped egocentric video features into a bounded spatial memory: embeddings are hashed into H3 cells, each cell keeps a sliding ring buffer of HyperLogLog sketches, and the visualizer replays the session as synchronized video plus map.

In the map view, hexes are colored along an RGB-cube tour (near-black for sparse cells, climbing through red, yellow, green, cyan, blue, magenta, and ending at white for the hottest cell relative to the current scene). Optional 3D extrusion (`E` to toggle) raises a cell's height in proportion to its intensity so dominant memory cells read at a glance. Older memory does not disappear continuously; it decays as time buckets roll over, and cells with history but little current activity fade by becoming more transparent.

## Why

This project is a systems-oriented model of episodic spatial memory under hard storage limits. The goal is not to predict behavior or reconstruct every past frame, but to maintain a compact representation of:

- what was seen
- where it was seen
- how that memory fades over time

The interesting constraint is that memory stays bounded even as session length grows. Instead of storing raw frames or full embeddings indefinitely, the system keeps approximate, mergeable summaries that can still answer useful questions about familiarity, novelty, and revisitation.

## Design goals

- **Bounded memory:** memory use should remain stable as more observations arrive.
- **Time decay:** recent experience should matter more than old experience without requiring explicit garbage collection of individual events.
- **Spatial locality:** observations should be attached to geographic regions so queries stay local and map rendering is natural.
- **Fast aggregation:** summaries should be cheap to merge across time windows for interactive exploration.
- **Inspectable behavior:** the visualization should make the memory model legible rather than hiding it behind offline metrics only.

## Core idea

Given a stream of observations tied to locations and timestamps, the system maintains compact approximate summaries per geographic region. Each region tracks distinct observations seen over sliding time windows using [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog) counters arranged in a ring buffer.

```
(timestamp, lat, lon, observation) -> spatial tile -> ring buffer of HLLs
```

This enables queries like:
- "How many distinct things were seen in this area in the last N intervals?"
- "Is this region becoming more or less novel over time?"

Memory usage is bounded regardless of how many observations are processed.

## What is being counted?

The current implementation hashes frame-level model embeddings into the spatial memory engine, so the primary signal is an approximate count of distinct semantic observations per region and time window.

That is closer to "unique visual or semantic states" than to raw frame count. In practice, the counting layer is generic enough to support several granularities:

- **Frame-level embeddings (current):** count distinct whole-scene observations.
- **Object-level embeddings:** count distinct objects encountered in space.
- **Clustered or LSH-stabilized embeddings:** count distinct semantic categories with more robustness to viewpoint or lighting changes.

This separation matters because the storage layer only needs stable hashed tokens; it does not need to know whether those tokens came from whole frames, objects, or semantic clusters.

## End-to-end pipeline

1. **Offline extraction:** a Project Aria session is processed into synchronized video, GPS, IMU, embeddings, and spatial heatmap outputs stored in HDF5.
2. **C ingestion:** the reader streams timestamped records, hashes embeddings, maps GPS fixes into H3 cells, and advances time windows from record timestamps rather than wall clock.
3. **Spatial memory:** each H3 cell owns a fixed-size ring buffer of HLL sketches, giving sliding-window distinct counts with natural forgetting.
4. **Interactive visualization:** the OpenGL visualizer replays the session as side-by-side video and map views, with model overlays on the video and memory intensity rendered over the map.

## Data: Project Aria

Input sessions are recorded on [Meta's Project Aria](https://www.projectaria.com/) glasses, which capture synchronized:

- **Egocentric video** — first-person perspective from the wearer's point of view
- **IMU** — 100 Hz accelerometer + gyroscope (6-axis inertial measurement)
- **GPS** — 1 Hz location fixes

A Python extraction pipeline (offline) runs each session through two vision foundation models and writes the results to an HDF5 file alongside the raw sensor streams:

| Model | Group | What it produces | Colormap |
|-------|-------|-----------------|----------|
| [DINOv3](https://github.com/facebookresearch/dinov3) | `dino` | 1024-d CLS embeddings + **CLS→patch attention maps** (14x14). Attention highlights *where the model is looking* — salient objects, textures, scene structure. | Inferno (black → red → yellow) |
| [V-JEPA 2](https://arxiv.org/abs/2412.08974) | `jepa` | 1024-d mean-pooled encoder tokens + **spatial prediction error maps** (16x16). Prediction error highlights *surprise* — regions the model fails to predict from context, indicating novelty or unusual content. | Viridis (purple → teal → yellow) |

The CLS embeddings are hashed into the spatial memory engine (HyperLogLog counters per H3 hex cell). The spatial maps are rendered as semi-transparent heatmap overlays on the video.

## HDF5 feature schema

PSM consumes a `features.h5` file containing per-frame embeddings, plus optional sensor streams. The C ingest only reads dataset arrays — attrs are advisory metadata for downstream tooling.

Schema **v2** (current, set by `psm-extraction`) makes the file self-documenting:

- **Root attrs**: `schema_version=2`, `producer="psm-extraction"`, `producer_version`, `created_at_utc`, `timestamp_unit="unix_seconds_f64"`, `coord_system="WGS84_degrees"`, optional `source_video` and `session_id`.
- **Sensor groups** (canonical raw sources of truth):
  - `gps/{timestamps, lat, lng}` — float64; optional `@rate_hz_nominal` attr.
  - `imu/{timestamps, accel, gyro}` — accel/gyro as `(N, 3)` float32; optional `@rate_hz_nominal`.
- **Model groups** (`dino`, `jepa`, `clip`, …) — per-frame data plus interpolated sensors:
  - Required datasets: `timestamps` (float64), `lat`/`lng` (float64), `embeddings` (`(N, D)` float32).
  - Optional: `attention_maps`/`prediction_maps` (`(N, h, w)` float32), `accel`/`gyro` (`(N, 3)` float32, interpolated from the canonical `imu` group).
  - Required attrs: `model`, `checkpoint`, `embedding_dim`, `sample_fps`, `sampling`, `preprocess`, `normalized`.
  - Optional attrs: `patch_grid` (`[h, w]`), `interpolation` (string describing how lat/lng/imu were resampled).

Schema **v1** is the format the existing external pipeline produces (data-only, no attrs). PSM reads both transparently; v1 → v2 is a non-destructive in-place migration:

```bash
# Auto-detect missing attrs and back-fill best-effort defaults for known
# groups (dino/jepa/clip). Idempotent and pure metadata: never modifies datasets.
python -m psm_extraction migrate /tmp/hdd/<session>/features.h5 \
  --source-video /tmp/hdd/<session>/data.mp4 \
  --session-id <session>
```

The extraction pipeline producing v2 files lives in `extraction/`. Optional extras (listed in `extraction/pyproject.toml`):

| Extra | Pulls in | Needed for |
|---|---|---|
| `dev` | `pytest` | running the extraction test suite |
| `clip` | `torch`, `transformers`, `Pillow` | CLIP runner — text/image embeddings, the `--search` query path |
| `mlx` | `mlx` | Apple Silicon MLX runners (currently stubbed; Phase 2 follow-up) |
| `aria` | `projectaria-tools` | raw VRS reading (Phase 3 follow-up; sidecar JSON works without it) |
| `viz` | `matplotlib`, `umap-learn`, `h3` | `scripts/embedding_atlas.py` (paper-figure helper) |
| `all` | union of the above | one-shot dev install |

Typical setup for the full demo: `pip install -e extraction[clip,viz]`.

See `TODO.md` → "Extraction Pipeline" for the phased roadmap (Phase 1 ships the schema + writer + migration; later phases add CLIP, DINOv3, V-JEPA 2, and Aria VRS readers).

## IMU visualization

The high-rate IMU stream drives three visual features on the map:

- **GPS trace ribbon** — color-coded by motion state: blue (stationary), green (walking), orange (running). Motion is classified from accelerometer magnitude deviation from gravity.
- **Heading** — integrated from gyroscope yaw rate (projected onto the gravity vector). Displayed as a camera frustum at the current position.
- **Pitch-dependent frustum** — the frustum shape changes based on phone tilt. Looking forward: long, narrow (far field of view). Looking down: short, wide (near ground). Derived from the smoothed gravity vector.

Dead reckoning (heading + estimated speed) is blended with GPS via a complementary filter to produce smooth inter-GPS-sample positioning.

## Memory layout

```
┌─────────────────────────────────────────┐
│             Spatial Memory              │
│  ┌──────┐  ┌──────┐  ┌──────┐           │
│  │Tile A│  │Tile B│  │Tile C│  ...      │
│  └──┬───┘  └──┬───┘  └──┬───┘           │
│     │         │         │               │
│  ┌──▼───────────────────▼──────────┐    │
│  │  Ring Buffer (per tile)         │    │
│  │  [HLL][HLL][HLL]...[HLL]        │    │
│  │   t-0  t-1  t-2      t-n        │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

- **Tile**: a geographic region (H3 hex cell) with its own ring buffer
- **Ring buffer**: fixed-size circular buffer of HLL counters, one per time window
- **HLL**: HyperLogLog sketch estimating distinct item count

Merging HLL slots gives "memory over the last N intervals" with natural time decay — oldest slots get overwritten as the buffer advances.

The **effective retention window** per tile is `capacity × time_window_sec`. With the defaults (`-C 12 -t 5.0`) that is **60 seconds** — observations older than that age out of each tile's ring buffer and stop contributing to `-j` / `--last-seen` output. For multi-minute sessions, widen either knob (e.g. `-C 30 -t 60` = 30-minute window) before you start worrying about empty query results.

## Building

```bash
make          # build library and CLI → targets/psm
make viz      # build visualizer → targets/psm-viz
make debug    # debug-profile build → targets/debug/...
make portable # portable optimized build → targets/portable/...
make sanitize # ASan/UBSan build → targets/sanitize/...
make bench-spatial-memory  # run a lightweight SpatialMemory throughput benchmark
make test     # build and run tests
make test-portable         # portable-profile test suite
make test-sanitize         # sanitizer-backed test suite
make clean    # remove build artifacts and targets/
```

Requires `clang` and a Unix-like environment. Dependencies can come from Homebrew, your system package manager, or custom install prefixes. Homebrew example:

```bash
brew install h3 hdf5                  # core engine
brew install glfw ffmpeg curl          # visualization (psm-viz)
```

If your libraries live outside standard system paths, point `make` at them directly:

```bash
make H3_PREFIX=/opt/h3 HDF5_PREFIX=/opt/hdf5 \
     GLFW_PREFIX=/opt/glfw FFMPEG_PREFIX=/opt/ffmpeg
```

Build profiles:
- `PROFILE=local` (default): fast native build with `-march=native`, `-mtune=native`, `-ffast-math`, and LTO.
- `PROFILE=portable`: optimized build without host-specific CPU assumptions.
- `PROFILE=debug`: `-O0 -g3` for debugging.
- `PROFILE=sanitize`: AddressSanitizer + UndefinedBehaviorSanitizer with frame pointers.

Non-default profiles write outputs under `build/<profile>/` and `targets/<profile>/` so local optimized artifacts stay separate from debug and sanitizer builds.

## CLI

`psm` ingests one HDF5 group into the spatial memory engine and prints either a human-readable summary or JSON.

```bash
# Human-readable summary
targets/psm -f features.h5 -g dino -t 5.0 -r 10

# Override ring-buffer configuration
targets/psm -f features.h5 -g jepa -C 24 -p 12

# JSON output for scripting
targets/psm -f features.h5 -g dino -j

# Last-seen interval query around a coordinate
targets/psm -f features.h5 -g dino --last-seen 37.484000,-122.148000 --k-ring 1 --top 5 -j

# Similarity retrieval from a raw float32 LE query vector
targets/psm -f clip_features.h5 -g clip --search query.f32 --top 5 --exemplars 8 -j

# Legacy positional args still work
targets/psm features.h5 dino 5.0 10 12 10
```

| Flag | Arg | Default | Description |
|------|-----|---------|-------------|
| `-f` | `<path>` | — | HDF5 feature file |
| `-g` | `<name>` | `dino` | HDF5 group name |
| `-t` | `<sec>` | `5.0` | Time window in seconds |
| `-r` | `<res>` | `10` | H3 resolution (0-15) |
| `-C` | `<count>` | `12` | Ring-buffer capacity |
| `-p` | `<bits>` | `10` | HLL precision |
| `-j` | — | — | Emit JSON summary instead of text |
| `-h` | — | — | Print help |
| `--last-seen` | `LAT,LNG` | — | Return top interval hits around a coordinate |
| `--search` | `<path>` | — | Query by embedding similarity from a raw `float32` LE vector |
| `--center` | `LAT,LNG` | — | Restrict `--search` to a centered search neighborhood |
| `--k-ring` | `<N>` | `0` | H3 neighborhood radius for `--last-seen` or centered `--search` |
| `--top` | `<N>` | `5` | Maximum query hits to print |
| `--exemplars` | `<N>` | `8*` | Per-tile exemplar reservoir; auto-set to `8` with `--search` |
| `--seed` | `<N>` | — | Reservoir-sampler PRNG seed (uint64). Same seed + same input ⇒ bit-identical exemplar selections. Without it, the sampler self-seeds entropically on first use. |

> **Retention:** each tile remembers observations for `capacity × time_window_sec` (default 60s). Query output is empty past that horizon — widen `-C` or `-t` for longer sessions.

`--search` expects a binary file containing a flat `float32` little-endian vector in the same embedding space as the HDF5 group's `embeddings` dataset. With `-j`, each result includes `cell` (H3 hex string), `lat`/`lng` (cell center, degrees), `similarity` (cosine), `exemplar_t` (timestamp of the matching exemplar), the retained `t_min`/`t_max` interval, and the HLL `count` over the active retention horizon. `--last-seen` results carry the same fields minus `similarity` and `exemplar_t`.

## E5 Demo

`scripts/e5_clip_demo.py` is a minimal plain-video harness for the E5 text-query path. It samples frames with `ffmpeg`, embeds the frames and the text query with CLIP, writes a native `clip` HDF5 group plus the raw `float32` LE query file, then calls `targets/psm --search -j` and prints the retrieved intervals.

When a session HDF5 with a per-frame `dino` / `jepa` / `gps` group sits next to the video (or is passed via `--gps-source PATH`), the script interpolates real GPS onto the CLIP frame timestamps so retrieved cells are real H3 cells around the captured route. With no GPS available — or if you pass `--no-gps` — frames lay onto a synthetic H3 snake-grid so each fixed-duration segment lands in its own pseudo-cell.

The demo is a thin shim over `python -m psm_extraction extract`. For programmatic / batch use, prefer the package CLI directly:

```bash
# CLIP only
python -m psm_extraction extract \
  --video /path/to/video.mp4 \
  --output /path/to/clip_features.h5 \
  --models clip \
  --backend auto \
  --sample-fps 2 --segment-sec 1

# Reproduce the Aria pipeline shape: clip + dino + jepa groups in one v2 file,
# pulling gps.json + imu.json + metadata.json from <video_dir> automatically.
python -m psm_extraction extract \
  --video /path/to/data.mp4 \
  --output /path/to/features.h5 \
  --models clip,dino,jepa \
  --checkpoint dino:facebook/dinov3-vitl16-pretrain-lvd1689m \
  --sample-fps 30 --segment-sec 1 \
  --session-id <session>
```

### Long-run survival features

Long extractions (a 22-minute DINOv3 ViT-Large pass on 27 k frames is normal on M4 MPS) ship with three pieces of survival hygiene:

- **Stage banners + throttled progress** on stderr — every long run prints `[extract] / [frames] / [embed:<group>] / [write]` markers and a `[embed:<group>] N/total (X.X%)  rate it/s  elapsed=Ts  eta=Ts` line every ~2 s. JSON manifest still piped on stdout for `jq`.
- **Frame cache** — ffmpeg writes a `.extract_manifest.json` next to the JPEGs recording `(video, sample_fps, frame_count)`. Subsequent runs with matching params skip the ffmpeg step. Pass `--force-reextract` to wipe and re-run.
- **Per-model embedding cache** — after each runner finishes, embeddings (and any attention/prediction maps) save to a hashed `.npz` sidecar. Resuming after a kill picks up the cached results instead of re-running inference. Pass `--force-reembed` to ignore the cache; `--cache-dir <path>` overrides the default location (next to the output HDF5).

Cache key for the embedding sidecar = `model_id + checkpoint + video_path + sample_fps + group_name`, so different params can never reuse stale caches.

```bash
# Plain video, no GPS — synthetic snake-grid
# activate your Python environment with torch + transformers installed
python scripts/e5_clip_demo.py /path/to/video.mp4 "a person opening a refrigerator" \
  --output-dir /tmp/psm-e5

# Aria session with features.h5 next to the video — auto-detected real GPS
python scripts/e5_clip_demo.py /tmp/hdd/<session>/data.mp4 "a zebra crossing" \
  --output-dir /tmp/hdd/<session>/e5/zebra-crossing

# Explicit GPS source (any HDF5 with a dino/jepa/gps group)
python scripts/e5_clip_demo.py video.mp4 "a red bus" \
  --gps-source /path/to/features.h5
```

First run downloads the CLIP checkpoint. Artifacts include `clip_features.h5`, `query.f32`, `psm_results.json`, and a `manifest.json` recording the track mode (`real_gps` vs `synthetic_snake_grid`) plus the source group when GPS was used.

## Visualization

`psm-viz` renders side-by-side video playback and a spatial memory heatmap with GPS trace overlay, synchronized by timestamp. It supports configurable raster tile basemaps, 3D hex extrusion, DINOv3 attention and V-JEPA 2 prediction-error overlays, and interactive scrubbing/panning/zooming.

```bash
targets/psm-viz -c psm-viz.toml.example
targets/psm-viz -d /path/to/session/ -g jepa
```

See [docs/VISUALIZER.md](docs/VISUALIZER.md) for full configuration reference, controls, heatmap semantics, tile style presets, and tuning keys.

## Benchmarks

Use the lightweight throughput benchmark to track `SpatialMemory` regressions over time:

```bash
make bench-spatial-memory
./targets/benchmark_spatial_memory [observe_ops] [grid_cells] [query_ops]
```

The benchmark prints three scenarios:
- `observe_same_cell`: hot-path repeated observations into one cell.
- `observe_grid`: observations spread across a grid of cells.
- `query_grid`: repeated queries after pre-populating the grid.

For tile streaming regressions, use the standalone PNG decode stress benchmark that exercises the same `stbi_load_from_memory(...)` path as the threaded tile worker:

```bash
make bench-tile-decode
./targets/benchmark_tile_decode [total_decodes] [thread_count]
```

For reproducible multi-point sweeps over the existing benchmark binaries, use:

```bash
benchmarks/sweep_spatial_memory.sh > spatial_memory_sweep.csv
benchmarks/sweep_tile_decode.sh > tile_decode_sweep.csv
```

Both scripts emit CSV to stdout, accept `PROFILE=local|portable|debug|sanitize`, and allow their sweep grids to be overridden with environment variables. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full protocol.

## Project structure

```
include/
  core/             # Core engine headers
    ring_buffer.h
    tile.h
    spatial_memory.h
  ingest/           # Data ingestion headers
    ingest.h
  viz/              # Visualization headers
    shader.h
    video_decoder.h
    hex_renderer.h
    tile_map.h
    tile_disk_cache.h
    gps_trace.h
    imu_processor.h
    viz_config.h
    viz_runtime.h
src/
  core/             # Core engine
    ring_buffer.c
    tile.c
    spatial_memory.c
  ingest/           # HDF5 ingestion pipeline
    ingest.c
  viz/              # OpenGL visualizer
    viz_main.c
    shader.c
    video_decoder.c
    hex_renderer.c
    tile_disk_cache.c
    tile_map.c
    gps_trace.c
    imu_processor.c
    viz_config.c
    viz_runtime.c
psm-viz.toml.example  # Sample visualizer config
configs/            # Ready-to-use psm-viz tuning presets
benchmarks/         # Lightweight performance benchmarks
  sweep_spatial_memory.sh
  sweep_tile_decode.sh
shaders/            # GLSL shaders
tests/              # Test suites
  test_ring_buffer.c
  test_tile.c
  test_tile_table.c
  test_spatial_memory.c
  test_ingest.c
  test_jepa_cache.c
  test_viz_math.c
  test_viz_config.c
  test_viz_runtime.c
  test_tile_disk_cache.c
EXPERIMENTS.md      # Reproducible experiment protocols and sweep recipes
targets/            # Build outputs (psm, psm-viz, libpsm.a)
build/              # Intermediate object files
vendor/             # Git submodule
  probabilistic_data_structures/
    hyperloglog/    # HyperLogLog implementation
    bloom_filter/   # Bloom filter implementation
    lib/            # Shared utilities (hash functions, bit arrays)
```

## Dependencies

- [probabilistic_data_structures](https://github.com/arjtala/probabilistic_data_structures) — HyperLogLog, Bloom filter, hash functions (included as git submodule)
- [H3](https://h3geo.org/) — Uber's hexagonal hierarchical spatial index (`brew install h3`)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/) — Reading embedding datasets produced by Python extraction pipeline (`brew install hdf5`)

## Experiment backlog

The previous open questions are now mapped to explicit experiments in [EXPERIMENTS.md](EXPERIMENTS.md):

- `E1. Embedding-hash stability under semantic-preserving change`
  Compare matched baseline and perturbed feature exports with `targets/psm -j`, then check hotspot stability in `psm-viz`.
- `E2. Counting-unit ablation`
  Re-run the same session with frame-level, object-level, or clustered tokens once those exports exist, using the same `psm -j` harness for comparison.
- `E3. Familiarity convergence sensitivity sweep`
  Sweep `h3_resolution` and `time_window_sec` with `targets/psm -j` and record `tile_count`, total mass, current mass, and hottest-tile intensity.
- `E4. Novelty definition comparison`
  Compare `total`, `current`, and `current / total` from `psm -j`, then visually cross-check against the JEPA prediction-error overlay in `psm-viz`.

For the localization-paradox-aligned experiments (E5–E9), the first set of results lives in `journal/localization_paradox.md` (initial demo, 12 questions × 5 seeds across 3 sessions, 68% Hit@5) and `journal/localization_paradox2.md` (follow-up with encoder ablation, 12 new place-aware questions, encoder-bypass query mode — 83% Hit@5 on the full 22-question set, 100% on the place-aware subset).

For pure performance regressions, the reproducible benchmark sweep entry points are:

- `benchmarks/sweep_spatial_memory.sh`
- `benchmarks/sweep_tile_decode.sh`

## Status

The core engine, ingestion pipeline, tests, and interactive visualizer are implemented and documented here. The README is the primary project guide.
