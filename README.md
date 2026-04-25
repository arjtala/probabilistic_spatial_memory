# Probabilistic Spatial Memory

A bounded-memory, time-decayed spatial memory system built on probabilistic data structures. Models *what has been seen*, *where it was seen*, and *how memory fades over time* — from egocentric video captured on [Project Aria](https://www.projectaria.com/) glasses.

## TL;DR

`psm` turns timestamped egocentric video features into a bounded spatial memory: embeddings are hashed into H3 cells, each cell keeps a sliding ring buffer of HyperLogLog sketches, and the visualizer replays the session as synchronized video plus map.

In the map view, brighter/yellower hexes indicate cells with higher distinct-observation counts relative to the hottest cell in the current scene. Older memory does not disappear continuously; it decays as time buckets roll over, and cells with history but little current activity fade by becoming more transparent.

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
targets/psm -f clip_features.h5 -g clip --similar-to query.f32 --top 5 --exemplars 8 -j

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
| `--similar-to` | `<path>` | — | Query by embedding similarity from a raw `float32` LE vector |
| `--center` | `LAT,LNG` | — | Restrict `--similar-to` to a centered search neighborhood |
| `--k-ring` | `<N>` | `0` | H3 neighborhood radius for `--last-seen` or centered `--similar-to` |
| `--top` | `<N>` | `5` | Maximum query hits to print |
| `--exemplars` | `<N>` | `8*` | Per-tile exemplar reservoir; auto-set to `8` with `--similar-to` |

> **Retention:** each tile remembers observations for `capacity × time_window_sec` (default 60s). Query output is empty past that horizon — widen `-C` or `-t` for longer sessions.

`--similar-to` expects a binary file containing a flat `float32` little-endian vector in the same embedding space as the HDF5 group's `embeddings` dataset. With `-j`, similarity results include `similarity`, `exemplar_t`, and the retained `t_min`/`t_max` interval for each matching tile.

## E5 Demo

`scripts/e5_clip_demo.py` is a minimal plain-video harness for the E5 text-query path. It samples frames with `ffmpeg`, embeds the frames and the text query with CLIP, writes a native `clip` HDF5 group plus the raw `float32` LE query file, then calls `targets/psm --similar-to -j` and prints the retrieved intervals.

When a session HDF5 with a per-frame `dino` / `jepa` / `gps` group sits next to the video (or is passed via `--gps-source PATH`), the script interpolates real GPS onto the CLIP frame timestamps so retrieved cells are real H3 cells around the captured route. With no GPS available — or if you pass `--no-gps` — frames lay onto a synthetic H3 snake-grid so each fixed-duration segment lands in its own pseudo-cell.

```bash
# Plain video, no GPS — synthetic snake-grid
conda activate playground
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

`psm-viz` renders side-by-side video playback and a spatial memory heatmap with GPS trace overlay, synchronized by timestamp.

```bash
# Config file (defaults < config < CLI)
targets/psm-viz -c psm-viz.toml.example
targets/psm-viz -c /path/to/psm-viz.toml -g jepa
targets/psm-viz -c configs/psm-viz-balanced.toml -d /path/to/session
targets/psm-viz -c configs/psm-viz-low-hitch.toml -d /path/to/session

# Directory mode — finds *.mp4 and features.h5 automatically
targets/psm-viz -d /path/to/session/
targets/psm-viz -d /path/to/session/ -g jepa

# Explicit flags
targets/psm-viz -v video.mp4 -f features.h5 -g dino -m total

# Legacy positional args
targets/psm-viz video.mp4 features.h5 dino 5.0 10
```

| Flag | Arg | Default | Description |
|------|-----|---------|-------------|
| `-c` | `<path>` | — | TOML-style config file |
| `-d` | `<dir>` | — | Directory containing `*.mp4` and `features.h5` |
| `-v` | `<path>` | — | Video file path |
| `-f` | `<path>` | — | HDF5 features file path |
| `-g` | `<name>` | `dino` | HDF5 group name (`dino` or `jepa`) |
| `-t` | `<sec>` | `5.0` | Time window (seconds) |
| `-r` | `<res>` | `10` | H3 resolution (0-15) |
| `-m` | `<mode>` | `total` | Heatmap mode (`total`, `current`, `recency`) |
| `-h` | — | — | Print help |

`psm-viz.toml` supports:

```toml
session_dir = "./session"
# video_path = "./session/video.mp4"
# features_path = "./session/features.h5"

group = "dino"
time_window_sec = 5.0
h3_resolution = 10
start_paused = true
debug_hud_enabled = true
scrub_sensitivity_sec = 2.0
map_follow_smoothing = 8.0
video_decode_budget = 6
ingest_record_budget = 128
imu_sample_budget = 512
gps_point_budget = 64
tile_uploads_per_frame = 1
tile_disk_cache_enabled = true
tile_disk_cache_max_mb = 512
heatmap_mode = "total"
tile_style = "CartoDB.Positron"

# Required for Stadia.* presets and any custom template using {api_key}
# tile_api_key = "..."

# Optional override; supports {z}, {x}, {y}, {s}, and {api_key}
# tile_url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
```

Relative paths in the config resolve relative to the config file itself. CLI flags override config values.

Ready-made presets:
- `configs/psm-viz-balanced.toml`: a good default balance between responsiveness and tile fill speed.
- `configs/psm-viz-low-hitch.toml`: prioritizes smoother interaction with fewer tile uploads per frame and smaller per-frame catch-up budgets.

Tuning keys:
- `start_paused`: when `true`, `psm-viz` opens on the first decoded frame and waits for `Space` before playback starts.
- `debug_hud_enabled`: enables the live window-title HUD by default. You can still toggle it at runtime with `H`.
- `scrub_sensitivity_sec`: seconds moved per horizontal scroll step on the video pane.
- `map_follow_smoothing`: exponential follow rate for GPS/IMU-driven recentering. Higher values snap faster.
- `video_decode_budget`: baseline video decode steps per frame at 1x playback. Faster playback scales up from this value, and sustained decode backlog can temporarily raise it further before it decays back down.
- `ingest_record_budget`: baseline max feature/embedding records applied per frame. Under sustained ingest backlog the runtime can temporarily raise this catch-up budget, then ease it back to the configured base.
- `imu_sample_budget`: baseline max IMU samples drained per frame, with the same temporary backlog-driven catch-up behavior.
- `gps_point_budget`: baseline max standalone GPS points drained per frame, with the same temporary backlog-driven catch-up behavior.
- `tile_uploads_per_frame`: baseline max ready tile textures uploaded per frame. Lower values reduce GL-side hitches; higher values fill tiles faster, and the runtime can temporarily raise this when decoded tiles are piling up.
- `tile_disk_cache_enabled`: enables or disables the on-disk raster tile cache.
- `tile_disk_cache_max_mb`: maximum on-disk tile cache size per configured tile source before older cached tiles are pruned.
- `heatmap_mode`: selects how H3 cells are scored before coloring. `total` shows the rolling merged count across the active ring buffer, `current` shows current-bucket activity only, and `recency` shows `current / total` to highlight cells that are active now relative to their longer-term history.

Available `tile_style` presets:
- `CartoDB.Positron`
- `CartoDB.PositronNoLabels`
- `CartoDB.Voyager`
- `CartoDB.DarkMatter`
- `Stadia.AlidadeSmooth` (`tile_api_key` required)
- `Stadia.AlidadeSmoothDark` (`tile_api_key` required)

Preview the providers here: <https://leaflet-extras.github.io/leaflet-providers/preview/>

Downloaded raster tiles are cached on disk and replay through the same threaded decode path as network tiles. The cache location is:
- macOS: `~/Library/Caches/psm-viz/tiles/...`
- other Unix-like systems: `$XDG_CACHE_HOME/psm-viz/tiles/...` or `~/.cache/psm-viz/tiles/...`

**Controls:**

| Key / Gesture | Action |
|---------------|--------|
| Space | Start / pause / resume playback (shows pause icon) |
| +/- | Zoom in / out around the map center |
| Left / Right | Slow down / speed up playback |
| Scroll H (video) | Scrub video timeline |
| Scroll V (map) | Zoom map toward the cursor |
| Drag (map) | Pan map manually |
| C | Re-center map and resume smooth follow |
| M | Cycle heatmap mode (`total` → `current` → `recency`) |
| L | Toggle the heatmap legend overlay |
| H | Toggle live debug title HUD |
| P | Save a screenshot of the current composed frame to `captures/` as `.png` |
| ? / F1 | Toggle the help overlay |
| Q / Esc | Quit |

The visualizer opens paused by default on the first decoded frame, shows the help/startup overlay immediately, and does not begin playback until you press `Space`. The first `Space` also dismisses that initial help panel.

Screenshots save into `<session_dir>/captures/` when a session directory is configured, otherwise `./captures/`, using names like `psm-viz-000042.png`.

The debug HUD lives in the window title and shows playback/decode budgets, ingest drain activity, tile pipeline queue counts, and tile disk-cache health in real time. For the `v`, `in`, `imu`, `gps`, and `up` fields, the HUD shows `work/current_budget`; when adaptive backpressure boosts a budget above its configured base, the base appears in parentheses, for example `256/384(128)`. A trailing `*` means that lane still had backlog after spending its frame budget. Tile fields are: `act` active network downloads, `rdy` compressed tiles ready for decode, `dec` tiles currently being decoded, `pix` decoded tiles waiting for GL upload, and `c` resident cached tile textures. Disk-cache fields are: `h` disk cache hits, `w` cache writes, `p` pruned files, and `m` cached MiB used versus cap.

**Layout:** Left half shows video with optional attention/prediction heatmap overlay. Right half shows configurable raster tiles (default: `CartoDB.Positron`) with H3 hex heatmap (viridis), GPS trace ribbon, and camera frustum. The map view follows the latest GPS/IMU-driven position smoothly by default; manual drag temporarily overrides that view until you re-center with `C`.

**Hex heatmap semantics:** The color ramp is always relative to the hottest tile currently rendered. Low-intensity tiles appear dark purple, mid-range tiles shift toward teal/cyan, and the hottest tiles appear yellow. Alpha also rises with intensity, so stronger cells look more solid.

Mode-specific behavior:
- `total`: colors by merged distinct-count across the full active ring-buffer horizon. This is the historical memory view.
- `current`: colors by the current time bucket only. This is the most immediate activity view.
- `recency`: colors by `current / total`, highlighting cells that are active now relative to their own accumulated history.

Time decay is only partly encoded in hue. In `total` mode, when a tile has historical count but little or no activity in the current bucket, the renderer reduces alpha so the cell lingers as a dim memory instead of vanishing immediately. The underlying forgetting is stepwise: as the ring buffer advances, the oldest bucket is overwritten, so hex intensity can drop in discrete steps at window boundaries rather than as a perfectly smooth fade.

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

For pure performance regressions, the reproducible benchmark sweep entry points are:

- `benchmarks/sweep_spatial_memory.sh`
- `benchmarks/sweep_tile_decode.sh`

## Status

The core engine, ingestion pipeline, tests, and interactive visualizer are implemented and documented here. The README is the primary project guide.
