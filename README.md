# Probabilistic Spatial Memory

A bounded-memory, time-decayed spatial memory system built on probabilistic data structures. Models *what has been seen*, *where it was seen*, and *how memory fades over time* — from egocentric video captured on [Project Aria](https://www.projectaria.com/) glasses.

## What it does

Given a stream of observations tied to locations and timestamps, the system maintains compact approximate summaries per geographic region. Each region tracks distinct items seen over sliding time windows using [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog) counters arranged in a ring buffer.

```
(timestamp, lat, lon, observation) → spatial tile → ring buffer of HLLs
```

This enables queries like:
- "How many distinct things were seen in this area in the last N intervals?"
- "Is this region becoming more or less novel over time?"

Memory usage is bounded regardless of how many observations are processed.

## Data: Project Aria

Input sessions are recorded on [Meta's Project Aria](https://www.projectaria.com/) glasses, which capture synchronized:

- **Egocentric video** — first-person perspective from the wearer's point of view
- **IMU** — 100 Hz accelerometer + gyroscope (6-axis inertial measurement)
- **GPS** — 1 Hz location fixes

A Python extraction pipeline (offline) runs each session through two vision foundation models and writes the results to an HDF5 file alongside the raw sensor streams:

| Model | Group | What it produces | Colormap |
|-------|-------|-----------------|----------|
| [DINOv2](https://arxiv.org/abs/2304.07193) | `dino` | 1024-d CLS embeddings + **CLS→patch attention maps** (14x14). Attention highlights *where the model is looking* — salient objects, textures, scene structure. | Inferno (black → red → yellow) |
| [V-JEPA 2](https://arxiv.org/abs/2412.08974) | `jepa` | 1024-d mean-pooled encoder tokens + **spatial prediction error maps** (16x16). Prediction error highlights *surprise* — regions the model fails to predict from context, indicating novelty or unusual content. | Viridis (purple → teal → yellow) |

The CLS embeddings are hashed into the spatial memory engine (HyperLogLog counters per H3 hex cell). The spatial maps are rendered as semi-transparent heatmap overlays on the video.

## IMU visualization

The high-rate IMU stream drives three visual features on the map:

- **GPS trace ribbon** — color-coded by motion state: blue (stationary), green (walking), orange (running). Motion is classified from accelerometer magnitude deviation from gravity.
- **Heading** — integrated from gyroscope yaw rate (projected onto the gravity vector). Displayed as a camera frustum at the current position.
- **Pitch-dependent frustum** — the frustum shape changes based on phone tilt. Looking forward: long, narrow (far field of view). Looking down: short, wide (near ground). Derived from the smoothed gravity vector.

Dead reckoning (heading + estimated speed) is blended with GPS via a complementary filter to produce smooth inter-GPS-sample positioning.

## Architecture

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

- **Tile**: A geographic region (H3 hex cell) with its own ring buffer
- **Ring Buffer**: Fixed-size circular buffer of HLL counters, one per time window
- **HLL**: HyperLogLog sketch estimating distinct item count

Merging HLL slots gives "memory over the last N intervals" with natural time decay — oldest slots get overwritten as the buffer advances.

## Building

```bash
make          # build library and CLI → targets/psm
make viz      # build visualizer → targets/psm-viz
make bench-spatial-memory  # run a lightweight SpatialMemory throughput benchmark
make test     # build and run tests
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
targets/psm-viz -v video.mp4 -f features.h5 -g dino

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
| H | Toggle live debug title HUD |
| Q / Esc | Quit |

The visualizer opens paused by default on the first decoded frame, shows a centered startup play overlay, and does not begin playback until you press `Space`.

The debug HUD lives in the window title and shows playback/decode budgets, ingest drain activity, tile pipeline queue counts, and tile disk-cache health in real time. For the `v`, `in`, `imu`, `gps`, and `up` fields, the HUD shows `work/current_budget`; when adaptive backpressure boosts a budget above its configured base, the base appears in parentheses, for example `256/384(128)`. A trailing `*` means that lane still had backlog after spending its frame budget. Tile fields are: `act` active network downloads, `rdy` compressed tiles ready for decode, `dec` tiles currently being decoded, `pix` decoded tiles waiting for GL upload, and `c` resident cached tile textures. Disk-cache fields are: `h` disk cache hits, `w` cache writes, `p` pruned files, and `m` cached MiB used versus cap.

**Layout:** Left half shows video with optional attention/prediction heatmap overlay. Right half shows configurable raster tiles (default: `CartoDB.Positron`) with H3 hex heatmap (viridis), GPS trace ribbon, and camera frustum. The map view follows the latest GPS/IMU-driven position smoothly by default; manual drag temporarily overrides that view until you re-center with `C`.

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
    gps_trace.h
    imu_processor.h
    viz_config.h
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
    tile_map.c
    gps_trace.c
    imu_processor.c
    viz_config.c
psm-viz.toml.example  # Sample visualizer config
configs/            # Ready-to-use psm-viz tuning presets
benchmarks/         # Lightweight performance benchmarks
shaders/            # GLSL shaders
tests/              # Test suites
  test_ring_buffer.c
  test_tile.c
  test_spatial_memory.c
  test_ingest.c
  test_jepa_cache.c
  test_viz_math.c
  test_viz_config.c
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

## Status

Work in progress. See [JOURNAL.md](JOURNAL.md) for development notes and detailed progress.
