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
tile_style = "CartoDB.Positron"

# Required for Stadia.* presets and any custom template using {api_key}
# tile_api_key = "..."

# Optional override; supports {z}, {x}, {y}, {s}, and {api_key}
# tile_url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
```

Relative paths in the config resolve relative to the config file itself. CLI flags override config values.

Available `tile_style` presets:
- `CartoDB.Positron`
- `CartoDB.PositronNoLabels`
- `CartoDB.Voyager`
- `CartoDB.DarkMatter`
- `Stadia.AlidadeSmooth` (`tile_api_key` required)
- `Stadia.AlidadeSmoothDark` (`tile_api_key` required)

Preview the providers here: <https://leaflet-extras.github.io/leaflet-providers/preview/>

**Controls:**

| Key / Gesture | Action |
|---------------|--------|
| Space | Pause / resume (shows pause icon) |
| +/- | Zoom in / out around the map center |
| Left / Right | Slow down / speed up playback |
| Scroll H (video) | Scrub video timeline |
| Scroll V (map) | Zoom map toward the cursor |
| Drag (map) | Pan map manually |
| C | Re-center map and resume smooth follow |
| Q / Esc | Quit |

**Layout:** Left half shows video with optional attention/prediction heatmap overlay. Right half shows configurable raster tiles (default: `CartoDB.Positron`) with H3 hex heatmap (viridis), GPS trace ribbon, and camera frustum. The map view follows the latest GPS/IMU-driven position smoothly by default; manual drag temporarily overrides that view until you re-center with `C`.

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
