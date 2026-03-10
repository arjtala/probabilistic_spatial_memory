# Probabilistic Spatial Memory

A bounded-memory, time-decayed spatial memory system built on probabilistic data structures. Models *what has been seen*, *where it was seen*, and *how memory fades over time*.

## What it does

Given a stream of observations tied to locations and timestamps, the system maintains compact approximate summaries per geographic region. Each region tracks distinct items seen over sliding time windows using [HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog) counters arranged in a ring buffer.

```
(timestamp, lat, lon, observation) → spatial tile → ring buffer of HLLs
```

This enables queries like:
- "How many distinct things were seen in this area in the last N intervals?"
- "Is this region becoming more or less novel over time?"

Memory usage is bounded regardless of how many observations are processed.

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
make          # build static library (libpsm.a) and CLI
make viz      # build visualizer (psm-viz)
make test     # build and run tests
make clean    # remove build artifacts
```

Requires `clang` and a Unix-like environment. Dependencies installed via Homebrew:

```bash
brew install h3 hdf5                  # core engine
brew install glfw ffmpeg curl          # visualization (psm-viz)
```

## Visualization

`psm-viz` renders side-by-side video playback and a spatial memory heatmap with GPS trace overlay, synchronized by timestamp.

```bash
# Directory mode — finds *.mp4 and features.h5 automatically
./psm-viz -d /path/to/session/
./psm-viz -d /path/to/session/ -g jepa

# Explicit flags
./psm-viz -v video.mp4 -f features.h5 -g dino

# Legacy positional args
./psm-viz video.mp4 features.h5 dino 5.0 10
```

| Flag | Arg | Default | Description |
|------|-----|---------|-------------|
| `-d` | `<dir>` | — | Directory containing `*.mp4` and `features.h5` |
| `-v` | `<path>` | — | Video file path |
| `-f` | `<path>` | — | HDF5 features file path |
| `-g` | `<name>` | `dino` | HDF5 group name |
| `-t` | `<sec>` | `5.0` | Time window (seconds) |
| `-r` | `<res>` | `10` | H3 resolution (0-15) |
| `-h` | — | — | Print help |

**Controls:** Space (pause), +/- (zoom), Left/Right (speed), Scroll (scrub video / zoom map), Q/Esc (quit).

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
shaders/            # GLSL shaders
tests/              # Test suites
  test_ring_buffer.c
  test_tile.c
  test_spatial_memory.c
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
