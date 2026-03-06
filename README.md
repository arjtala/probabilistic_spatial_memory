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
│  ┌──▼───────────────────▼─────────┐    │
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
make          # build static library (libpsm.a)
make test     # build and run tests
make clean    # remove build artifacts
```

Requires `clang` and a Unix-like environment.

## Project structure

```
include/            # Headers
  ring_buffer.h
  tile.h
  spatial_memory.h
src/                # Implementations
  ring_buffer.c
  tile.c
tests/              # Test suites
  test_ring_buffer.c
vendor/             # Git submodule
  probabilistic_data_structures/
    hyperloglog/    # HyperLogLog implementation
    bloom_filter/   # Bloom filter implementation
    lib/            # Shared utilities (hash functions, bit arrays)
```

## Dependencies

- [probabilistic_data_structures](https://github.com/arjtala/probabilistic_data_structures) — HyperLogLog, Bloom filter, hash functions (included as git submodule)
- [H3](https://h3geo.org/) — Uber's hexagonal hierarchical spatial index (`brew install h3`)

## Status

Work in progress. See [JOURNAL.md](JOURNAL.md) for development notes and detailed progress.
