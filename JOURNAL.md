# Time-Decayed Probabilistic Spatial Memory from Egocentric Video

## Overview

This project proposes a **Probabilistic Episodic Spatial Memory Engine** that builds a bounded-memory, time-decayed representation of the world from:

- Egocentric video
- GPS/WPS trajectory data
- Frame-level semantic embeddings (e.g., from V-JEPA2, DINO-style models)

The system models *what has been seen*, *where it was seen*, and *how memory fades over time*, using probabilistic data structures such as HyperLogLog and Bloom filters.

---

## Core Idea

> Build a spatial memory system that approximates what an agent has seen, where it saw it, and how that memory decays — under strict memory constraints.

Instead of storing raw data or full embeddings indefinitely, the system maintains compact, mergeable, approximate summaries tied to geographic regions and time windows.

---

## Key Components

### 1. Input Stream

For each video frame:

```
frame → embedding (d-dimensional)
      → timestamp
      → GPS coordinate
```

This produces a stream of tuples:

```
(t, lat, lon, embedding)
```

---

### 2. Spatial Partitioning

Space is divided into tiles using one of:

- Geohash
- Quadtree
- Uniform lat/lon grid

Each tile maintains probabilistic memory summaries.

---

### 3. Probabilistic Memory Structures

Each spatial tile stores:

- **HyperLogLog (HLL)**
  Estimates number of distinct semantic items encountered.

- **Bloom Filter**
  Approximate membership test: “Have we seen this before?”

- **Time-Sliced Ring Buffer of Sketches**
  Enables sliding-window memory and time decay.

---

## What Is Being Counted?

Three possible semantic granularities:

### Option A — Unique Visual States
Hash full-frame embeddings.

Measures:
> Distinct perceptual experiences in a region.

---

### Option B — Unique Objects
Extract object-level embeddings and hash them.

Measures:
> Distinct object identities encountered in space.

---

### Option C — Unique Semantic Clusters
Apply locality-sensitive hashing or clustering to embeddings before sketching.

Measures:
> Distinct semantic categories encountered in space.

This is more stable under lighting and viewpoint changes.

---

## Time Decay Mechanism

Each spatial tile maintains a fixed-size ring buffer:

```
Tile:
  HLL[t-0]
  HLL[t-1]
  HLL[t-2]
  ...
```

Each bucket represents a time slice (e.g., 5 minutes).

To compute memory for the last N minutes:
- Merge recent HLLs
- Ignore expired buckets

This implements:

- Sliding-window distinct counts
- Natural forgetting
- Bounded memory usage

---

## Visualization (OpenGL)

### 1. Memory Glow Map
Tiles glow by:
- Recent semantic novelty
- Distinct object counts
- Diversity of experiences

Older memory fades over time.

---

### 2. Ego Path Visualization
Render trajectory as polyline:
- Color-coded by novelty
- Highlight revisited regions
- Show familiarity gradients

---

### 3. Memory Persistence Panel
Click a tile to display:

- Estimated distinct entities over time
- Decay curve of memory strength
- Novelty spikes

---

## Research Questions

- How stable are embedding hashes under semantic perturbation?
- How does approximate memory compare to exact memory?
- How quickly does spatial familiarity converge?
- How does memory decay affect perceived novelty?

---

## Why This Is Interesting

This system combines:

- Self-supervised visual embeddings
- Probabilistic data structures
- Geospatial indexing
- Time decay and forgetting
- Real-time visualization

It models a resource-bounded approximation of episodic memory.

Unlike trajectory prediction systems, this focuses on:

> Representation of memory, not prediction of behavior.

---

## Implementation Phases

### Phase 1
- Offline embedding extraction
- GPS-aligned spatial tiling
- HLL + Bloom filter implementation in C
- Basic OpenGL map visualization

### Phase 2
- Sliding-window time decay
- Novelty detection
- Interactive exploration

### Phase 3
- Online embedding inference
- GPU-assisted visualization
- Internal sketch visualization (e.g., HLL registers as 3D structures)

---

## Stretch Ideas

- Locality-sensitive hashing for semantic stability
- Compare raw hashing vs clustered hashing
- Visualize HLL register distribution
- Model drift detection in semantic memory

---

## Conceptual Framing

This project can be viewed as:

> A computational model of episodic spatial memory under bounded storage constraints.

It bridges systems design, probabilistic algorithms, embodied perception, and visualization.

---

## Dev Journal

### 2026-03-04 — Code Review: `ring_buffer.c`

Reviewed the `RingBuffer` implementation — a fixed-size circular buffer of HyperLogLog counters that powers the time-decay mechanism.

#### Bugs Found & Fixed

| Severity | Issue | Resolution |
|----------|-------|------------|
| **Critical** | `RingBuffer_advance` wrote to `rb->hll` (nonexistent field) instead of `rb->hlls[rb->head]`, leaving a dangling pointer after `freeHLL` | Fixed to `rb->hlls[rb->head] = HLL_default(rb->precision)` |
| **High** | `RingBuffer_merge_window` had inconsistent return ownership: `n == 0` returned an internal pointer (caller must not free), `n >= 1` returned a new allocation (caller must free) | Fixed by returning `HLL_merge_copy(curr_hll, curr_hll)` for `n == 0`, giving all paths consistent caller-owned semantics |
| **Medium** | `head` was declared as `int` but used in `size_t` modular arithmetic | Changed to `size_t head` in the struct definition |

#### Bugs Introduced & Caught During Fixes

| Issue | Resolution |
|-------|------------|
| `rb->size_t = 0` — accidentally used the type name as a field name | Reverted to `rb->head = 0` |
| `RingBuffer_free(merge_hll)` / `RingBuffer_free(prev_hll)` — wrong function (`RingBuffer_free` vs `freeHLL`), typo (`merge_hll` vs `merged_hll`), and freeing an internal pointer (`prev_hll`) | Removed both lines; no cleanup needed since the merged result is returned to the caller |
| `freeHLL(merged_hll)` before `return curr_hll` — use-after-free since both point to the same object after the loop | Removed the free; caller owns the returned HLL |
| `RingBuffer *curr_rb = RingBuffer_current(rb)` — type mismatch (`RingBuffer*` vs `HLL*`) | Replaced with `HLL_merge_copy` deep-copy approach |

#### Key Design Decisions

- **Shallow vs deep copy**: Dereferencing an `HLL*` to a stack `HLL` creates a shallow copy. Since `HLL` contains `uint8_t *registers` (heap-allocated), both copies share the same register array. Freeing one corrupts the other. A deep copy is required.
- **Deep copy via self-merge**: `HLL_merge_copy(hll, hll)` produces a deep copy because it allocates a new HLL with its own registers and takes the element-wise max (identity operation with itself). This avoids modifying vendor code to add a dedicated copy function.
- **Error handling in vendor code**: `HLL_new` (called by `HLL_default`) calls `exit(EXIT_FAILURE)` on allocation failure, so it never returns NULL. No additional NULL checks needed for `HLL_default`. However, `HLL_merge_copy` *can* return NULL on mismatched parameters.

#### Final State

All code paths in `RingBuffer_merge_window` now return a caller-owned `HLL*` that must be freed with `freeHLL()`.

---

### 2026-03-05 — Project Status & Next Steps

#### Current State

| Component | File(s) | Status |
|-----------|---------|--------|
| HLL (vendor) | `vendor/probabilistic_data_structures/hyperloglog/hll.h` | Complete |
| Ring Buffer | `src/ring_buffer.c`, `include/ring_buffer.h` | Complete (reviewed & bug-fixed) |
| Spatial Tile | `src/tile.c`, `include/tile.h` | Stubbed (empty) |
| Spatial Memory Engine | `include/spatial_memory.h` | Stubbed (empty) |
| Ring Buffer Tests | `tests/test_ring_buffer.c` | Stubbed (empty) |
| Build System | `Makefile` | Empty |

#### Next Steps (Phase 1)

1. **Write tests for `ring_buffer.c`** — Lock down correctness after the bug-fix pass before building on top of it
2. **Implement `tile.c`** — Spatial tile that pairs a geographic region with a ring buffer of HLLs
3. **Set up the `Makefile`** — Get the project building and tests running

---

### 2026-03-06 — Ring Buffer Tests & Vendor Refactor

#### Ring Buffer Tests

Wrote `tests/test_ring_buffer.c` with 5 test cases covering all public API functions:

| Test | What it verifies |
|------|-----------------|
| `test_ring_buffer_new` | Capacity, precision, and head initialized correctly |
| `test_ring_buffer_current` | Returns `hlls[head]`, data added through pointer is reflected |
| `test_ring_buffer_advance` | Head increments, new slot is empty, previous slot retains data |
| `test_ring_buffer_wrap` | Head wraps to 0 after `capacity` advances, wrapped slot is fresh |
| `test_ring_buffer_merge_window` | Merge of 1 previous slot and full capacity merge produce expected counts, caller-owned results freed correctly |

Used vendor's `ASSERT` and `RUN_TEST` macros from `utilities.h`. All tests passing.

#### Common C mistakes caught during test writing

- `void fn()` vs `void fn(void)` — empty parens in C means unspecified parameters, not zero
- `const char **p = "string"` — double pointer instead of single, hashes the pointer not the string
- Asserting `0 <= count` (always true for unsigned) instead of `0 == count`
- Using `RingBuffer_current(rb)` to check old slot after advance (returns new slot, not old)
- Forgetting to free caller-owned `HLL*` returned by `RingBuffer_merge_window`
- Referencing undefined variable names (`curr_rb` instead of `curr_hll`)

#### Vendor Library Refactor (`probabilistic_data_structures`)

Separated function definitions from headers to allow proper multi-translation-unit compilation:

| File | Change |
|------|--------|
| `lib/hash.h` → `lib/hash.c` | Moved `djb2`, `sdbm`, `hash_64`, `fnv_64`, `murmur64` |
| `lib/bitarray.h` → `lib/bitarray.c` | Moved `createBitArray`, `freeBitArray`, `printBits`, `unit_to_binary`, `printBinary`, `msb_position` |
| `lib/utilities.h` → `lib/utilities.c` | Moved `printSeparator`, `load_sentences`, `format_with_commas` |
| `hyperloglog/hll.h` → `hyperloglog/hll.c` | Moved all HLL function implementations; changed `const int NUM_BITS_PER_REGISTER` to `#define` |
| `bloom_filter/bloom.h` → `bloom_filter/bloom.c` | Moved all BloomFilter function implementations |

Added missing declarations to headers: `HLL_merge_copy`, `HLL_memory_usage`, `countBitsSet`.

Replaced per-directory Makefiles with a single top-level Makefile that builds a combined `libpds.a` and runs tests for both HLL and Bloom filter.

#### Build System

Set up `Makefile` for the spatial memory project:
- Compiles project sources (`src/*.c`) and vendor sources into separate `.o` files
- Vendor objects go under `build/vendor/` mirroring source layout
- `make` builds `libpsm.a`, `make test` compiles and runs ring buffer tests
- Include paths: `-Iinclude -I.` for project, `-I$(VENDOR)/lib -I$(VENDOR)/hyperloglog -I$(VENDOR)/bloom_filter` for vendor

#### Updated Project Status

| Component | File(s) | Status |
|-----------|---------|--------|
| HLL (vendor) | `vendor/probabilistic_data_structures/hyperloglog/hll.{h,c}` | Complete (refactored) |
| Bloom Filter (vendor) | `vendor/probabilistic_data_structures/bloom_filter/bloom.{h,c}` | Complete (refactored) |
| Ring Buffer | `src/ring_buffer.c`, `include/ring_buffer.h` | Complete (tested) |
| Ring Buffer Tests | `tests/test_ring_buffer.c` | Complete (5 tests passing) |
| Build System | `Makefile` | Complete |
| Spatial Tile | `src/tile.c`, `include/tile.h` | Stubbed (empty) |
| Spatial Memory Engine | `include/spatial_memory.h` | Stubbed (empty) |

#### Next Steps

1. **Implement `tile.c`** — Spatial tile pairing a geographic region with a ring buffer of HLLs
2. **Implement `spatial_memory.h`** — Top-level engine managing a collection of tiles
3. **Add `compile_flags.txt`** or generate `compile_commands.json` via Bear for LSP support

---

### 2026-03-06 — Tile Implementation & H3 Integration

#### Tile Implementation

Implemented `tile.c` — a spatial tile that pairs a geographic region (H3 cell) with a ring buffer of HLL counters:

| Function | Description |
|----------|------------|
| `Tile_new(lat, lng, resolution, capacity, precision)` | Converts lat/lng to H3 cell ID, allocates ring buffer |
| `Tile_free(tile)` | Frees ring buffer and tile |
| `Tile_add(tile, data, size)` | Records observation in current time window via `HLL_add` |
| `Tile_advance(tile)` | Rotates to next time window via `RingBuffer_advance` |
| `Tile_query(tile, n)` | Returns distinct count over last N windows, properly frees merged HLL |

#### H3 Integration

Replaced geohash strings with [Uber's H3](https://h3geo.org/) hexagonal spatial index:

- Installed via `brew install h3`, linked with `-lh3`
- `H3Index` is a `uint64_t` — no string allocation/freeing needed
- `latLngToCell()` converts coordinates to cell IDs with error checking
- Advantages over geohash: uniform cell area, no edge discontinuities, hierarchical via bit-shifting, integer comparisons

Makefile updated with `H3_PREFIX = $(shell brew --prefix h3)` for portable Homebrew path resolution.

#### Tile Tests

Wrote `tests/test_tile.c` with 5 test cases:

| Test | What it verifies |
|------|-----------------|
| `test_tile_new` | H3 cell ID matches known value, ring buffer initialized correctly |
| `test_tile_add` | Observation increases HLL count |
| `test_tile_advance` | New time window is empty after advance |
| `test_tile_query` | Merged count across windows reflects distinct items |
| `test_tile_same_cell` | Two nearby coordinates at same resolution produce same cell ID |

#### Build System Updates

- `make test` now runs both `test-ring-buffer` and `test-tile`
- `make test-tile` / `make test-ring-buffer` for running individual suites
- LSP support via `bear -- make test` to generate `compile_commands.json`

#### Updated Project Status

| Component | File(s) | Status |
|-----------|---------|--------|
| HLL (vendor) | `vendor/probabilistic_data_structures/hyperloglog/hll.{h,c}` | Complete (refactored) |
| Bloom Filter (vendor) | `vendor/probabilistic_data_structures/bloom_filter/bloom.{h,c}` | Complete (refactored) |
| Ring Buffer | `src/ring_buffer.c`, `include/ring_buffer.h` | Complete (tested) |
| Ring Buffer Tests | `tests/test_ring_buffer.c` | Complete (5 tests passing) |
| Tile | `src/tile.c`, `include/tile.h` | Complete (tested) |
| Tile Tests | `tests/test_tile.c` | Complete (5 tests passing) |
| Build System | `Makefile` | Complete (H3 integrated) |
| Spatial Memory Engine | `include/spatial_memory.h` | Stubbed (empty) |

#### Next Steps

1. **Implement `spatial_memory`** — Top-level engine managing a collection of tiles (hash map of H3Index → Tile)
2. **Visualization** — OpenGL memory glow map (Phase 1 stretch)
