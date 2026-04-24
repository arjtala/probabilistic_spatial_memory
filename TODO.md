# TODO Items

## Bugs

- [x] `gps_trace.c:73-78` — `GpsTrace_push` realloc has a dangling pointer bug: if the first `realloc` succeeds (freeing the old buffer) but a subsequent one fails, the early-return leaves `gt->lats`/`gt->lngs`/`gt->imu_meta` pointing at freed memory
- [x] `ingest.c:90-98` — `IngestReader_open` error path leaks HDF5 dataset handles: if any of `dataset_ts`/`lat`/`lng`/`emb` fail to open, the ones that succeeded are never closed
- [x] `jepa_cache.c:80-85` — `JepaCache_load` leaks `timestamps` and `prediction_maps` if the final `malloc(sizeof(JepaCache))` fails
- [x] `spatial_memory.h:12-15` — Block-commented-out API documentation should be cleaned up or uncommented

## Error Handling Issues

- [x] `ring_buffer.c:5-8` — `RingBuffer_new` calls `exit(EXIT_FAILURE)` on malloc failure instead of returning NULL
- [x] `tile.c:6-9` — `Tile_new` calls `exit(EXIT_FAILURE)` on malloc/H3 failure instead of returning NULL
- [x] `spatial_memory.c:8-16` — `SpatialMemory_new` calls `exit(EXIT_FAILURE)` instead of returning NULL
- [x] `spatial_memory.c:30-33` — `SpatialMemory_observe` calls `exit(EXIT_FAILURE)` on H3 conversion failure instead of handling gracefully
- [x] `ingest.c` — `IngestReader_next` never checks return values of `H5Dread`/`H5Sselect_hyperslab` calls
- [x] `ingest.c:324-326` — `ImuGpsReader_open` doesn't check if all three GPS mallocs (`gps_ts`/`gps_lat`/`gps_lng`) succeeded before reading into them
- [x] `viz_main.c:633-634` — `atof`/`atoi` used for CLI arg parsing with no validation; `strtod`/`strtol` would catch non-numeric input
- [x] Add bounds checking for H3 resolution parameters in `SpatialMemory_new`

## Code Duplication Issues

- [x] `spatial_memory.c` — `SpatialMemory_observe` and `SpatialMemory_query` both duplicate the `latLngToCell` + `h3ToString` pattern; extract a helper
- [x] `ingest.c` — IMU rank-2 validation (accel/gyro shape check with `H5Sget_simple_extent_ndims`) is duplicated between `IngestReader_open` and `ImuGpsReader_open`
- [x] `ingest.c` — HDF5 row-read pattern (create memspace → get dataspace → select hyperslab → read → close) repeated ~10 times; extract a helper
- [x] `viz_main.c` — `VideoQuad_update_aspect` and `AttentionOverlay_update_aspect` are identical; extract shared function
- [x] `viz_main.c` — Identity matrix construction duplicated in `ProgressBar_draw` and `ProgressBar_draw_pause_icon`
- [x] Ortho projection matrix built identically in `HexRenderer_draw`, `GpsTrace_draw`, and `TileMap_draw`
- [x] Consolidate H3 index creation code between `Tile_new` and `SpatialMemory_observe`/`SpatialMemory_query`

## Consistency Issues

- [x] `ring_buffer.c` / `tile.c` — use `fprintf`, `exit`, `malloc`, `free` without explicit `<stdio.h>` / `<stdlib.h>` includes (relying on transitive includes from vendor headers)
- [x] Core modules (`ring_buffer`, `tile`, `spatial_memory`) call `exit()` on errors while ingest/viz modules return NULL — should pick one strategy
- [x] Standardize precision parameter handling across all HLL operations
- [x] Ensure consistent error return values in all HDF5 reader functions

## Memory Management Issues

- [x] Fix potential memory leaks in `ImuGpsReader_interpolate_gps` when GPS data is absent
- [x] Add proper reference counting for HLL objects in ring buffer operations
- [x] Implement proper buffer management in `GpsTrace_push` to prevent overflow

## Architecture / Refactoring

- [x] `viz_main.c` is ~1060 lines with inline types (`VideoQuad`, `ProgressBar`, `AttentionOverlay`) — extract these into their own source files
- [x] `viz_main.c` uses ~20 global variables for GLFW callbacks — use `glfwSetWindowUserPointer` with a context struct instead
- [x] `SpatialMemory` forces `H3Index` → string conversion on every observe/query because `HashTable` requires string keys — consider a numeric hash map keyed by `H3Index` directly

## Portability

- [x] `#include <OpenGL/gl3.h>` in all viz headers is macOS-only; needs platform-conditional includes for Linux/Windows
- [x] Makefile uses `brew --prefix` exclusively — no fallback for non-Homebrew systems

## Testing

- [x] No tests for ingest module (`IngestReader`, `ImuGpsReader`)
- [x] No tests for pure-logic viz functions (`count_to_color`, `classify_motion`, `osm_zoom_from_degrees`, `latlon_to_tile`, `normalize_angle`, `estimate_speed`)
- [x] No test for `SpatialMemory_observe` adding to the same cell twice (verifying HLL de-duplication)

## Next Phase

- [x] Add CI plus safer build profiles in `Makefile` (`debug`, sanitizers, portable release) and run them on macOS/Linux
- [x] Add selectable heatmap modes for the map view, configurable via viz config and switchable at runtime
- [x] Split remaining large viz modules (`src/viz/viz_main.c`, `src/viz/tile_map.c`) into smaller controller / scheduler / HUD / tile-pipeline pieces
- [x] Expand headless tests for viz interaction, adaptive budgets, and tile-cache behavior
- [x] Add an on-screen help overlay plus heatmap legend overlay for the visualizer
- [x] Add dependency-free screenshot export for composed viz frames under `captures/`
- [x] Upgrade `targets/psm` from a demo entrypoint to a real CLI with flags for resolution/capacity/precision and structured output
- [x] Turn the open questions in `README.md` into explicit experiments and reproducible benchmark sweeps
- [x] Remove the accidental tracked top-level `endif` artifact

## Follow-Up

- [x] Improve the startup/help overlay readability and make the `P` screenshot action explicit in the on-screen controls
- [x] Switch screenshot export from BMP to PNG and validate the written files in tests

## Render & Frame Pipeline

- [ ] Replace `TileMap` linear cache scan with an open-addressed hash keyed on packed `(x, y, z)` — `src/viz/tile_map.c:29-55` (eliminates ~7.7k compares/frame at radius 5)
- [ ] Preallocate `HexRenderer` vertex buffer on the struct and grow-only — `src/viz/hex_renderer.c:240-263` (no more per-frame malloc/free of the scratch buffer)
- [ ] Cache `H3_boundary` and `cell_center` per `Tile` so `HexRenderer_update` doesn't recompute H3 geometry every frame — `src/viz/tile.c`
- [ ] Batch `TileMap_draw` into a single draw call instead of per-tile VBO uploads
- [ ] Move video decode + `sws_scale` off the main thread into a producer thread, reusing the tile-pipeline SPSC pattern — `src/viz/video_decoder.c`
- [ ] Cache `cos(center_lat * π/180)` in `HexRenderer_draw` rather than recomputing every draw — `src/viz/hex_renderer.c:279-281`
- [ ] Dirty-check the HUD title so `snprintf` + `glfwSetWindowTitle` only run when fields change — `src/viz/viz_debug_hud.c:59`

## Core Engine Clarity

- [ ] Add an explicit `HLL_clone` helper and replace `HLL_merge_copy(curr, curr)` self-merge-as-clone — `src/core/ring_buffer.c:143`
- [ ] Distinguish OOM from empty-ring returns in `RingBuffer_merge_window` (error out-param or sentinel) — `src/core/ring_buffer.c:129-158`
- [ ] Rename `ret` → `send_ret` / `recv_ret` in `VideoDecoder_next_frame` and annotate the state machine — `src/viz/video_decoder.c:121-178`
- [ ] Add a `max_iterations` guard to `VideoDecoder_seek` to prevent hangs on pathological files
- [ ] Remove or document the unused running-mean state in `GpsTrace_push`
- [ ] Delete dead API `VizScreenshot_build_default_path` — `src/viz/screenshot.c:249`

## Architecture & API Boundaries

- [ ] Split `viz_main.c` (1027 LOC) into `viz_session` (init/teardown), `viz_event_loop` (tick + input), and `viz_render` (draw submission); replace the duplicated cleanup block with a `goto cleanup` ladder
- [ ] Expose `ImuGpsReader_reset()` and remove direct `gps_cursor = 0` reach-ins from `src/viz/viz_main.c`
- [ ] Collapse `viz_config.c:322-522` per-key if-ladder into a static `{key, type, offset, parser}` dispatch table

## Screenshot & Export

- [ ] Replace the uncompressed STORE-only zlib with real DEFLATE — `src/viz/screenshot.c:90-184` (libpng `png_set_compression_level(9)` when `USE_LIBPNG`; miniz `tdefl` for the fallback path; expected 70-85% size reduction)
- [ ] Add image-sequence PNG export (`--save-every N`) for short recordings — covers 80% of "record a run" use cases before committing to MP4/FFmpeg muxing

## Disk Cache

- [ ] Maintain an in-memory inventory updated incrementally on insert/evict; only rescan the tile cache tree on startup — `src/viz/tile_disk_cache.c:248-295` (avoid main-thread stall on large caches)

## Visualizer UX

- [ ] Add a lightweight map-cell inspector: hover or click a hex, show its count, mode value, recency, and H3 id; wire to an `I` toggle key
- [ ] Add a legend panel showing the numeric ramp for the active `HexHeatmapMode` (today's legend only shows "LOW"/"HIGH")

## CLI & Security

- [ ] Add `--version` to `psm` and `psm-viz`; embed `git describe` at build time via `-DPSM_VERSION` in the Makefile — `src/main.c`
- [ ] Add a `schema_version` field to `psm -j` JSON output so downstream `jq` pipelines stay stable across schema evolution
- [ ] Add a `--verify-hdf5` subcommand that checks dataset shapes, dtypes, and timestamp monotonicity before ingest
- [ ] Validate URL template tokens (whitelist `{s}`/`{z}`/`{x}`/`{y}`/`{api_key}`) and warn when `{api_key}` is used over plain HTTP — API-key exfiltration risk
- [ ] Reject `..` sequences and null bytes in configured paths (tile cache root, capture dir, HDF5 input)

## Testing

- [ ] Edge-case suite: NaN/Inf lat/lng, H3 resolution 15 (edge of valid range), truncated HDF5, zero-capacity ring buffer — confirm clean error paths rather than crashes
- [ ] Add visual regression coverage for the overlay/screenshot path so UI changes are harder to break silently (headless EGL + golden-PNG diff, tolerance >1%)

## CI & Tooling

- [ ] Add an advisory `clang-tidy` CI job + `make lint` target; promote to a gate after the `viz_main.c` split (pre-split noise would drown signal)
- [ ] Add `make check-format` using the existing `.clang-format`
- [ ] Add Linux CI for `viz` builds/tests (`xvfb-run` + OSMesa/EGL headless) now that the portability work is in place
- [ ] Migrate Makefile test dependencies from the `$(HEADERS)` wildcard to generated per-TU deps (`-MD -MP`) for accurate incremental builds

## Deferred / Measure First

- [ ] HDF5 dataspace reuse across row reads — realistic budget 5-15% on ingest-heavy workloads; benchmark before committing to a target number
- [ ] Add a Performance section to `README.md` documenting Big-O for the hot paths: observe O(1), query O(capacity × log(precision)), advance O(tiles)

## Localization Paradox Alignment

Context: a forthcoming NeurIPS 2026 streaming egocentric memory benchmark (the "Localization Paradox benchmark" after its headline finding) exposes models' failure to return supporting `[t_start, t_end]` intervals for look-back questions — frontier MLLMs score near-zero `mIoU` despite respectable semantic accuracy. PSM's H3-indexed ring-buffered memory is a natural substrate for closing that gap. These items add the minimum primitives needed to emit intervals and retrieve exemplars; experiments E5-E7 in `EXPERIMENTS.md` consume them.

- [ ] Retain `(t_min, t_max)` per ring-buffer bucket alongside the HLL sketch — enables returning `[t_start, t_end]` candidate intervals; cost ~`16B × capacity × tile_count`
- [ ] Reservoir-sampled per-tile exemplar embeddings (configurable `N` per tile) — enables k-NN retrieval against past observations for "visual detail recall" and "last seen" queries
- [ ] Expose `SpatialMemory_query_intervals(lat, lng, k_ring, out_tuples)` returning top-k `(cell, t_start, t_end, count)` tuples over the H3 neighborhood
- [ ] `psm --last-seen lat,lng --k N` CLI surface + JSON output (bump `schema_version` when adding fields; see also CLI & Security → `schema_version` item)
- [ ] Benchmark scenario in `benchmarks/benchmark_spatial_memory.c`: "location-trace query latency" over a populated session — first-class measurement for E7
