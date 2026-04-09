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
- [ ] Add proper reference counting for HLL objects in ring buffer operations
- [x] Implement proper buffer management in `GpsTrace_push` to prevent overflow

## Architecture / Refactoring

- [ ] `viz_main.c` is ~1060 lines with inline types (`VideoQuad`, `ProgressBar`, `AttentionOverlay`) — extract these into their own source files
- [x] `viz_main.c` uses ~20 global variables for GLFW callbacks — use `glfwSetWindowUserPointer` with a context struct instead
- [ ] `SpatialMemory` forces `H3Index` → string conversion on every observe/query because `HashTable` requires string keys — consider a numeric hash map keyed by `H3Index` directly

## Portability

- [x] `#include <OpenGL/gl3.h>` in all viz headers is macOS-only; needs platform-conditional includes for Linux/Windows
- [x] Makefile uses `brew --prefix` exclusively — no fallback for non-Homebrew systems

## Testing

- [x] No tests for ingest module (`IngestReader`, `ImuGpsReader`)
- [x] No tests for pure-logic viz functions (`count_to_color`, `classify_motion`, `osm_zoom_from_degrees`, `latlon_to_tile`, `normalize_angle`, `estimate_speed`)
- [x] No test for `SpatialMemory_observe` adding to the same cell twice (verifying HLL de-duplication)
