# TODO Items

## Review 2026-07-08 — full-codebase review + fixes (applied)

Full-codebase review (10 subsystem lanes, adversarial per-finding verification): 56 raw findings → 50 survived (36 confirmed, 14 plausible; 6 rejected as false positives). All verified findings were fixed the same day and the build + full test suite were re-verified green (C `-Werror` + 17 C tests + 59 extraction pytest). Fix writeup: [journal/review_fixes_2026-07-08.md](journal/review_fixes_2026-07-08.md).

### Fixed — High
- [x] `src/viz/progress_bar.c` — VBO allocated for `12*6` floats but start-overlay uploads `30*6` → `glBufferSubData` rejected (`GL_INVALID_VALUE`), splash drew stale geometry. VBO now sized `30*6` (covers all three draw paths).
- [x] `extraction/psm_extraction/models/dino_pytorch.py` — added `attn_implementation="eager"`; on `transformers>=4.40` the default SDPA returned `attentions=None`, silently dropping attention maps + `patch_grid`. Plus a config-derived `patch_grid` fallback.
- [x] `scripts/eval_mllm_baseline.py` — exemplar hit was Hit@1 (top-1 only) under a `_at_5` key; now ORs over all top-k predictions to match `_eval_common`/`eval_lookback` (fixes the MLLM-vs-PSM headline comparison).
- [x] `.github/workflows/ci.yml` — jobs installed only `h3 hdf5` while `make test` links GLFW viz tests → CI could never pass on a clean runner. Added `glfw` (brew) / `libglfw3-dev` (apt) to all jobs.

### Fixed — Medium
- [x] `src/core/spatial_memory.c` — `SpatialMemory_advance_to_timestamp` capped physical advances at ring capacity (was unbounded on large timestamp gaps); anchor still moves to the exact grid position.
- [x] `extraction/psm_extraction/extract.py` — VRS/EgoExo model-group vs `--gps-json` sensor-group clock mismatch: sensor groups now co-located on the model group's origin via an explicit `rebase_origin` (residual documented; partial where clocks are genuinely independent).
- [x] `extraction/psm_extraction/io/json_sidecar.py` — `read_gps_json`/`read_imu_json` now select the stream with the most *valid* samples (was largest-raw-count before validity filtering).
- [x] `README.md` — CLI flag table + `--search` JSON fields brought in sync with `psm --help` (see doc fixes below).
- [x] CI: added a Python `extraction` pytest job (was no Python coverage in CI).
- [x] `tests/test_imu_processor.c` (new) + `test-imu-processor` Makefile target — the IMU dead-reckoning integrator now has coverage (heading integration, dead-reckoning displacement, dt-gating boundary).
- [x] `extraction/psm_extraction/io/nymeria_narration.py` — removed alignment overclaims; documented the `trajectory_t0` vs `rgb_t0` offset and exposed `trajectory_t0_sec` for downstream re-alignment (partial — full rebase needs the RGB-frame origin from the extractor).

### Fixed — Low
- [x] `src/core/spatial_memory.c` — guarded `size_t` overflow in `per_cell_cap` scratch sizing.
- [x] `src/core/tile.c` — `exemplar_seen` now advanced only after a successful encode (Algorithm-R bias on encode failure).
- [x] `src/core/ring_buffer.c` — consistent NULL-slot guard in `RingBuffer_merge_window`.
- [x] `include/core/spatial_memory.h` — corrected stale `query_similar` return-value + `SpatialMemorySimilar` per-cell docs.
- [x] `src/ingest/ingest.c` — backward in-range GPS query resets cursor (was forward-only extrapolation returning success); reject zero-width embedding datasets; documented the non-decreasing-timestamp precondition.
- [x] `src/main.c` — explicit `-g` no longer overridden by a positional; explicit `--exemplars 0` now honored (added `has_group`/`has_exemplar_capacity` flags).
- [x] `src/viz/viz_config.c` — `hex_extrude_scale` parser now rejects trailing garbage / errno (matches sibling parsers).
- [x] `src/viz/viz_main.c` — NULL-check `malloc` in `find_file_in_dir`.
- [x] `src/viz/shader.c` — check `ftell` return before allocating (avoids silently-empty shader source).
- [x] `extraction/psm_extraction/models/clip_pytorch.py` — replaced tensor-truthiness `or` chain; probe `max_position_embeddings` instead of hard-coded 77.
- [x] `extraction/psm_extraction/models/jepa_pytorch.py` — `embed_images` default `batch_size` 4 → 16 (matches ABC).
- [x] `extraction/psm_extraction/__main__.py` — `--group` flag now honored (was dead) for single-model extraction.
- [x] `extraction/psm_extraction/migrate.py` — added `siglip`/`longclip` to `KNOWN_GROUP_DEFAULTS` (were skipped while still stamping schema v2).
- [x] `extraction/psm_extraction/writer.py` — `write_model_group` now enforces `schema.MODEL_REQUIRED_ATTRS`/`_DATASETS` (constants are now the enforced contract).
- [x] `extraction/psm_extraction/io/aria_vrs.py` — sort + dedup SLAM/GPS timestamps before `np.interp` (undefined on non-monotonic `xp`).
- [x] `extraction/psm_extraction/io/hdd.py` — `video_start_skew` handles tz-aware ISO timestamps instead of blindly stripping tzinfo.
- [x] `scripts/bootstrap_ci.py` — `_extract_arrays` tolerates `eval_mllm_baseline` shape (metrics under `predictions`) instead of KeyError.
- [x] `scripts/eval_aggregate.py` — negative-control set now excludes counting/spatial-only questions (matches `eval_lookback`).
- [x] `scripts/eval_mllm_baseline.py` — dropped the `max(0.0, …)` bucket-window clamp so IoU matches the shared scorer.
- [x] `.gitignore` — removed the over-broad `*/*.txt` (would silently ignore a future `extraction/requirements-dev.txt`); anchored `*.gch`/`*.tmp`.
- [x] CI: added a compile step for the C benchmarks (were never built by CI → bit-rot risk).

### Docs synced
- [x] `README.md` — added `-v/--version`, `--per-cell-cap`, `--exemplar-codec`, `--exemplars 0` to the CLI table; documented `exemplar_codec`/`exemplar_payload_bytes` JSON fields; added `torchvision` to the `clip` extra; refreshed the `src/`, `include/`-adjacent, `benchmarks/`, and `tests/` structure listings.
- [x] `EXPERIMENTS.md` — E0 pipeline item 2 pointed at the real `scripts/extract_bigg_all.sh` (the never-created `extract_nymeria_all.sh` reference).
- [x] `TODO.md` — corrected the stale `--clip-checkpoint`-threading claim for the sliding-window / uniform-sample sweeps (both already had it; remaining blocker is compute only).

### Rejected (false positives — no change needed)
GPS-monotonicity "crash" (negative `dt` short-circuits safely), `attention_overlay` OOB (buffer sized from the same dim as the loop), DINO register-count collision (impossible for real grids), IMU/GPS clock reconciliation (consumer already does it via `_resolve_offset`), `_at_5` key "mislabel" (suffix is discarded), `make run` "fails" (it's `.PHONY`, exits 0).

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
- [x] Preallocate `HexRenderer` vertex buffer on the struct and grow-only — `src/viz/hex_renderer.c:240-263` (no more per-frame malloc/free of the scratch buffer)
- [ ] Cache `H3_boundary` and `cell_center` per `Tile` so `HexRenderer_update` doesn't recompute H3 geometry every frame — `src/viz/tile.c`
- [ ] Batch `TileMap_draw` into a single draw call instead of per-tile VBO uploads
- [ ] Move video decode + `sws_scale` off the main thread into a producer thread, reusing the tile-pipeline SPSC pattern — `src/viz/video_decoder.c`
- [x] Cache `cos(center_lat * π/180)` in `HexRenderer_draw` rather than recomputing every draw — `src/viz/hex_renderer.c:279-281`
- [ ] Dirty-check the HUD title so `snprintf` + `glfwSetWindowTitle` only run when fields change — `src/viz/viz_debug_hud.c:59`

## Core Engine Clarity

- [x] Add an explicit `HLL_clone` helper and replace `HLL_merge_copy(curr, curr)` self-merge-as-clone — `src/core/ring_buffer.c:143`
- [x] Distinguish OOM from empty-ring returns in `RingBuffer_merge_window` (error out-param or sentinel) — `src/core/ring_buffer.c:129-158`
- [x] Rename `ret` → `send_ret` / `recv_ret` in `VideoDecoder_next_frame` and annotate the state machine — `src/viz/video_decoder.c:121-178`
- [x] Add a `max_iterations` guard to `VideoDecoder_seek` to prevent hangs on pathological files
- [x] Remove or document the unused running-mean state in `GpsTrace_push`
- [x] Delete dead API `VizScreenshot_build_default_path` — `src/viz/screenshot.c:249`

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
- [x] 3D hex extrusion mode: cabinet-projection hex height proportional to `count / max_count`. Config knob (`hex_extrude_scale`) + `E` toggle. Lives entirely inside `src/viz/hex_renderer.c` and `shaders/hex.vert` — basemap stays flat top-down.

## Vector Basemap Rebuild (Future)

A full Mapbox/Apple-style 3D map look (tilted camera, billboarded labels, extruded buildings) is out of scope for the cabinet-extrusion idiom that ships today. It needs a real 3D camera, vector tiles instead of raster, and a label renderer. Tracked here so we don't lose the scope estimate:

- [ ] Phase 1 — Vector tile pipeline. Replace raster CartoDB tiles with Protomaps PMTiles. Tasks: PMTiles header/index parser + range-request fetcher (~300 LOC); MVT (Mapbox Vector Tile) protobuf decoder (nanopb or hand-rolled, ~500 LOC); replace `tile_map.c` raster path with vector polygon + line rendering (~1 day GL work for line tessellation with miter joins). End state: water/parks/road network on screen, no labels, no buildings.
- [ ] Phase 2 — True perspective camera. Replace ortho 2×2 basis with view + perspective matrices, pitch, far plane, and frustum-cull. Touches every renderer's draw call. End state: tiltable map view, raster-equivalent rendering still works.
- [ ] Phase 3 — Building extrusion. Read `osm:height` (or estimated) tag from MVT, generate side-wall + top-cap mesh per building footprint. End state: 3D city silhouette under hex memory layer.
- [ ] Phase 4 — Billboarded labels. Glyph atlas (extend `ui_overlay`'s font path), world-positioned text always facing camera, basic anti-collision. ~2 days. The visual gap between "data hexes on a map" and "looks like a real map" is mostly here.
- [ ] Phase 5 — Style. Hardcoded color/width/font-size table per feature class; defer JSON style-spec parsing.

Total realistic effort: 5–6 focused days for a baseline; weeks to match a polished Mapbox aesthetic. Not on the critical path for the research questions in `EXPERIMENTS.md`; revisit when (a) experiments stabilize, or (b) `psm-viz` becomes a polished demo for a talk.

## Paper Figures

- [ ] Embedding-space companion figure to the geographic heatmap: project per-tile exemplar embeddings to 2D (UMAP or PCA), render as inferno density. Pairs with `psm-viz`'s geographic view to show what memory *contains* alongside where it *landed*. ~30 LOC Python over `features.h5`; lives under `scripts/` not the C engine.

## CLI & Security

- [x] Add `--version` to `psm` and `psm-viz`; embed `git describe` at build time via `-DPSM_VERSION` in the Makefile — `src/main.c`
- [x] Add a `schema_version` field to `psm -j` JSON output so downstream `jq` pipelines stay stable across schema evolution
- [ ] Add a `--verify-hdf5` subcommand that checks dataset shapes, dtypes, and timestamp monotonicity before ingest
- [ ] Validate URL template tokens (whitelist `{s}`/`{z}`/`{x}`/`{y}`/`{api_key}`) and warn when `{api_key}` is used over plain HTTP — API-key exfiltration risk
- [ ] Reject `..` sequences and null bytes in configured paths (tile cache root, capture dir, HDF5 input)

## Testing

- [ ] Edge-case suite: NaN/Inf lat/lng, H3 resolution 15 (edge of valid range), truncated HDF5, zero-capacity ring buffer — confirm clean error paths rather than crashes
- [ ] Add visual regression coverage for the overlay/screenshot path so UI changes are harder to break silently (headless EGL + golden-PNG diff, tolerance >1%)

## CI & Tooling

- [ ] Add an advisory `clang-tidy` CI job + `make lint` target; promote to a gate after the `viz_main.c` split (pre-split noise would drown signal)
- [x] Add `make check-format` using the existing `.clang-format`
- [ ] Add Linux CI for `viz` builds/tests (`xvfb-run` + OSMesa/EGL headless) now that the portability work is in place
- [x] Migrate Makefile test dependencies from the `$(HEADERS)` wildcard to generated per-TU deps (`-MD -MP`) for accurate incremental builds

## Deferred / Measure First

- [ ] HDF5 dataspace reuse across row reads — realistic budget 5-15% on ingest-heavy workloads; benchmark before committing to a target number
- [ ] Add a Performance section to `README.md` documenting Big-O for the hot paths: observe O(1), query O(capacity × log(precision)), advance O(tiles)

## Localization Paradox Alignment

Context: a forthcoming streaming egocentric memory benchmark (the "Localization Paradox benchmark" after its headline finding) exposes models' failure to return supporting `[t_start, t_end]` intervals for look-back questions — frontier MLLMs score near-zero `mIoU` despite respectable semantic accuracy. PSM's H3-indexed ring-buffered memory is a natural substrate for closing that gap. These items add the minimum primitives needed to emit intervals and retrieve exemplars; experiments E5-E7 in `EXPERIMENTS.md` consume them.

- [x] Retain `(t_min, t_max)` per ring-buffer bucket alongside the HLL sketch — enables returning `[t_start, t_end]` candidate intervals; cost ~`16B × capacity × tile_count`
- [x] Reservoir-sampled per-tile exemplar embeddings (configurable `N` per tile) — enables k-NN retrieval against past observations for "visual detail recall" and "last seen" queries
- [x] Expose `SpatialMemory_query_intervals(lat, lng, k_ring, out_tuples)` returning top-k `(cell, t_start, t_end, count)` tuples over the H3 neighborhood
- [x] `psm --last-seen lat,lng --k-ring N --top N` CLI surface + JSON output (`"mode": "last_seen"` discriminator; `schema_version` unchanged at 1)
- [x] Benchmark scenario in `benchmarks/benchmark_spatial_memory.c`: "location-trace query latency" over a populated session — first-class measurement for E7
- [x] `SpatialMemory_query_similar(query, dim, k_ring, center, out)` — rank tiles by cosine similarity of the best exemplar; `psm --search <bin>` / `--center LAT,LNG` / `--exemplars N` CLI; benchmark scenario `query_similar` (E5's text-query adapter now has a concrete backend to target)
- [x] `scripts/eval_lookback.py` `query_mode: last_seen` field — routes through `psm --last-seen` for true location_trace questions, bypassing CLIP entirely. Demonstrated bucket Hit@5 = 100% deterministic on Palo Alto q9.
- [x] `scripts/eval_lookback.py` `count: <int>` and `expected: <string>` fields — counting and categorical answers, recorded per-record (counting flagged diagnostic-only pending a real cardinality scorer).
- [x] `scripts/extract_bigg_all.sh` + `scripts/eval_bigg_all.sh` — paired encoder seed sweeps (CLIP-L vs OpenCLIP-bigG), powering the v2 follow-up writeup.
- [x] TurboQuant->PSM experiment (E9): compare raw float32 exemplar reservoirs against 2/3/4-bit TurboQuant-style compressed exemplars for `psm --search`, reporting top-k cell overlap, rank stability, cosine error, bytes/tile, and query latency. Result on bigG, 3 sessions × 5 seeds × 20 questions: Hit@5 statistically flat across all four codecs (raw 83.0±2.7%, tq2 81.0±4.2%); top-5 cell-set Jaccard 0.88 (4-bit) → 0.77 (2-bit); 4.9–9.7× exemplar-memory reduction. Full breakdown in `journal/localization_paradox2.md` § G; summary in `EXPERIMENTS.md` § E9 Result.
- [x] Add an exemplar codec boundary (`raw_f32` first, then TurboQuant-style bitpacked payloads) so `TileExemplar` can store compressed embeddings without changing the HLL counting path. Landed as `core/exemplar_codec.{h,c}` with `EXEMPLAR_CODEC_RAW`; threaded through `Tile_new` / `SpatialMemory_new` and consumed by `SpatialMemory_query_similar`. Adding a TurboQuant codec is now a localized change.
- [x] Faithful TurboQuant codecs (`turboquant_{2,3,4}b`): randomized Hadamard transform + Lloyd-Max-optimal quantization, bit-packed payload. Wired through `--exemplar-codec` CLI + `"exemplar_codec"` / `"exemplar_payload_bytes"` JSON fields. Smoke test on a CLIP-512 session: top-1 cell preserved at 4-bit (273 B/exemplar, 7.5x reduction vs raw 2048 B) and at 2-bit (145 B, 14x).
- [x] Extend `benchmarks/benchmark_spatial_memory.c` or add a small query-bank script to sweep exemplar codec, bit budget, and reservoir size against the raw-float32 baseline. Done on the question-bank side via `scripts/eval_codec_drift.py` + the 4-codec sweep through `eval_bigg_all.sh CODEC=...`; the in-engine bytes/tile + median µs sweep in `benchmark_spatial_memory.c` is still open and tracked under E9 follow-ups in EXPERIMENTS.md.
- [ ] **Real counting scorer for E6.** The current `count_predicted = len(distinct cells in top-k)` is `--top`-cap-bound (under-predicts at top-5, saturates at top-20). Two options: a similarity-threshold cell counter in `eval_lookback.py`, or a `psm --cardinality "<text>" --threshold τ` CLI surface that reads `RingBuffer_merge_window` directly. Latter is cleaner and lands counting numbers in the engine itself.
- [ ] **Per-question retention overrides** in `scripts/eval_lookback.py`. Today every question shares the same `--time-window × --capacity` retention. A `psm_overrides:` YAML field would let location_trace questions (whose answer is a *moment* inside a long bucket) score the event interval directly without degrading the rest of the corpus.
- [ ] **Categorical / spatial answer grading.** Three follow-up questions (Fulham q7 river vs railway, Tucson q9 bridge under/over, Palo Alto q7 palms+truck ordering) return deterministic top-1 cells but can't be auto-scored without an OSM-overlay annotation tool. Stand-alone helper outside `libpsm`.

## Extraction Pipeline

Self-contained Python package under `extraction/` that produces the `features.h5` files PSM consumes. Replaces the external pipeline whose machine has gone missing. The C engine stays light; the Python pipeline is an optional sibling consumed only by people producing data.

Schema v1 is the format the existing `features.h5` uses (no file-level metadata, no per-group model attrs). Schema v2 adds explicit `schema_version`, `producer`, `model`, `checkpoint`, `embedding_dim`, `sample_fps`, `normalized`, etc. so a consumer can audit a file. The C ingest treats both as compatible since it only reads dataset arrays, not attrs.

### Phase 1 — Schema v2 + writer + migration

- [x] `extraction/psm_extraction/schema.py` — versioned constants (root + per-group attrs, dataset names, dtypes, expected shapes), plus `ModelGroupSpec` dataclass for the writer's typed surface
- [x] `extraction/psm_extraction/writer.py` — `FeaturesWriter` context manager that emits v2-compliant files (root attrs, sensor groups, model groups), validating dataset shapes/dtypes
- [x] `extraction/psm_extraction/migrate.py` — `migrate_v1_to_v2` adds missing attrs in-place using best-effort defaults for known groups (dino/jepa/clip); idempotent on already-v2 files
- [x] `extraction/psm_extraction/__main__.py` — `python -m psm_extraction migrate <file>` CLI surface
- [x] `extraction/pyproject.toml` — minimal package metadata (h5py + numpy core; pytest dev; clip/aria as optional extras for later phases)
- [x] Round-trip + migration tests under `extraction/tests/` (pytest)
- [x] README §"HDF5 feature schema" — short doc describing v2 root + per-group contracts

### Phase 2 — CLIP runner end-to-end

- [x] `models/base.py` ModelRunner ABC with `model_id`, `checkpoint`, `embedding_dim`, `normalized`, `preprocess`, `patch_grid`, `backend`, `embed_images`, `embed_text`, `close`. CLI knob `--backend {auto,pytorch,mlx,cpu}` lands in `python -m psm_extraction extract`.
- [x] `models/clip_pytorch.py` runner backed by HuggingFace transformers; auto-picks cuda > mps > cpu when `device='auto'`.
- [ ] `models/clip_mlx.py` MLX-native CLIP for Apple Silicon. CURRENTLY A STUB raising NotImplementedError so the auto-pick falls through to PyTorch MPS until the upstream mlx-clip API is verified end-to-end. Tracked as a Phase 2 follow-up.
- [x] `io/video.py` ffmpeg-backed frame reader (lifted from the demo).
- [x] `align.py` GPS interpolation onto frame timestamps via `load_session_track` + `map_frames_to_gps`; synthetic snake-grid fallback for plain videos. IMU interpolation is deferred (the C ingest doesn't consume per-frame IMU snapshots; viz consumers can read `imu/` directly from the canonical sensor group).
- [x] `python -m psm_extraction extract --video data.mp4 --models clip --output features.h5` produces a v2-compliant file consumed unchanged by `psm --search`. Same flow exposed via the refactored `scripts/e5_clip_demo.py` thin shim.
- [ ] Smoke test against a synthetic video fixture (FFmpeg `testsrc`) under both backends, verifying embeddings match within a tight cosine-similarity tolerance. Blocked on the MLX runner.

### Phase 3 — Aria VRS + DINOv3 + V-JEPA 2

- [x] `io/json_sidecar.py` — Aria-style `gps.json` + `imu.json` reader. Picks the largest non-empty stream by default; filters Aria's denormalized pre-fix samples; emits numpy arrays sorted by timestamp. `metadata.json` parser surfaces the `capture_time_epoch` so the orchestrator can convert relative sidecar timestamps to absolute Unix seconds, matching the existing pipeline's output.
- [x] `models/dino_pytorch.py` — DINO runner via `AutoModel.from_pretrained`. Mean-pooled patch tokens for embeddings; last-layer CLS-to-patch attention reshaped to the probed patch grid for `attention_maps`. `model_id` auto-derives `facebookresearch/dinov2` vs `dinov3` from the checkpoint string.
- [x] `models/jepa_pytorch.py` — V-JEPA 2 encoder runner via `AutoVideoProcessor` + `AutoModel` (requires `transformers >= 4.53`). Replicates each ffmpeg-extracted frame across the model's clip window (`fpc`, e.g. 64 for `vjepa2-vitl-fpc64-256`) per the upstream model-card recipe; mean-pooled encoder tokens give the 1024-d embedding. Prediction-map computation is deferred to Phase 4. Compute caveat: each "video" forward pass touches `fpc` frames, so V-JEPA 2 inference is ~`fpc`x more expensive than DINO; pick a sparse `--sample-fps` (0.5–1.0) for multi-minute videos. On macOS the runner intentionally bypasses `decord` (which is unsupported there) by feeding pre-decoded PIL frames directly to `AutoVideoProcessor.videos=`.
- [x] Multi-model orchestration — `ExtractOptions.runners: list[(group_name, runner)]` lets one frame pass populate any combination of `clip`/`dino`/`jepa` groups in the same v2 file. CLI exposes `--models clip,dino,jepa` and `--checkpoint FAMILY:PATH` overrides per family.
- [x] Sensor groups — when `gps.json` / `imu.json` (and optionally `metadata.json`) sit next to the video, the orchestrator writes `gps` and `imu` sensor groups too, so the produced `features.h5` matches the original Aria pipeline's shape end-to-end.
- [x] DINO register-token support — DINOv3 prepends 4 register tokens after the CLS; the runner reads `config.num_register_tokens` and skips them when slicing CLS-to-patch attention and when mean-pooling the embedding. Falls back to brute-forcing common register counts if config doesn't expose the field.
- [x] V-JEPA 2 SDPA OOM cap — `VJEPAPyTorchRunner._safe_batch_size()` clamps batch_size based on `fpc` so the orchestrator's default doesn't trigger a 64 GB attention-matrix allocation. Emits a stderr note when clamping kicks in.
- [x] Sensor groups land independently of track resolution — orchestrator writes `gps` from `gps.json` and `imu` from `imu.json` regardless of whether the per-frame track came from features.h5 or a JSON sidecar. (Bug from initial Phase 3: gps group was only written when track_source itself was JSON.)
- [x] V-JEPA 2 default checkpoint fixed to `facebook/vjepa2-vitl-fpc64-256` in the registry fallback (was a placeholder string in the Phase 3 first cut).
- [x] End-to-end smoke run reproducing the Fulham `features.h5` shape on Apple Silicon (M4): DINOv3 ViT-Large at 30 fps over 27069 frames in 22m09s, attention-distribution parity verified at frame 628 (top-1 fraction ~5.7%, matching the original).
- [ ] `io/aria_vrs.py` — VRS reader behind an optional `[aria]` extra (depends on `projectaria-tools`). Deferred: the user's existing sessions ship JSON sidecars that cover the GPS/IMU path; raw VRS support is only needed for fresh captures.
- [ ] `models/dino_mlx.py` / `models/jepa_mlx.py` — MLX-native runners. Deferred: no MLX-CLIP/DINO/JEPA package detected in the user's env; PyTorch MPS is the auto-pick on Apple Silicon.
- [ ] Pinned checkpoints recorded in HDF5 attrs; refuse to mix groups produced by different versions inside one file. Partial: `checkpoint` is recorded per group; cross-version mixing isn't enforced yet.
- [ ] Synthetic-video smoke test (FFmpeg `testsrc`) producing a v2 file via `clip` runner under PyTorch+CPU — covers the full pipeline in CI without GPU. Phase 2 follow-up still open.

### Phase 3.5 — Engineering hygiene for long runs

After losing ~18 minutes of DINOv3 inference to a poorly-timed `pkill`, shipped a small set of "long-run survival" features so the next time something kills mid-extraction, the cost is bounded.

- [x] `psm_extraction/progress.py` — `stage_banner(stage, msg)` for stage transitions and `make_progress_logger(stage, n_total)` for throttled per-batch progress (~once per 2 s, first/last samples always print). Output to stderr so stdout's JSON manifest still pipes through `jq`.
- [x] `ModelRunner.embed_images(..., progress=callable)` kwarg threaded through every runner — CLIP, DINO, V-JEPA 2 all call it after each batch.
- [x] Frame cache in `io/video.py` — ffmpeg writes a `.extract_manifest.json` recording `(video, sample_fps, frame_count)`. Subsequent runs with matching params skip ffmpeg and reuse the JPEGs. `--force-reextract` bypasses.
- [x] Per-model embedding cache in `extract.py` — after each runner finishes, embeddings (+ attention/prediction maps) save to a hashed `.npz` sidecar in `<output>.parent`. Cache key includes `model_id + checkpoint + video_path + sample_fps + group_name`, so different params can never reuse stale caches. `--force-reembed` bypasses; `--cache-dir` overrides location.

### Phase 4 — Polish

- [ ] V-JEPA 2 `prediction_maps` — currently the runner emits encoder embeddings only. The original Aria pipeline produced 16×16 prediction-error maps via the JEPA predictor head against context+target patch sampling. Re-enables the JEPA prediction-error overlay in `psm-viz`. ~200 LOC + careful reading of V-JEPA 2's loss path.
- [ ] Per-model `--sample-fps` — the orchestrator currently shares one frame-extraction pass across all runners. Adding per-model rates lets a single command produce DINO at 30 fps + JEPA at 0.5 fps in one run, matching the original pipeline's pattern without merging two output files.
- [ ] `add-group` subcommand: append a new model group to an existing v2 file (closes the Level 3 merge with the existing Aria pipeline once a CLIP runner exists)
- [ ] Configurable interpolation method on `align.py` with explicit `@interpolation` attr in HDF5
- [ ] Optional gzip/lzf compression on `attention_maps` / `prediction_maps`
- [ ] Drop or make optional the redundant per-group IMU snapshots (`dino/accel`, `dino/gyro`, etc.); the canonical `imu/` group is enough for downstream consumers
- [ ] Incremental embedding checkpoint (mid-batch) — the per-model `.npz` cache saves only after a runner finishes. Saving every N batches would protect against kills *during* a 22-minute DINOv3 inference, not just between runners.

## Paper Workstream (ECCV 2026)

Engineering tasks needed for the workshop paper. Research direction +
status live in [journal/PAPER.md](journal/PAPER.md); the experiment
specs live in `EXPERIMENTS.md` E10/E11/E12 (plus existing E5/E7/E8).
This section is just for the harness/glue code those experiments need.

### Baselines + sweeps

- [x] `scripts/eval_brute_force_clip.py` — for E11. Embed every frame of a session, rank all frames by cosine against the query, take top-k. Use the same scorer as `eval_lookback.py` so numbers are directly comparable. Landed; ran on Nymeria `shelby_arroyo_act0` (13.4% Hit@5).
- [x] `scripts/eval_sliding_window.py` — for E11. Slide 5-second windows, mean-pool frame embeddings inside each window, rank windows. Same scorer. Script landed with `--clip-checkpoint` threading already in place; Nymeria sweep still pending (compute only).
- [x] `scripts/eval_uniform_sample.py` — for E11. 1-frame-per-time-window sampling, no learned aggregation. Lower-bound baseline. Script landed with `--clip-checkpoint` threading already in place; Nymeria sweep still pending (compute only).
- [x] Hyperparameter sweep loop on top of `eval_bigg_all.sh` — for E12. Adds `H3_RESOLUTION`, `RETENTION` (as `TIME_WINDOW × CAPACITY`), and `EXEMPLARS` env knobs; auto-suffixes the TAG so outputs don't clobber the v2 raw runs. Landed as standalone `scripts/eval_hyperparam_sweep.sh`.
- [x] `scripts/eval_hyperparam_aggregate.py` — small extension to the aggregator that pools across hyperparameter axis and plots single-axis sensitivity curves. The plot is paper Figure F3. Landed as `scripts/eval_hyperparam_plot.py`; F3 SVG at `journal/figures/hyperparam_sensitivity.svg`.

### SLOPER4D street-scale corpus (2026-06-17 pivot)

After the full Nymeria-30 clipL hyperparam sweep returned flat ~2% Hit@5 across every operating point (26/30 sessions ≤9.4m bbox → temporal-localization regime, not spatial), pivoted street-scale story to SLOPER4D + Aria Gen 2 walks. See [journal/PAPER.md](journal/PAPER.md) 2026-06-17.

- [x] `extraction/psm_extraction/io/sloper4d.py` — LiDAR trajectory reader + WGS84 projection at Xiamen University fake origin (commit `866de6d`).
- [x] `scripts/extract_sloper4d_sessions.py` — orchestrator wrapper writing Aria-style `gps.json` sidecar next to MP4 then calling `python -m psm_extraction extract --gps-json …`; cleans up sidecar. Encoder-aware H5 basename (commits `fb5375c`/`acb341a`).
- [x] `scripts/slurm/extract_sloper4d.sbatch` — 5-task array per encoder; drops `seq002_football_001` (room-scale only); submitted clipL + bigG (jobs `8315558` + `8315559`) on 2026-06-17.
- [ ] H3-resolution sweep on `seq009_running_002` (446m bbox, primary street-scale anchor) once extraction lands. Acceptance: Hit@5 at r12 ≥ 2× Hit@5 at r10 for both clipL and bigG. This replicates the Nymeria-street single-session finding (3.2% → 8.9% bigG); failure implies that finding was an N=1 artifact.
- [ ] `query_mode: last_seen` question generator. For each sequence pick N timestamps, look up GPS ground truth, generate "where was C at time t?" questions. GPS-grounded, no manual annotation. This is what makes SLOPER4D annotation-free.
- [ ] Email SLOPER4D authors (Yudi Dai et al.) requesting `001_campus_001` (908 m), `010_park_001` (642 m), `011_park_002` (1,025 m) — three additional street-scale sequences not in the 6 currently public. Adds ~2.6 km of trajectory, triples multi-session street-scale coverage, makes the multi-session claim bulletproof. Worth doing today even if reply is days away — they slot into the same sweep without code changes.



### MLLM in the loop

- [x] `scripts/mllm_client.py` — uniform interface over Gemini 3 Pro API, GPT-5 API, and a locally-served vLLM endpoint (for Llama-3.2-90B-Vision or InternVL3.5). Returns predicted `[t_start, t_end]` intervals from (video_path, question). Hides the API-vs-local-serving difference from the caller. Landed as `scripts/_mllm_client.py`; covers Gemini 3.1 Pro + Claude 4.6 Opus via the internal OpenAI-compat proxy. GPT-5 / open-source vLLM still not wired.
- [x] `scripts/eval_mllm_baseline.py` — for E10. Run `mllm_client` over the (session, question) grid, parse intervals, score with the same IoU scorer. (Vanilla-MLLM baseline — distinct from the PSM→MLLM reranker `eval_psm_mllm.py` that already landed.) Landed 2026-06-18 (commits `7197033`/`6fce5f4`/`d7cb6b7`). Run on the 4 v1 street-scale sequences at K=8: 0.0 % on Nymeria (187q), 6.7–7.1 % on SLOPER4D; PSM clipL r12 at 7.6–30.0 % across the same set — 2.5×–∞× gap. K-sweep on seq009 (K=8/16/32; 64/128 blocked by proxy 500s) shows oracle upper bound climbs to 21.4 % at K=32 while Gemini's pick rate stays flat at 7.1 % — discrimination over uniform-sampled multi-frame baselines degrades as K grows. See PAPER.md 2026-06-18 entries + EXPERIMENTS.md E10 status block.
- [x] Frozen prompting protocol (text + 2-shot exemplars) for interval elicitation. Store as `prompts/grounding_v1.txt`; pin the protocol in PAPER.md once it's frozen so reviewers can audit. Landed inline as `_FROZEN_PROMPT_TEMPLATE` in `scripts/eval_psm_mllm.py`; reranker-only (frame-index style), no 2-shot exemplars and no separate `prompts/` file yet. Extract + add exemplars before E10 ships.
- [x] `scripts/eval_psm_mllm_pipeline.py` — for E5. Run PSM, take top-k `(cell, t_start, t_end)`, sample N evidence frames per candidate, feed (question, evidence frames) to the MLLM, return predicted interval. Score with the same scorer. Landed as `scripts/eval_psm_mllm.py` (commits `471e4ab`/`d051757`).

### Question bank expansion

- [ ] Add 30-60 new questions across the 3 existing sessions, biased toward the under-represented categories (Time Duration, Sequential Action, Spatial Awareness). Target: 50-80 IoU-scoreable questions total. Annotation labor; no scripts needed beyond the existing YAML schema.
- [ ] Encoder-bypass stress test: 5-10 new `query_mode: last_seen` questions in cells the wearer revisited ambiguously. Strengthens v2 §3 if Hit @5 holds; reveals a real limitation if it crashes.
- [ ] One longer session (30-60 min) ingested + annotated. Stresses the bounded-memory claim. Probably the Tucson cycling session if we have a longer take.

### Paper figures

- [ ] F1 architecture diagram (PSM as prefilter). Hand-drawn or draw.io export.
- [ ] F2 retrieval method ablation bar chart. Hand-authored SVG from E11 outputs (matches existing journal/figures style).
- [ ] F3 hyperparameter sensitivity 3-panel plot. From E12.
- [ ] F4 MLLM baseline vs PSM bar chart. From E10.
- [ ] F5 PSM + MLLM reranker results. From E5.
- [ ] F6 memory footprint vs session length. From an existing or new benchmark.
- [ ] F7 qualitative top-5 candidates over a query. Screenshot from psm-viz + annotation.

### Submission housekeeping

- [ ] Anonymize for double-blind: scrub author from journal markdown + PDF metadata; investigate whether commit history needs scrubbing.
- [ ] Pin `requirements.txt` for the paper-time scripts (separate from `extraction/pyproject.toml` which is for the extraction package).
- [ ] `scripts/reproduce_paper.sh` — one-shot script that runs every paper experiment end-to-end from a fresh clone. Required by most workshops' artifact tracks.
- [ ] Tag the submission commit: `git tag eccv2026-submission` once frozen.

## 2026-07-09 12:50 -- 8pp archival squeeze (OpenSUN3D archival + Huawei SpatialMem)

- [x] F2 (PSM-vs-MLLM per-session) figure -> supp multi-corpus (body 11->10pp)
- [x] §5.3 memory/latency/failure-modes -> supp sec:supp-memory-latency
- [x] §5.2 strict-diversity cost + brute-force ceiling -> supp ablations
- [x] Vanilla-MLLM temporal-coverage mechanism + K=32 oracle -> supp mllm-rerank
- [x] Tighten HDD 4-paragraph prose (findings + hdd_combined figure kept in body)
- [x] Demote single-session Nymeria tab:headline -> supp ablations; 2-sentence summary in body
- [x] Compress §7 "What the evidence supports" recap -> 3-sentence synthesis
- [x] Repoint all body refs to supp anchors; 0 undefined refs; both F1 + hdd_combined kept in body
- [x] Pushed 3545b8e + 51120d5 to paper-8pp-cut
- [ ] Confirming build on real ECCV kit (cluster) -- local lmodern build ~9.5pp; p10 ~75% full
- [ ] If still >8pp on real kit: trim §7 follow-up directions + §4 related work + §5.1 setup

## 2026-07-09 20:34 -- Figure rebalance (body 2->4 figs) + prose squeeze

- [x] Verify unused figures: f7_qualitative (submission-quality), f3_multi_corpus_h3 (actually H3-resolution sensitivity, not 13/14 verdict)
- [x] Promote F2 (PSM-vs-MLLM) supp -> body §5.5; trim MLLM prose figure self-explains
- [x] Add F7 (qualitative retrievals) to body §5.2 -- previously unused
- [x] Collapse §5.6 HDD 4 paragraphs -> 2 (figure carries panels)
- [x] Body now F1+F2+F7+hdd_combined; F3 held in supp; pushed 46d2849
- [x] Prose squeeze to fund 4 figs: compress §1 intro, §2 method, §5.1 setup; pushed e019077
- [x] Body ~11->~10pp local (total 18->17); 0 undefined refs; no results lost
- [ ] Real-kit confirming build (cluster) -- ~9pp estimate; if >8pp trim §6 limitations next
- [ ] (open) Confirm OpenSUN3D archival page limit -- if >8pp the squeeze is moot

## 2026-07-09 21:06 -- F7->supp (3 figs) + verified 8pp limit

- [x] Real-kit trace (user): body=10pp with 4 figs; local build agrees (~10pp)
- [x] Chosen fork: F7->supp (keep F1+F2+hdd_combined in body); pushed e073d10
- [x] Trims: §6 limitations, §7 follow-ups, hdd_combined width 1.0->0.82
- [x] VERIFIED via CFP: OpenSUN3D archival = 8pp full papers (refs excluded, supp unlimited); Huawei SpatialMem = submission flag
- [x] F2 aspect 0.43 (wide-short) -> shrinking saves ~nothing; figures tapped out
- [x] §5.5 vanilla-MLLM body prose trimmed (mechanism already in supp); pushed 1ef8764
- [x] Result: body ~9pp local (float-quantized; supp starts p10). Still 1pp over 8pp.
- [ ] DECISION PENDING: low-damage levers exhausted. To reach 8pp: (a) F2->supp too (2 figs F1+hdd, clean 8pp) OR (b) demote a ~1pp §5 results block (keep 3 figs). Confirm with real-kit build first.

## 2026-07-09 21:56 -- Reviewer cut list (keep 3 figs) + git reconcile

- [x] Applied prioritized cut list: §7 evidence recap -> 3 sentences (+ fixed §4.3,§4.3 dup ref); §6 limitations 4->2 paragraphs; §5.2 crossover para -> 1 sentence; §2.4 novelty sentence deleted; §2.5 encoder sizes -> footnote; abstract trimmed
- [x] Reconciled divergence: cluster 1be0bf2 (F2->supp, 2 figs) vs review (keep 3 figs). Per user: kept F2 in body; merged 1be0bf2 into history then reverted its F2->supp
- [x] Verified: F2 single (body, not supp); 3 body figs (F1+F2+hdd_combined); 0 undefined/duplicate refs; pushed b350cae
- [x] Body ~9.5pp -> ~8.3pp local (supp p10->p9, total 17->16); conclusion bleeds ~9 lines onto p9 on loose lmodern build
- [ ] Real-kit confirming build (cluster): 3-fig body was 9pp pre-cuts; cut list frees ~1pg -> expect ~8pp on tighter Times kit
- [ ] If ~1 line over on real kit: reserve cut = drop §7 follow-up directions / tighten §6 evaluation-scope
- [ ] Table 1 promotion to body (review minor #3): held -- fights 8pp; left in supp

## 2026-07-09 22:32 -- Micro-cuts A-E (final ~0.5pg)

- [x] A: abstract generalization sentence compressed
- [x] B: §4.4 'not a plateau' restatement dropped
- [x] C: Python-reimpl detail (2.4e-5) MOVED to supp HLL §A.5 (was not there -> moved, not deleted)
- [x] D: §3 R-tree/primitives disclaimer compressed (drops sole guttman cite)
- [x] E: §4.4 AUC-vs-hit-rate justification -> footnote
- [x] Fallback: §7 follow-ups + MIT line compressed
- [x] Local p9 conclusion bleed ~9 -> ~3 lines; 0 undefined/dup refs; no evidence lost; pushed eee2be8
- [ ] Real-kit confirming build: expected to clear 8pp (~3 loose-lmodern lines should absorb on tighter Times); if ~1 line over, one micro-trim remains

## 2026-07-09 23:34 -- Internal review (Kamek) fixes; decision: submit non-archival as-is

- [x] Decision: OpenSUN3D non-archival + SpatialMem flag; NO 3D experiment / venue pivot (accept topical-fit risk)
- [x] Fix '8x smaller' claim (abstract + §4.2): 8x = R=128-vs-R=1024 reservoir, not vs brute-force; lead with session-length-invariance
- [x] cap=1 R-non-monotonicity: confirmed Table 1 is single default seed (sweep_per_cell_cap.py); footnoted as reservoir noise; 30-session sweep monotone in R
- [x] Honest rounding: rerank mIoU +37% -> +36.5% (supp §A.6)
- [x] §5.1: note no purpose-built outdoor egocentric look-back QA benchmark exists
- [x] Surface sliding-window-CLIP-edges-PSM (+1.6pp, full N-frame) in body §4.2
- [x] Page-neutral (conclusion p8, supp p9); 0 undefined/dup refs; pushed d650095
- [ ] Deferred to future main-venue version: genuine look-back QA annotation; scene-graph/SLAM baseline; on-device profiling; compose PSM w/ memory-augmented video-LLM; 3D (Aria MPS xyz) indexing

## 2026-07-09 23:51 -- Venue switch to Wearable AI (ECCV 2026, 14pp); full-depth restore

- [x] Assessed Wearable AI workshop CFP: native fit (wearable egocentric memory/QA), 14pp both tracks, deadline Jul 25, ECCV format, archival+non-archival; no 3D-fit risk
- [x] Decision: switch OpenSUN3D->Wearable AI; restore to full ~12-14pp
- [x] New branch paper-wearable-14pp (8pp version preserved on paper-8pp-cut)
- [x] Restored to body: Table 1 (headline), F7 (qualitative), 5.3 memory/latency/failure, 5.4 multi-corpus discussion + Table 2, 5.5 MLLM mechanism, 5.6 detail, un-abbreviated abstract, limitations 2->4 paragraphs
- [x] Kept all internal-review honesty fixes (8x claim, cap=1 seed footnote, +36.5%, proxy, sliding-window)
- [x] Body ~8pp -> ~12pp local (supp starts p12); 0 undefined/duplicate refs; no evidence lost; pushed 8eccdff
- [ ] Optional further depth to ~13-14pp: promote 30-session monotonicity and/or F6 crossover to body; restore fuller method prose
- [ ] Real-kit confirming build; submit non-archival (or archival) + no 3D pressure. Deadline Jul 25

## 2026-07-10 00:33 -- Two WearablesAI reviews (both Accept); polish pass

- [x] Both reviews: Accept (4/5 and 4.5/5 relevance); all prior honesty fixes confirmed applied
- [x] Author selected: 30-session CIs, restructure abstract, genuine look-back QA (NOT on-device profiling)
- [x] #2 abstract restructured: lead w/ wearable problem + O(area)-not-O(time) + one headline number, softened jargon, section signposts; page-neutral (~12pp); pushed b045a04
- [x] #1 CIs PREPPED: aggregate_cap_sweep_30.py emits mean+/-std, 95% CI, paste-ready supp §A.1 sentence
- [x] #3 genuine-QA PREPPED: journal/genuine_lookback_qa/{template_questions.yaml,PROTOCOL.md}
- [ ] BLOCKER (cluster-only): #1 + #3 need data at /checkpoint/.../nymeria_atomic (not in clone; captures/ empty; no committed per-session numbers). No fabrication.
  - #1: run scripts/multisession_cap_sweep_30.sh then aggregate_cap_sweep_30.py -> send CI line -> I insert into §A.1
  - #3: annotate 15-20 genuine questions on cluster footage -> run eval_lookback.py -> send eval_genuine_*.json -> I write §4.x paragraph
- [ ] Deadline Jul 25; paper is Accept-grade as-is on paper-wearable-14pp

## 2026-07-10 04:40 -- Android POC assessment + fine-detail fixes (merged w/ cluster)

- [x] Android on-device: YES feasible/recommended (Snapdragon/Tensor ~ Aria/Quest silicon; better proxy than RPi). POC = self-contained benchmark_spatial_memory (no HDF5/ffmpeg/encoder; core-only build needs libh3). Post-submission-safe (camera-ready/poster), do NOT block Jul 25.
- [x] Guide: journal/on_device/ANDROID_BENCH.md (Termux + NDK paths, peak-RSS, §4.3 fill-in)
- [x] Device-independent fixes: §2.3 memory 0.44->0.38 (reservoir; 0.44 total); §2 'Memory bound' formal statement; 83%->84% (abstract+§4.2); Table 2 acceptance criterion in caption
- [x] Merged cluster fixes (18de6f3): abstract knob semantics 'dialing recall against per-query retrieval fan-out' (supersedes my nit -- per_cell_cap is a fan-out knob, not state); §1 drop chen2024vljepa; supp HLL 3.2%-bound reword. Conflict resolved (kept both correct parts). + d7f7993 PAPER_INSERTS.md
- [x] Build clean (~12pp, 0 undefined/dup, no conflict markers); pushed 2e28ad6
- [ ] Android POC run on author's phone -> send µs/MiB -> I insert §4.3 row (post-submission ok)
- [ ] HELD (need input): EgoVLP/TimeChat/EgoTimeQA citations (verified bib); §5.4 cap=1 456us vs cap=K 153us latency ordering (flagged by cluster commit for verification); optional aggressive abstract re-cut

## 2026-07-10 05:02 -- Citations + abstract denominator fix (pulled latest first: already at 22a25d8)

- [x] Pulled latest: already up to date (22a25d8)
- [x] Abstract '84% (21/187 vs 25/187)' -> '21 of a brute-force CLIP oracle's 25 Hit@5 hits (84%)' (removes the 83.6%-rate ambiguity)
- [x] Added EgoVLP (lin2022egovlp, NeurIPS 2022) to §3 egocentric para; TimeChat (ren2024timechat, CVPR 2024) to §3 memory-augmented para; verified bib metadata
- [x] EgoTimeQA NOT added (not standalone -- CG-Bench split; grounded-QA line already cited via di2024grounded)
- [x] 37 refs, 0 undefined citations; build clean (~12pp); pushed 4f84945
- [ ] Cluster-only verifications (data, not text; per advisor notes): §4.3 cap=1 456us vs cap=K 153us ordering; Hillsdale6 13.8% vs Fig.3 14% label
- [ ] Android on-device POC (post-submission): run journal/on_device/ANDROID_BENCH.md -> send numbers -> §4.3 row
- [ ] Abstract aggressive re-cut: HELD (advisors agree don't destabilize)
- [ ] Paper deemed READY for Jul 25 submission by both advisor notes

## 2026-07-10 05:06 -- cap=1 vs cap=K latency: code-verified REAL (not a swap)

- [x] Traced SpatialMemory_query_similar: dominant per-query cost = fill_window_for_top_k/RingBuffer_merge_window, run once per UNIQUE cell in top-K (dedup lines 458-466)
- [x] cap=1 -> K distinct cells -> K merge-window passes; cap=K -> shared cells -> fewer passes (cosine identical). 456>153us is mechanistically correct, ~3x matches K=5 vs ~1.6 unique cells. NOT a transcription swap.
- [x] Added one-clause explanation to §4.3; measured numbers unchanged; build clean (~12pp); pushed acc5996
- [ ] Hillsdale6 13.8% vs Fig.3 14% label: still a cluster data-verify (per-session JSON); consistent if raw count rounds to 14%
- [ ] Android on-device POC (post-submission): journal/on_device/ANDROID_BENCH.md
- [ ] Paper READY for Jul 25 (both advisor notes); citations + all correctness items in

## 2026-07-10 17:49 -- Anonymized code link + pulled bib cleanup

- [x] Pulled latest: ff to 9b926e1 (bib drop guttman1984/chen2024vljepa -> 35 refs)
- [x] Conclusion footnote: replaced 'URL withheld' with anonymized double-blind link https://anonymous.4open.science/r/probabilistic_spatial_memory-3C17/ ; kept reproduce_paper.sh pointer; build clean (~12pp, 0 undefined); pushed e031803
- [ ] Author: confirm the 4open.science repo actually mirrors the code incl. scripts/reproduce_paper.sh and renders (it anonymizes a real GitHub repo)
- [ ] Android on-device POC (post-submission): ANDROID_BENCH.md -> numbers -> §4.3 row

## 2026-07-10 18:22 -- Reviewed d4dd7e8: benchmark compile+run VERIFIED; fixed regime note

- [x] Pulled d4dd7e8 (per_cell_cap arg + peak_VmHWM in benchmark_spatial_memory.c; S22 walkthrough)
- [x] Code review: edits correct, default-preserving; API usage matches query_similar signature + argv pattern
- [x] COMPILE+RUN VERIFIED locally: built libh3 (cmake from uber/h3) + fetched PDS submodule (ssh:443, pinned 309dc39); core-only build compiled+linked+ran. per_cell_cap arg, per_cell_cap= print, peak_VmHWM all work; default unchanged. Compile-unverified caveat CLOSED.
- [x] FINDING: synthetic bench (1024 sparse tiles x 4 exemplars) shows cap=1 FASTER (~1560us) than cap=K (~2106us) -- OPPOSITE §4.3 Nymeria (cap=1 slower via merge-window). Different regimes, both correct.
- [x] Fixed ANDROID_BENCH.md: removed wrong 'cap=1>cap=K' sanity note; added regime note; reframed §4.3 insert to absolute on-device query us + RSS (not a cap=1/K split). Pushed 0ed0f31.
- [x] Cleaned up temp deps (submodule dir restored empty, /tmp removed); container run numbers NOT for paper
- [ ] Author: run on S22 + host (identical synthetic bench) -> send stdouts -> I fill §4.3 wearable-feasibility line (absolute us + RSS)

## 2026-07-10 20:43 -- S22 on-device benchmark RUN + MEASURED (Path B, NDK cross-compile)

- [x] Confirmed device: SM-S901B = Galaxy S22 **global (Exynos 2200)**; arm64-v8a; CPU7 part 0xd48 = Cortex-X2 -> taskset 80 correct. No Termux/compiler on phone + no NDK on host -> chose Path B (NDK cross-compile, drive over adb).
- [x] Installed NDK r29 (brew cask; must run from a real shell -- /opt writes blocked in sandboxed tool shell even un-sandboxed). Toolchain aarch64-linux-android34-clang (clang 21).
- [x] Cross-compiled libh3 4.5.0 **static** for arm64-v8a (NDK cmake toolchain, android-34); verified elf64-littleaarch64. Then bench_psm: ARM aarch64 PIE, -mcpu=cortex-x2, static libh3, only bionic libc/libm/libdl. Pushed to /data/local/tmp; runs; peak_VmHWM prints.
- [x] BUG in ANDROID_BENCH.md compile one-liners: **missing vendor include dirs** (-Ivendor/.../{lib,hyperloglog,bloom_filter}); without them hll.c #include "hash.h" fails. Fixed all 4 compile commands (Path A/B + Step 2 + host baseline) + added $VINC definition. Root-caused on host, would've broken the S22 build too.
- [x] Full run: 200000/1024/200000, taskset 80 (PSR=7 verified on-device), 10 reps/cap, rep1 dropped (n=9). **No thermal throttling** (query_similar flat across reps).
- [x] **S22 numbers (median):** ingest 0.86 us/frame; query_intervals 104 us; query_similar 1163 us (cap=1) / 1746 us (cap=10); peak RSS 20.2-20.3 MiB. **Host baseline (Apple Silicon):** ingest 0.33 us/frame; query_intervals 30 us; query_similar 366/548 us. **host->ARM ~2.6-3.4x.** cap=1<cap=10 on both (regime note holds).
- [x] Docs: results saved to journal/on_device/results_s22.md + results_host_baseline.md; ANDROID_BENCH.md placeholder §4.3 block replaced with measured table + ready-to-insert §4.3 line + as-executed reproduction appendix (exact NDK/cmake/adb commands).
- [ ] Author review: insert the §4.3 wearable-feasibility line (drafted in ANDROID_BENCH.md) + soften the four "on-device timing unmeasured" hedges (§1, §4.3, §6, §7) to "measured on Galaxy S22". Post-submission (camera-ready/poster) -- do NOT block Jul 25.
- [ ] Raw stdouts in /tmp/{s22_out,host_baseline}_out.txt (not committed; numbers captured in the results md files).

## 2026-07-10 20:43 -- Added NWM citation to §3 (learned-world-model contrast)

- [x] Verified metadata (arXiv abstract): Bar, Zhou, Tran, Darrell, LeCun, "Navigation World Models", CVPR 2025, arXiv:2412.03572 (facebookresearch/nwm). Matches advisor table.
- [x] bib: added bar2025nwm @inproceedings (CVPR 2025, arXiv note) after plizzari2025spatial in "Ego streaming memory + grounding".
- [x] §3 Related Work (section_4_related.tex), egocentric paragraph: added the learned-world-model contrast -- NWM = implicit/learned/unbounded autoregressive generative model; PSM = explicit/non-learned O(area)-bounded index (dual). Sharpens positioning for FAIR/egocentric reviewers.
- [x] Build VERIFIED: latexmk full build exit 0, bar2025nwm resolved in main.bbl, 0 undefined citations, 38 refs (was 37). Renders "Bar, A., Zhou, G., ...: Navigation world models. In: CVPR (2025), arXiv:2412.03572". Page-neutral in main body.
- [x] Assessed the other 4 advisor links: AM-RADIO/E-RADIO (2312.06709) marginal (only fits on-device/future-work efficient-encoder swap; NOT added); CALICO + OpenTAD not relevant (no memory/spatial/retrieval overlap). EgoVLP (2206.01670) already cited (lin2022egovlp).

## 2026-07-11 04:31 -- On-device S22 numbers inserted into paper (reviewers' #1 gap CLOSED)

- [x] Pulled 2d529fa (results_s22.md, results_host_baseline.md, ANDROID_BENCH.md results/drop-in, TODO)
- [x] §4.3: inserted measured Galaxy S22 (Exynos 2200) line -- 0.86us/frame ingest, 104us location, 1.16/1.75ms semantic (cap=1/K), 20.2MiB RSS, 2.6-3.5x host->ARM; noted synthetic cap=1-faster vs dense Nymeria bench
- [x] Softened 3 hedges (§1/§6/§7): 'on-device unmeasured' -> 'measured on Galaxy S22'; KEPT honest caveat that the CLIP encoder (heavy component) is still unmeasured on-device (only engine core measured)
- [x] Numbers verified against committed results_s22.md; body ~12pp (within 14pp); 0 undefined refs; pushed bf4fa10
- [ ] Author: final real-kit build + OpenReview submission (non-archival + SpatialMem). All strengtheners now in.

## 2026-07-11 04:31 -- Efficient-encoder future-work note (§6) + resolved rebase

- [x] Resolved `git pull` rebase conflict in TODO.md (upstream S22-paper-insert entry vs local NWM entry): kept BOTH, chronological order. .bib/.tex applied clean (NWM independent of §4.3 edits). Rebase continued; branch linear (7900874 on 8b42e06).
- [x] §6 "Evaluation scope": where the CLIP encoder is named as the remaining unmeasured on-device component, added the efficient-encoder drop-in -- MobileCLIP (vasu2024mobileclip) / AM-RADIO/E-RADIO (ranzinger2024radio) -- since PSM is encoder-agnostic. Ties the on-device story to a concrete mitigation.
- [x] bib: added vasu2024mobileclip (CVPR 2024, arXiv:2311.17049) + ranzinger2024radio (CVPR 2024, arXiv:2312.06709); author lists verified vs arXiv.
- [x] Build VERIFIED: latexmk exit 0, both resolve in main.bbl, 0 undefined citations, 40 refs (was 38). Page-neutral in body.

## 2026-07-11 06:12 -- Two final reviews (both "ship it"); addressed the one flag

- [x] Review flag: 2.6-3.5x on-device ratio was vs host SYNTHETIC bench, but body only showed Nymeria host (456/153us). Fixed §4.3: stated host synthetic figures (0.33us/frame, 30us, 0.37/0.55ms) explicitly, disambiguated from Nymeria. Verified vs results_host_baseline.md. Pushed af6b9b0.
- [x] Confirmed anonymized code link IS in paper (conclusion footnote, 4open.science) -- both reviews verified it; not missing.
- [x] Hillsdale6 13.8% vs 14%: cosmetic rounding (4/29), already code-verified; both reviews call it non-blocking. No change.
- [x] Build clean (0 undefined, body ~12pp <=14pp, 40 refs)
- [ ] Author: final real-kit build + OpenReview submission (non-archival + SpatialMem), Jul 25. Both reviews: ready to ship.

## 2026-07-11 06:26 -- Reference audit follow-through (40 refs)

- [x] Stripped leaked reading-note 'note' fields (render in splncs04): InfiniPot-V (94% mem), SAVEMem (+10pp OVO-Bench), TGLG (editorial), HERMES (redundant). Entries now standard venue/year.
- [x] [37] Weller verified REAL: arXiv:2508.21038 (confirmed via search); added arXiv note.
- [x] Build clean: 0 undefined, 40 refs, leaked notes gone from bbl; pushed eb211da
- [ ] AUTHOR MUST CONFIRM existence (I cannot verify 2026 papers): [38] SAVEMem (arXiv:2605.07897), [40] HERMES (ACL 2026, Zhang et al.). If these came from auto-complete/search rather than papers you read, verify before submission -- fabricated cites are a credibility risk.
- [x] Hillsdale6 13.8%/14% cosmetic (prior); anonymized code link present (prior)

## 2026-07-11 18:12 -- FULL author-level verification of all recent refs (integrity)

- [x] User confirmed [38]/[40] exist via links -> but our bib had WRONG authors on both. Verified ALL recent (2024-2026) cited refs against arXiv/DOI/proceedings.
- [x] FABRICATED author lists FIXED (5): SAVEMem, HERMES, InfiniPot-V (4x Kim->Minsoo Kim/Kyuhong Shim/Jungwook Choi/Simyung Chang), VideoRAG (Cho/Seo/Seo -> Baek Jinheon/Hwang Sung Ju), LookOut (phantom Engelmann removed -> Pan/Harley/Liu/Guibas)
- [x] TITLE errors FIXED (2): aguerrebere ('Vector Quantization'->'Quantization'); SAVEMem (dropped fake 'SAVEMem:' prefix)
- [x] VERIFIED CORRECT (no change): MovieChat, MA-LMM, Flash-VStream, LifelongMemory, di2024grounded, SLOPER4D, Plizzari, EgoMask, ESOM, Arora, VidEgoThink, Nymeria, SigLIP2, grounded-multihop + session-verified (EgoVLP, TimeChat, NWM, AM-RADIO, MobileCLIP, Weller)
- [x] Classics/org untouched (CLIP, HLL, reservoir, Ego4D, DiskANN, SPANN, HDD, Aria, LAION, Gemini, Aria-Gen2)
- [x] 0 undefined, 40 refs, build clean (local accsupp.sty stub restored); pushed b7dcf97
- [ ] Author: this was a real integrity save (5/~20 recent refs had hallucinated authors). Bib now verified. Final real-kit build + submit.

## 2026-07-12 21:47 -- Independent audit: 5 more ref fixes + track correction

- [x] engel2024aria: FABRICATED authors fixed (Kang,Kiran->Somasundaram,Kiran; Gupta,Sach[phantom]->Goesele,Michael). I had wrongly skipped Aria as a 'classic'.
- [x] pan2025lookout: RESTORED Engelmann,Francis (3rd author in cited ICCV-2025 proceedings; DBLP-confirmed). My 07-11 removal (based on arXiv preprint's 4-author list) was wrong for the cited venue.
- [x] ren2024timechat: Sun,Xiao->Sun,Xu (arXiv-confirmed)
- [x] zhang2024flashvstream: was mixing 2024 'Memory-Based' preprint (7 authors) with ICCV-2025 'Efficient' venue -> made consistent as 2024 arXiv preprint (2406.08085)
- [x] aria-gen2-pilot: year 2024->2025 (arXiv:2510.16134)
- [x] 0 undefined, 40 refs, build clean; pushed 1ca2b39
- [x] CORRECTION: venue is WEARABLE AI WORKSHOP @ ECCV 2026 (archival OR non-archival). 'Huawei SpatialMem' was OpenSUN3D's sub-track (abandoned venue) -- does NOT apply here. Prior 'non-archival + SpatialMem' notes were a stale carryover error.
- [ ] Author 30-sec check: F2 legend -- confirm the 'vanilla Gemini K=8' bars plot Gemini's actual pick-rate, not the any-of-K coverage oracle (exemplar_hit_rate_at_5); §4.5 sharply distinguishes them (7.1% pick vs 21.4% oracle).
- [ ] Committed main.pdf is stale (07-10, 16pp); regenerate from the real kit for the actual upload.

## 2026-07-13 06:11 -- F2 mislabel fixed (Option 1: relabel as any-of-K coverage)

- [x] Confirmed: F2 red bar = exemplar_hit_rate_at_5 = any-of-K coverage oracle (21.4%), was mislabeled 'vanilla Gemini' (contradicting §4.5's oracle 21.4% vs pick 7.1%)
- [x] Fix (Option 1, relabel): §4.5 + F2 caption now say 'any-of-K coverage' (upper bound on Gemini's single pick); caption cites 7.1% pick vs 21.4% oracle
- [x] plot_f2_psm_vs_mllm.py: docstring/title/legend relabeled -> so regen bakes correct in-figure label
- [x] 0 undefined, body p12; pushed faedd7f
- [ ] MUST regen F2 on cluster: `python scripts/plot_f2_psm_vs_mllm.py --out journal/figures/f2_psm_vs_mllm.svg` (+ svg->pdf). The figure PDF's baked legend still reads 'vanilla MLLM'; caption/text already correct, so regen closes the last gap.

## 2026-07-13 06:17 -- Abstract S22 clause + F2 legend precision (reviewer re-verify green)

- [x] Reviewer confirmed F2 code-level (any-of-K coverage) -- matches; (a) §4.5/caption/legend relabel was already in faedd7f (reviewer saw pre-faedd7f state)
- [x] Abstract: added compact '(sub-2 ms/query, ~20 MiB on a Galaxy S22)' parenthetical on the per-cell-state clause -- on-device headline up front, no new sentence
- [x] plot_f2 legend refined 'Gemini K=8 coverage' -> 'uniform K=8 coverage' (per reviewer, more precise)
- [x] 0 undefined, 40 refs, body p12; pushed d2e3d1a
- [ ] STILL PENDING (cluster): regen F2 (python scripts/plot_f2_psm_vs_mllm.py --out journal/figures/f2_psm_vs_mllm.svg + svg->pdf) so the figure PDF's baked legend updates from 'Vanilla Gemini' to 'uniform K=8 coverage'. Text/caption already correct.
