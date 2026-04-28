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

Context: a forthcoming NeurIPS 2026 streaming egocentric memory benchmark (the "Localization Paradox benchmark" after its headline finding) exposes models' failure to return supporting `[t_start, t_end]` intervals for look-back questions — frontier MLLMs score near-zero `mIoU` despite respectable semantic accuracy. PSM's H3-indexed ring-buffered memory is a natural substrate for closing that gap. These items add the minimum primitives needed to emit intervals and retrieve exemplars; experiments E5-E7 in `EXPERIMENTS.md` consume them.

- [x] Retain `(t_min, t_max)` per ring-buffer bucket alongside the HLL sketch — enables returning `[t_start, t_end]` candidate intervals; cost ~`16B × capacity × tile_count`
- [x] Reservoir-sampled per-tile exemplar embeddings (configurable `N` per tile) — enables k-NN retrieval against past observations for "visual detail recall" and "last seen" queries
- [x] Expose `SpatialMemory_query_intervals(lat, lng, k_ring, out_tuples)` returning top-k `(cell, t_start, t_end, count)` tuples over the H3 neighborhood
- [x] `psm --last-seen lat,lng --k-ring N --top N` CLI surface + JSON output (`"mode": "last_seen"` discriminator; `schema_version` unchanged at 1)
- [x] Benchmark scenario in `benchmarks/benchmark_spatial_memory.c`: "location-trace query latency" over a populated session — first-class measurement for E7
- [x] `SpatialMemory_query_similar(query, dim, k_ring, center, out)` — rank tiles by cosine similarity of the best exemplar; `psm --search <bin>` / `--center LAT,LNG` / `--exemplars N` CLI; benchmark scenario `query_similar` (E5's text-query adapter now has a concrete backend to target)

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

### Phase 4 — Polish

- [ ] V-JEPA 2 `prediction_maps` — currently the runner emits encoder embeddings only. The original Aria pipeline produced 16×16 prediction-error maps via the JEPA predictor head against context+target patch sampling. Re-enables the JEPA prediction-error overlay in `psm-viz`. ~200 LOC + careful reading of V-JEPA 2's loss path.
- [ ] Per-model `--sample-fps` — the orchestrator currently shares one frame-extraction pass across all runners. Adding per-model rates lets a single command produce DINO at 30 fps + JEPA at 0.5 fps in one run, matching the original pipeline's pattern without merging two output files.
- [ ] `add-group` subcommand: append a new model group to an existing v2 file (closes the Level 3 merge with the existing Aria pipeline once a CLIP runner exists)
- [ ] Configurable interpolation method on `align.py` with explicit `@interpolation` attr in HDF5
- [ ] Optional gzip/lzf compression on `attention_maps` / `prediction_maps`
- [ ] Drop or make optional the redundant per-group IMU snapshots (`dino/accel`, `dino/gyro`, etc.); the canonical `imu/` group is enough for downstream consumers
- [ ] Incremental embedding checkpoint (mid-batch) — the per-model `.npz` cache saves only after a runner finishes. Saving every N batches would protect against kills *during* a 22-minute DINOv3 inference, not just between runners.
