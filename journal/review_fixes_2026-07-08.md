# Codebase Review & Fixes — 2026-07-08

Full-codebase review of `probabilistic_spatial_memory`, followed by same-day
application of every verified finding. This document is the durable record of
what was reviewed, what was fixed, and how it was verified. The condensed
checklist lives in [`TODO.md`](../TODO.md) → "Review 2026-07-08".

## Method

- **Review:** 10 subsystem lanes (core engine, HDF5 ingest, CLI/runtime,
  viz render, Python extraction core, model runners, dataset IO readers,
  eval scripts, docs consistency, build/test/CI) reviewed in parallel.
- **Adversarial verification:** every raw finding was handed to an independent
  verifier instructed to *refute* it by re-reading the actual code, and to
  default to REJECTED unless the defect and its failure scenario could be
  concretely confirmed. Findings were re-anchored to correct lines/severity.
- **Fixes:** applied in parallel across lanes with strictly **disjoint file
  ownership** so concurrent edits could not collide. Each fix was minimal and
  style-matching; no opportunistic refactors.
- **Scope guard:** per the repo's doc-placement policy, `README.md` /
  `EXPERIMENTS.md` / `TODO.md` stay at root; this longer writeup lives in
  `journal/`.

### Review outcome

| | Count |
|---|---|
| Raw findings | 56 |
| Survived verification | 50 (36 confirmed, 14 plausible) |
| Rejected as false positives | 6 |
| Fixed this session | all 50 verified findings + doc sync |

## Verification (post-fix)

All green, re-run after every fix landed:

- **C build:** `make clean && make all` — clean under `-Wall -Wextra -Werror`, exit 0.
- **C tests:** `make test` — 17 test binaries (incl. the new `test-imu-processor`), exit 0.
- **Python tests:** `pip install -e extraction[dev] && pytest extraction/tests` in a fresh venv — 59 passed, exit 0.
- **CI sanity:** `ci.yml` validated as YAML; new benchmark build step compiles (`benchmark_spatial_memory`, `benchmark_nymeria_psm_query`); `.gitignore` change verified to newly-ignore no tracked file and keep `requirements-paper.txt` trackable.

Diffstat: **28 code files changed (+445 / −99)**, plus **1 new test file**
(`tests/test_imu_processor.c`) and 3 docs updated.

---

## Fixes by severity

### High

| File | Defect | Fix |
|---|---|---|
| `src/viz/progress_bar.c` | VBO allocated for `12*6` floats but `ProgressBar_draw_start_overlay` uploads `30*6` → `glBufferSubData` rejected with `GL_INVALID_VALUE`; the "press space to start" splash drew stale/garbage geometry. | Size the VBO for the largest consumer (`30*6*sizeof(float)`); covers all three draw paths. |
| `extraction/psm_extraction/models/dino_pytorch.py` | `AutoModel.from_pretrained(..., output_attentions=True)` without `attn_implementation="eager"`; on the project's `transformers>=4.40` pin the SDPA default returns `attentions=None`, silently dropping attention maps and `patch_grid`. | Pass `attn_implementation="eager"`; add a config-derived `patch_grid` fallback + token-count validation so it is not silently `None` when attentions are unavailable. |
| `scripts/eval_mllm_baseline.py` | Exemplar hit scored only from the top-1 pick (`if rank == 0`) but reported under `exemplar_hit_rate_at_5` — a Hit@1 mislabeled @5, not comparable to every other scorer (which ORs over top-k). Depressed the vanilla-MLLM number anchoring the paper's PSM-vs-MLLM gap. | OR the exemplar hit over **all** top-k predictions, matching `_eval_common.summarize_question` / `eval_lookback.py`. Summary key name kept (external consumers read the literal string). |
| `.github/workflows/ci.yml` | Every job installed only `h3 hdf5`, but `make test` compiles/links GLFW+OpenGL viz tests (`test-screenshot`, `test-viz-debug-hud`, `test-gps-trace`) → CI could never pass on a clean runner. | Add `glfw` (brew) / `libglfw3-dev` (apt) to all jobs. |

### Medium

| File | Defect | Fix |
|---|---|---|
| `src/core/spatial_memory.c` | `SpatialMemory_advance_to_timestamp` looped once per `time_window` from the anchor with no cap — a large timestamp gap spun thousands of redundant window rotations (an HLL alloc/free per tile each). | Cap physical advances at ring capacity (extra advances are no-ops since every slot is already reset); still move the anchor to the exact grid position. |
| `extraction/psm_extraction/extract.py` | For VRS/EgoExo inputs, model-group timestamps rebased to `t=0` while `--gps-json` sensor groups were written at absolute Unix — two groups on clocks ~1.7e9 s apart despite a "shared clock" comment. | Introduce an explicit `rebase_origin`; co-locate JSON sensor groups on the model group's origin (no-op for MP4). **Partial:** exact only when the sidecar clock coincides with the VRS device clock; the residual is documented in-code. |
| `extraction/psm_extraction/io/json_sidecar.py` | `read_gps_json` / `read_imu_json` picked the largest stream by raw sample count *before* validity filtering, so a large all-zero/pre-fix indoor stream could win and the function raised "no valid GPS fixes" even when a smaller sibling had real fixes. | Select the stream with the most *valid* samples. |
| `.github/workflows/ci.yml` | No Python job → extraction pytest never ran in CI. | Added an `extraction` job (`setup-python` → `pip install -e extraction[dev]` → `pytest`). |
| `tests/test_imu_processor.c` (new) + `Makefile` | The IMU dead-reckoning integrator (`src/viz/imu_processor.c`) had zero test coverage. | New pure-math test (heading integration, dead-reckoning displacement, dt-gating boundary); `test-imu-processor` target wired into `test` and `.PHONY`. |
| `extraction/psm_extraction/io/nymeria_narration.py` | Narration rebased against SLAM `trajectory_t0` while `features.h5` frames rebase against `rgb_t0` → intervals offset by `(rgb_t0 − slam_t0)`; docstrings overclaimed alignment. | Removed overclaims; documented the offset; exposed `trajectory_t0_sec` for downstream re-alignment. **Partial:** a full in-module rebase needs the RGB-frame origin, which lives in the extractor. |

### Low

Core engine:
- `src/core/spatial_memory.c` — guard `size_t` overflow in `per_cell_cap` scratch sizing (adversarial input).
- `src/core/tile.c` — advance `exemplar_seen` only after a successful encode (Algorithm-R bias on encode failure); sampling probability preserved bit-for-bit on the success path.
- `src/core/ring_buffer.c` — consistent NULL-slot guard in `RingBuffer_merge_window`.
- `include/core/spatial_memory.h` — corrected `query_similar` return-value doc and the stale `SpatialMemorySimilar` "single winner per tile" comment.

Ingest:
- `src/ingest/ingest.c` — backward in-range GPS query resets the cursor (was forward-only, silently extrapolating + returning success); reject zero-width embedding datasets; document the non-decreasing-timestamp precondition (`include/ingest/ingest.h`).

CLI / viz:
- `src/main.c` — explicit `-g` no longer overridden by a positional; explicit `--exemplars 0` honored (`has_group` / `has_exemplar_capacity` flags, mirroring `has_seed`/`has_center`).
- `src/viz/viz_config.c` — `hex_extrude_scale` parser rejects trailing garbage / errno (matches sibling parsers).
- `src/viz/viz_main.c` — NULL-check `malloc` in `find_file_in_dir`.
- `src/viz/shader.c` — check `ftell` return before allocating (avoids silently-empty shader source).

Python models / IO:
- `models/clip_pytorch.py` — replace tensor-truthiness `or` chain with explicit None checks; probe `text_config.max_position_embeddings` instead of hard-coded 77.
- `models/jepa_pytorch.py` — `embed_images` default `batch_size` 4 → 16 (matches the ABC and siblings).
- `__main__.py` — honor the previously-dead `--group` flag for single-model extraction.
- `migrate.py` — add `siglip`/`longclip` to `KNOWN_GROUP_DEFAULTS` (were skipped while still stamping schema v2).
- `writer.py` — `write_model_group` enforces `schema.MODEL_REQUIRED_ATTRS`/`_DATASETS` (constants are now the enforced contract).
- `io/aria_vrs.py` — sort + dedup SLAM/GPS timestamps before `np.interp` (undefined on non-monotonic `xp`).
- `io/hdd.py` — `video_start_skew` handles tz-aware ISO timestamps instead of blindly stripping tzinfo.

Eval scripts:
- `scripts/bootstrap_ci.py` — `_extract_arrays` tolerates the `eval_mllm_baseline` shape (metrics under `predictions`) instead of `KeyError`.
- `scripts/eval_aggregate.py` — negative-control set excludes counting/spatial-only questions (matches `eval_lookback`).
- `scripts/eval_mllm_baseline.py` — drop the `max(0.0, …)` bucket-window clamp so IoU matches the shared scorer.

Build hygiene:
- `.gitignore` — remove over-broad `*/*.txt` (would silently ignore a future `extraction/requirements-dev.txt`); anchor `*.gch`/`*.tmp`.
- `.github/workflows/ci.yml` — compile the C benchmarks in CI (were never built → bit-rot risk).

### Docs synced
- `README.md` — CLI table + `--search` JSON fields brought in sync with `psm --help` (added `-v/--version`, `--per-cell-cap`, `--exemplar-codec`, `--exemplars 0`, `exemplar_codec`/`exemplar_payload_bytes`); `torchvision` added to the `clip` extra; refreshed `src/`, `benchmarks/`, and `tests/` structure listings.
- `EXPERIMENTS.md` — E0 pipeline item 2 repointed to the real `scripts/extract_bigg_all.sh` (the never-created `extract_nymeria_all.sh` reference).
- `TODO.md` — corrected the stale `--clip-checkpoint`-threading claim (both sweep scripts already had it; the remaining blocker is compute only) and added the dated review checklist.

---

## Rejected findings (false positives — no change made)

The adversarial pass killed six plausible-sounding but incorrect findings.
Recorded here so they are not "re-discovered" later:

1. **GPS-monotonicity crash** (`ingest.c`) — a negative `dt` satisfies the `dt < 1e-12` guard and short-circuits before interpolation; the claimed bogus-extrapolation chain cannot occur as described.
2. **`attention_overlay.c` OOB** — the loop bound `size*size` is derived from the same dimension that sized the buffer at both call sites; no divergence is reachable.
3. **DINO register-count collision** — two perfect squares differing by a register-count delta is mathematically impossible for any real DINOv2/v3 patch grid.
4. **IMU/GPS different clocks** — the sole consumer (`extract.py`) already reconciles both sidecars via `_resolve_offset`; the fix the finding proposed already exists.
5. **`_at_5` key mislabel** — `eval_aggregate._summary_field` discards the numeric suffix; nothing labels a run by it, so no mislabeling occurs.
6. **`make run` "fails"** — `run` is declared `.PHONY`; `make run` prints "Nothing to be done" and exits 0.

## Notes / residuals

- Two Python fixes are **partial** by necessity (documented in-code): the
  VRS/`--gps-json` clock co-location is exact only when the sidecar and VRS
  device clocks coincide, and the Nymeria narration rebase can only expose
  `trajectory_t0_sec` for downstream re-alignment because `rgb_t0` lives in the
  extractor, not the reader.
- `_largest_stream` in `json_sidecar.py` is now unused (left in place to keep
  the edit minimal); a future cleanup can remove it.

## Follow-ups for the full paper (not the 4pp abstract)

Two fixes in this pass change values that feed the *full* paper (8pp/ICCV-CVPR),
not the 4pp abstract. Recorded here so they are not missed at full-paper build:

- **`eval_mllm_baseline.py` (Hit@1→Hit@k scorer fix)** changes the vanilla-MLLM
  side of the **PSM-vs-MLLM headline comparison (F2)**. The corrected scorer
  raises the vanilla-MLLM Hit@k, so **F2 must be regenerated and the
  vanilla-MLLM numbers re-run** for the full paper. The 4pp abstract does not
  show F2, so it is unaffected.
- **`dino_pytorch.py` (attention maps silently `None` on transformers≥4.40)** —
  verified **no paper-number impact**: the 4pp body uses CLIP-L/CLIP-bigG/SigLIP/
  JEPA only (no DINO), and `attention_maps` is consumed solely by the viz
  pipeline (`src/ingest/ingest.c`, `src/viz/viz_main.c`) — no `eval_*.py` reads
  it. DINO attention is a visualization feature, not a metric input.

## Open (not addressed here): 4pp page count

The 4pp body still runs to **5 pages on the real ECCV kit**. The overflow onto
p5 is the full 10-entry References block (plus the 4-line Limitations
paragraph), i.e. ~a full page over — not something a small prose trim closes.
The `paper(4pp)` commit dropped the host-CPU latency caveat from Limitations
(5→4 lines) but that alone does not land 4pp. A structural fix (bibliography
`\small`/spacing squeeze and/or trimming the weakest 1–2 of the 10 citations)
is deferred per author decision.
