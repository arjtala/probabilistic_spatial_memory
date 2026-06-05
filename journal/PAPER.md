# ECCV 2026 Workshop Paper Plan

Living plan for a workshop submission on PSM as MLLM prefilter for the
Localization Paradox. Updates here override the journal write-ups for
*plan-level* questions (what to do, in what order, by when); the journal
writeups remain the source of truth for *results*.

## Target

- **Venue**: ECCV 2026 **Wearables AI Workshop** (primary). Organized by
  Seungwhan Moon; focus is real-time multimodal contextual assistants
  for wearable devices.
- **Corpus** (locked 2026-06-02 after third revision): **Aria Gen 2 Pilot Dataset + Nymeria subset.** Two corpora chosen to exercise PSM's spatial axis at two distinct scales:
  - **Aria Gen 2 Pilot** (12 sessions, ~1.1h total). `walk_0` + `walk_1` have real GPS (confirmed `vrs_gps` track_mode); the other 10 are indoor (`vrs_slam`, fake-origin projection). Native Wearables AI fit; rich per-session annotations (depth, scene, diarization, hand-object interaction) available for ablation. **Question annotation: manual** (5-10 q/session, ~120 total — task #7, in progress). **If annotation doesn't land by the submission deadline, v1 ships Nymeria-only and Aria Gen 2 is deferred to camera-ready / journal extension.** The paper draft's §4.8 cross-corpus subsection is marked `[PENDING]` until annotation completes.
  - **Nymeria subset** (30 sessions on cluster at `/checkpoint/.../nymeria_partial/`, ~805 GB, each with `recording_head/data/data.vrs` + `recording_head/mps/slam/closed_loop_trajectory.csv` + `narration/atomic_action.csv`). **Question annotation: automatic** from the atomic_action.csv narrations (~5,600 questions across the 30 sessions; the existing VRS reader handles the Nymeria layout via `_locate_vrs_file`'s `recording_head/data/data.vrs` candidate).
  - **Spatial-axis displacement analysis** (`scripts/nymeria_slam_displacement.py`, commit `0415ba4`/`85e4b6d`) shows Nymeria's per-session bounding-box extent is graded across three scales:
    - **Street-scale** (1 session, 69m extent): carves at H3 res 10
    - **Building-scale** (2 sessions, ~27m): carves at res 11
    - **Room-scale** (23 sessions, 4-13m): carves at res 12-13
    - **Sub-room** (4 sessions, <3m): spatial axis degenerates; temporal-only regime
  Combined with Aria Gen 2's outdoor walks (street-scale, real GPS), this gives the paper a complete graded scale benchmark from sit-down to street, all on egocentric wearable footage. **H3 resolution is the one knob tuned per-corpus**; paper reports per-resolution accuracy with an ablation showing the failure mode when `r` is mismatched to the actual mobility scale.

  **Corpora considered and rejected this week**: Ego4D NLQ (annotations not on cluster; AWS token flow blocked); Ego-Exo4D atomic_descriptions (only 7/696 takes >10m bbox extent — fundamentally room-scale skilled-task footage where spatial axis degenerates regardless of H3 tuning; commits `db19ab4`/`9b1aafb` and the displacement-distribution diagnostics). Engine, scripts, codec, perf fix, prompting protocols transfer to whichever corpora end up in the final write-up unchanged.
- **Fallback if Wearables rejects**: MUSTCV (Spatial Intelligence
  through Time). Same paper, slight reframing to lead with the
  "spatial reasoning over time" axis instead of "wearable assistant."
- **Workshops dropped from consideration**: NeuSLAM / ViLMa (PSM ≠
  SLAM or visual localization); Embodied Multimodal Reasoning (needs
  actuation we don't have); Perception Test / OpenSUN3D (wrong scale).
- **Format**: 4-6 page workshop paper, with supplementary if useful.
- **Hard deadline**: TBD (CFPs not yet posted as of 2026-05-27; expect
  late June / July given Sep 8-9 in Malmö). Plan working backward
  from a July 1 internal submission-ready milestone.
- **Soft milestone**: have items 1-4 below complete (re-run on Nymeria)
  + draft Section 1-3 by Jun 17 so the deadline doesn't catch us with
  an MLLM experiment in flight.

## Headline claim

Frontier MLLMs collapse on temporal grounding for egocentric look-back
questions. PSM — a bounded, time-decayed spatial memory built on H3
hex cells and HyperLogLog sketches with reservoir-sampled exemplars —
returns top-k `(cell, t_start, t_end)` candidates with **brute-force-CLIP
accuracy at structurally bounded memory and query cost**, plus a usable
encoder-bypass mode for questions retrieval can't answer at all.

Concretely on a 3-session egocentric corpus (E11 result, 2026-05-26):
- **Accuracy**: PSM 83.0% ± 2.7% Hit @5 on bigG (79.0% ± 2.2% on CLIP-L)
  matches brute-force CLIP's 80.0% on both encoders within seed noise.
- **Wider grounding window**: PSM's bucket mIoU @5 is 0.31 vs
  brute-force's 0.20 — 60% relative improvement on the metric an MLLM
  reranker actually consumes.
- **Bounded memory**: brute-force keeps all N frame embeddings in RAM
  (~14 MB at a 15-min session, scales linearly with session length).
  PSM's H3+ring-buffer state is fixed-size per H3 cell, bounded
  regardless of session length.
- **Encoder bypass**: PSM's `--last-seen` mode answers "where did I…"
  questions with no embedding-space query (100% Hit @5 on one
  demonstration question; 5-10 more in question-bank expansion).
- **Exemplar compression**: 5-10× memory reduction via TurboQuant
  codec at no Hit @5 cost (already-landed E9 result).

The contribution is the **architecture**: PSM is the structurally
appropriate retrieval substrate for bounded, streaming egocentric
memory. MLLMs use it as a prefilter; PSM doesn't compete on raw
accuracy because retrieval over a fully-buffered embedding bank is
already a strong baseline at this corpus size — but PSM achieves the
same accuracy without the buffer.

## Required experiments

Seven experiments, ordered by criticality. Items 1-4 are reviewer-fatal
if missing; items 5-7 strengthen but aren't deal-breakers.

**Corpus pivot 2026-06-02**: status notes below preserve the Aria
internal-validation results as proof the pipeline works end-to-end,
but the published numbers come from Aria Gen 2 + Nymeria reruns.
Item 0 is the load-bearing prerequisite for everything else.

| # | Experiment | Tracked in | Status | Blocker |
|---|---|---|---|---|
| 0a | Aria Gen 2 pipeline: VRS reader + extraction of 12 sessions | EXPERIMENTS.md | **done** (commits `9ee1c9b`..`e7babe3`, verified all 12 with `vrs_gps`/`vrs_slam`) | — |
| 0b | Nymeria pipeline: reader + atomic_action questions + extraction of 30 sessions | EXPERIMENTS.md | **in progress** (reader `a3e4a72`, sbatch ready; extraction sweep submitted today) | extraction wall time |
| 1 | Naive retrieval baselines (no H3) | EXPERIMENTS.md E11 | Aria-internal done (2026-05-26); **Aria Gen 2 + Nymeria rerun pending** | item 0b |
| 2 | PSM hyperparameter sensitivity (incl. H3 res across mobility scales) | EXPERIMENTS.md E12 | Aria-internal done (2026-05-28); **rerun pending** | item 0b |
| 3 | MLLM baselines: Gemini 3.1 Pro + Claude 4.6 Opus (api.llama.com proxy) | EXPERIMENTS.md E10 | **client + harness done** (commit `471e4ab`/`d051757`/`6c10625`/`ed9f2f7`), smoke-tested on Gemini end-to-end; **full sweep pending** | item 0b + questions |
| 4 | End-to-end PSM → MLLM reranker | EXPERIMENTS.md E5 | spec only; harness is the same `eval_psm_mllm.py` from #3 | depends on #3 (same protocol) |
| 5 | Question-bank for Aria Gen 2 sessions (target ~120 q manually authored) | TODO.md, this file, task #7 | **annotation pending** | manual viewing pass |
| 6 | Encoder-bypass stress test (last-seen mode) | this file | not started | needs `query_mode: last_seen` questions with real GPS (Aria Gen 2 walks) |
| 7 | Memory + latency vs session length | this file | Aria-internal done (2026-05-26); **rerun pending on Nymeria for scale-credibility** | item 0b |

## Critical-path order

**Updated 2026-05-26 after E11 landed.** E11 revealed that brute-force
CLIP matches PSM on accuracy (80% vs 83% Hit @5, within noise), which
killed the "PSM is more accurate" story and pivoted the paper toward
the bounded-memory framing above. The new critical path:

1. ~~**E11 (naive baselines)**~~ — done. Brute-force CLIP gets 80% Hit @5
   on bigG vs PSM 83%, statistically indistinguishable. Sliding-window
   at 3-10s holds 67-75%. Uniform-sample collapses to 15-42% as the
   trivial floor. Detailed table in `journal/eccv2026_paper_plan.md`
   status section.
2. **E7-bounded-memory data (item 7)** is now the highest-priority
   unstarted experiment. The paper claim shifts from accuracy to
   memory/latency, so we need the numbers that make that claim. Quick:
   a brute-force-CLIP latency benchmark + a memory-vs-session-length
   plot. CPU only, no MLLM, ~1 day.
3. **E12 (hyperparameter sensitivity)** stays at #2 priority — same
   fail-fast rationale, plus reviewers will doubly demand it now that
   the paper rides on a memory claim rather than an accuracy one.
4. **E10 (MLLM baseline)** still gating for the prefilter half of the
   story. Without showing the MLLM gap exists on our corpus, we can't
   pitch PSM as a prefilter for anything. The plan now reports **two
   proprietary MLLMs**, both via the internal `api.llama.com/.../openai/v1`
   proxy (no SGLang ops burden):
   - **Gemini 3.1 Pro** (`gemini-3-1-pro-preview-genai`). Frontier
     reasoning-model baseline; thinking tokens come out of `max_tokens`,
     so the client uses 1024+ token budgets by default.
   - **Claude 4.6 Opus** (`claude-4-6-opus-genai`). Second proprietary
     vendor; same proxy, same request shape — only the `model` field
     and `CLAUDE_API_KEY` differ.
   The two-vendor structure pre-empts "you picked a weak baseline"
   and supports the framing: *"two independent frontier MLLMs collapse on
   temporal grounding; the gap is architectural, not vendor-specific."*
   Llama-3.2-90B via SGLang is parked for v2 / camera-ready — the
   proxy already serves what we need.
5. **E5 (PSM → MLLM reranker)** is still the punchline. Now needs to
   show: PSM + MLLM reranker > vanilla MLLM on *grounding* metrics
   (mIoU), regardless of who wins on raw question-answering accuracy.
6. **Question bank expansion** continues in parallel.
7. **Encoder-bypass stress test** strengthens v2 §3 and the bounded-
   memory framing (encoder-bypass IS the most extreme form of bounded
   memory — zero embedding bytes consulted).

## Figures the paper needs

Distinct from the journal/figures set (which is for the writeup, not the paper).

| Figure | Status | Source |
|---|---|---|
| F1. Architecture diagram (PSM as prefilter) | not started | hand-drawn / draw.io / TikZ |
| F2. Retrieval method bar chart (E11) | data ready | E11 captures/baselines + PSM JSONs |
| F3. Hyperparameter sensitivity 3-panel | not started | E12 output |
| F4. MLLM baseline vs PSM bar chart | not started | E10 output |
| F5. PSM + MLLM reranker results | not started | E5 output |
| F6. Memory + latency vs session length | not started | item 7 — brute-force + PSM latency benchmarks |
| F7. Qualitative example: PSM top-5 candidates over a query | not started | screenshot from psm-viz + annotation |

F6 is now the headline figure given the framing shift. Reuse from v2:
- Embedding atlas paired view (geographic vs UMAP)
- Codec trade-off plot

## Paper outline (4-6 pages)

1. **Introduction** (~1 page) — Localization paradox; PSM as
   bounded-memory retrieval substrate (NEW framing); contributions.
2. **PSM architecture** (~1 page) — H3 + ring buffer + reservoir +
   cosine retrieval. Bounded-memory claim is now load-bearing here,
   not in §5.
3. **Method: PSM-as-prefilter pipeline** (~0.5 page) — top-k → MLLM
   reranker. Bucket-IoU framing — PSM returns a usable window, not a
   single frame.
4. **Experimental setup** (~0.5 page) — 3-session corpus, question
   bank, encoders, retrieval baselines.
5. **Results** (~2 pages) — order: E11 (accuracy parity with
   brute-force, beats sliding/uniform) → item 7 (memory + latency,
   the actual win) → E10 (MLLM baseline) → E5 (PSM as MLLM prefilter)
   → E12 (hyperparameter sensitivity, appendix-flavored).
6. **Discussion + limitations** (~0.5 page) — accuracy parity isn't a
   weakness when paired with bounded memory; corpus is small; encoder
   variance bounded; no main-conference scale claims.
7. **Conclusion** (~0.25 page).

Supplementary if useful: full hyperparameter sweep tables, per-question
diffs (Appendix A from v2 writeup), prompting protocol full text, MLLM
raw responses for representative questions.

## What the paper does NOT claim (scope guard)

- Not a generic MLLM grounding method — we're a prefilter, not a localizer.
- Not a counting benchmark — counting questions are diagnostic-only until the HLL-cardinality work lands.
- Not a Nymeria-scale or full-Localization-Paradox-benchmark result — those are future work.
- Not the only architecture for the problem — we're proposing one shape, not the only one.

## Reviewer-anticipation log

Things reviewers will flag, with our pre-canned response (rough drafts):

- **"Cite the Localization Paradox paper."** Currently the source paper is unpublished under codename. Two paths: (a) wait for the public preprint and cite, (b) frame as concurrent work and reproduce the paradox ourselves via E10. Plan defaults to (b) — it's the more defensible position.
- **"How do I know 83% isn't trivial on these questions?"** Answer: E11 baselines. Brute-force CLIP gets 80%, sliding-window 67-75%, uniform-sample 15-42%. The corpus separates real methods from trivial ones.
- **"Brute-force CLIP matches PSM on accuracy. What does PSM actually contribute?"** This is the central reviewer concern given the E11 result. Answer: PSM matches at *bounded memory* and provides a wider grounding window (60% better bucket mIoU @5). Brute-force needs the full N-frame embedding bank in RAM and scales linearly with session length; PSM is fixed-size per cell. F6 + the §5 ordering (results lead with memory not accuracy) are the structural responses to this question.
- **"You tuned hyperparameters on the test set."** Answer: E12 sensitivity sweep (still TODO).
- **"20 questions × 3 sessions is small."** Answer: E11 + E12 + question-bank expansion to 50-80 q; acknowledge in limitations. The story rides on the *pattern* (PSM matches brute-force on accuracy, wins on memory, beats sliding/uniform by wide margins on 4 metrics), not the absolute Hit @5.
- **"You don't compare to retrieval baselines like CLIP4Clip / X-CLIP / etc."** Address with one paragraph + cite, scope-down argument: those are video-retrieval methods for trimmed video, not bounded-memory streaming. PSM operates online; CLIP4Clip-style methods need the full video buffered.
- **"How does PSM degrade as session length grows?"** Answer: bounded by design — the ring buffer ages out, the HLL is fixed-size per cell. The memory-vs-session-length plot (F6, item 7) makes this empirical instead of theoretical.

## Repository housekeeping for submission

- Anonymize repo if double-blind: scrub author names from commit history (`sl` doesn't have this trivially — TBD); strip author from `journal/localization_paradox*.md` and PDF metadata.
- Pin all deps for the artifact: `extraction/pyproject.toml` already has them; verify `requirements.txt` for the scripts is current.
- Generate a clean reproducibility script (`scripts/reproduce_paper.sh`) that runs the full sweep from a fresh clone.
- Tag the repo at submission commit: `git tag eccv2026-submission`.

## Out-of-scope for this paper (parking lot)

These are real but defer to the v3 / main conference version:

- Nymeria-scale extraction (1.2K hours, multi-day, two-wearer) — TODO entry "E10 Cross-session at scale".
- Federated memory (two-wearer PSM union via HLL merge) — TODO entry "E11 Two-wearer union".
- Counting via HLL cardinality (real scorer) — TODO §"Real counting scorer".
- MLX runners for Apple Silicon parity — Extraction Phase 3 follow-up.

## Status updates

Add dated entries here as work lands; mirror to the EXPERIMENTS.md experiment when it has a result.

- 2026-05-24 — Plan opened. Items 1-6 not started.
- 2026-05-26 — **E11 (item 1) done.** Full cluster sweep: PSM 83.0% ± 2.7% Hit @5 on bigG, brute-force CLIP 80.0%, sliding-window 67-75%, uniform-sample 15-42%. Confirmed on CLIP-L (PSM 79.0%, brute-force 80.0%). PSM matches brute-force on accuracy, beats it on bucket mIoU @5 (0.31 vs 0.20). Headline framing pivoted from "PSM is more accurate" to "PSM achieves brute-force accuracy at bounded memory." New item 7 added: memory + latency vs session length, now the load-bearing experiment for the new framing.
- 2026-05-26 — **Item 7 (memory + latency) done.** Three new data sources locked in:
  1. **Brute-force-CLIP latency benchmark** (`scripts/bench_brute_force_clip.py`) on all 3 sessions × 2 encoders: median **37-53 µs/query** (linear in N × dim). Memory: **2.3-13.3 MiB** (linear in N).
  2. **PSM `query_similar` latency** (`targets/benchmark_spatial_memory`): **697 µs/query** on the cluster at the 1024-tile / 4-exemplar / 128-d synthetic workload. Independent of session length by design (function of matching tile count, not total frames).
  3. **PSM perf fix** (commit `ab3dc90`): deferred `RingBuffer_merge_window` from per-tile-during-scoring to per-top-k-after-sort. **22.7× speedup on the cluster** (15,825 µs → 697 µs); 81× on M-series with `-flto`. Real cosine work was always sub-millisecond; the 15ms before was malloc-bound from per-tile HLL clones that were thrown away after sorting.

  **F6 message**: at session-relevant scale (~3k frames), both methods are sub-millisecond per query. The asymptotic crossover is the load-bearing claim — brute-force grows linearly, PSM is flat. At 100k frames brute-force ~2 ms / query; PSM still ~700 µs. At 1M frames (Nymeria-scale): brute-force ~20 ms, PSM unchanged. Memory crossover happens earlier and is more dramatic (linear vs bounded).

  **Caveats to flag in paper**: PSM's 697 µs is a synthetic-workload number (1024 tiles, dim=128, 4 exemplars). Real session has ~10 tiles at dim=1280 with 128 exemplars; per-call work shape differs but asymptotic-in-N independence holds. A real-session PSM latency micro-benchmark would tighten the comparison; deferred.
- 2026-05-27 — **Venue chosen: Wearables AI Workshop (ECCV 2026)**, MUSTCV as fallback. CFPs not yet posted; expect late June / July deadlines (workshops Sep 8-9 Malmö). Plan now keyed to a July 1 submission-ready milestone, with items 1-4 done + draft §1-§3 by Jun 17.
- 2026-05-27 — **E10 stack decided**: Llama-3.2-90B-Vision via vLLM on H200 (open-weights reproducibility baseline, protocol iteration), plus Gemini 3 Pro via API (frontier-ceiling baseline, single validation pass). Both reported in the final paper. Reasoning: pre-empts "you picked a weak baseline" and supports the "gap is architectural, not a matter of model scale" framing.
- 2026-05-27 — **E12 (item 2) sweep + plot wired up** (commit `397b00f`). Three artifacts: `eval_hyperparam_sweep.sh`, `slurm/eval_hyperparam.sbatch`, `eval_hyperparam_plot.py`. Smoke-tested locally; cluster run pending. Expected ~30-60 min when HF cache is warm.
- 2026-05-28 — **E12 (item 2) Aria-internal cluster run done.** Three axes swept, 225 runs total. Honest read of each (now classified internal-preliminary; Nymeria rerun pending):

  - **H3 resolution** (range 8-12): monotone improvement, **v2 default of 10 is not at the peak**. Res 11 / 12 give 85.0% ± 0.0% Hit @5 vs res 10's 83.0% ± 2.7%. Res 8 collapses to 67%. The 2pp lift at finer resolutions is small (within 1σ) but the direction is consistent. The ±0% std at res 11-12 is plausible — finer cells are place-specific enough that all 5 reservoir seeds sample the same exemplar.
  - **Retention horizon** (range 720-900s, configured as `time_window × capacity`): **perfectly flat across all 5 settings** (83.0% ± 2.7% identical). Not actually "robust" — corpus-limited. Our shortest session is 4.4 min and longest 15 min, so every retention horizon ≥ session length means the ring buffer never wraps and bucket aging is unreachable. **Honest scope statement for §5**: retention sensitivity requires longer sessions — Nymeria multi-day captures will actually exercise this axis.
  - **Reservoir size** (range 16-256): clear monotone improvement, v2 default of 128 is just-below-peak. 256 gives 85.0% ± 0.0%; 16 collapses to 68.0% (worse than brute-force). This is a *principled accuracy/memory tradeoff*: 256 × 1280-d × 4B = 1.3 MB / cell × 10 tiles = 13 MB, comparable to brute-force's 13 MB embedding bank for Fulham — the bounded-memory advantage erodes at large reservoirs. **128 is the right operating point for the bounded-memory claim**; the paper should say so explicitly.

  **Pre-pivot action items (preserved here as the playbook for the Nymeria rerun):** re-run at H3 res=11 to report tuned operating point; update §1-§3 drafts to use res=11; F3 figure SVG at `journal/figures/hyperparam_sensitivity.svg`. `eval_bigg_all.sh` now exposes H3_RES + EXEMPLARS + TIME_WINDOW + CAPACITY env knobs (commit `e8922a6`) so the Nymeria rerun lands at the tuned operating point in one flag.
- 2026-05-28 — **Corpus pivot: Nymeria-only from now on.** The 3-session Aria corpus used through this date is internal-only and cannot appear in a published paper. All Aria result numbers (E11 83% Hit @5, E12 hyperparameter curves, item 7 latency benchmarks, v1/v2 writeups) are now classified "internal preliminary validation" — they confirm the pipeline works end-to-end but cannot be cited. Engine, scripts, codec, perf fix, eval harness all transfer to Nymeria unchanged. New item 0 added to the experiments table: **Nymeria pipeline** (download + VRS reader + extraction subset + question annotation). Hard prerequisite for items 1-7. Aria experiments stop immediately; all cluster cycles redirect to Nymeria from here. Wearables AI Jul 1 milestone preserved — no scope cuts to the paper (E10/E5 still in), but timeline pressure intensifies because items 1, 2, 7 need to re-land on Nymeria before being publishable.
- 2026-05-28 — **Corpus pivot finalized (third revision today): Aria Gen 2 Pilot + Ego4D NLQ val.** Aria Gen 2 Pilot inspection showed 12 sequences / 1.1h total — roughly the same scale as our internal corpus, so it doesn't solve the small-corpus reviewer flag on its own. Pairing with Ego4D NLQ adds: (a) ~1.6K pre-annotated `[t_start, t_end]` QA pairs (skips the annotation-labor risk that was the highest-timeline item on the plan), (b) benchmark-credibility — "we report on Ego4D NLQ" is one line, vs three paragraphs defending a hand-annotated 20-question set, and (c) a natural test of the Localization Paradox hypothesis on the right kind of dataset. Aria Gen 2 still anchors the headline (Wearables-AI-native, rich annotations: diarization + HOI + depth + scene). Two extraction passes: VRS reader for Aria Gen 2 + Aria Everyday Activities (next session), MP4 + JSON-sidecar path for Ego4D NLQ. Engine transfers unchanged.

- 2026-05-29 — **Aria Gen 2 extraction landed.** All 12 sessions extracted with CLIP-L (commits `9ee1c9b`..`e7babe3`); verified `walk_0`/`walk_1` -> `vrs_gps` (real GPS streams 281-2 read from VRS), other 10 -> `vrs_slam` (closed-loop SLAM trajectory projected at fake origin). Discovered + fixed two bugs along the way: (a) GPS API contact firing the `projectaria-tools` quality-warning per call (silenced via fd-2 redirect, commit `870caed`/`cc761b2`), (b) `track_mode` was computed in the orchestrator but never serialized onto h5 group attrs (fixed in commit `5f9bdbe`, schema bump for ModelGroupSpec).

- 2026-05-31 — **E10 client + harness landed end-to-end.** `scripts/_mllm_client.py` (Gemini 3.1 Pro + Claude 4.6 Opus via api.llama.com proxy) + `scripts/eval_psm_mllm.py` (PSM `--search` -> top-k candidates -> nearest cached / on-the-fly ffmpeg-decoded JPEG per candidate -> MLLM picks 1-based index -> emit per-question record in eval_lookback's schema for the aggregator). Smoke-tested on Gemini end-to-end. Three follow-ups along the way: (i) bumped smoke_test budget so Gemini's hidden thinking tokens don't exhaust it, (ii) auto-resolve source video via h5's `source_video` attr so cross-tree layouts work without --video, (iii) MLLM picks consistently came back as "1" on the cmu_soccer16_2 smoke take — diagnosis was that PSM returned 1 candidate per query (single H3 cell), not an MLLM order-bias bug.

- 2026-06-01 — **Ego-Exo4D dropped after mobility-distribution diagnostic.** Ego-Exo4D atomic_descriptions looked attractive on annotation density (53K ego-visible+sure narrations across 696 takes, no AWS dance), but the wearer-trajectory bounding-box distribution (commits `db19ab4`/`9b1aafb`) showed median per-take displacement is 2.7m (skilled-task takes happen at a workbench / counter / soccer sideline). Only 7/696 takes exceed even 10m of bbox extent. Tried tightening PSM operating point on the longest take (`georgiatech_cooking_06_01_5`, 18 min, 229 q): still 1 H3 cell, 1 time bucket. Even at H3 res 13 (3.5m edge), most takes don't cross a cell boundary. **Conclusion**: Ego-Exo4D's skilled-task structure is fundamentally room-scale, no PSM operating point recovers spatial signal. Dropped from v1; mentioned in related work as motivating the per-corpus H3-resolution ablation.

- 2026-06-02 — **Nymeria added as the v1 second corpus.** Discovered 30 fully-populated Nymeria sessions at `/checkpoint/.../nymeria_partial/` (~805 GB, each with VRS + SLAM trajectory + atomic_action.csv narrations). Inventory and SLAM-displacement probe (`scripts/nymeria_inventory.py` commit `d687c12`; `scripts/nymeria_slam_displacement.py` `0415ba4`/`85e4b6d`) show a graded mobility distribution: 1 session at street scale (69m bbox -> r10), 2 at building scale (~27m -> r11), 23 at room scale (4-13m -> r12-r13), 4 sub-room (< 3m, temporal-only regime). Combined with Aria Gen 2's outdoor walks (real GPS, street scale), this gives a complete graded-scale benchmark across all of egocentric wearable footage. Nymeria narration reader + per-session questions.yaml converter committed (`a3e4a72`); extraction sbatch ready (30-session array, 16-way concurrency, ~30-45 min wall). VRS reader's existing `_locate_vrs_file` handles Nymeria's `recording_head/data/data.vrs` layout without code changes. **Aria Gen 2 + Nymeria = v1 corpus locked.** Ego4D NLQ + Ego-Exo4D parked as v2 candidates if the published v1 reception warrants a scale extension.

- 2026-06-03 — **First real v1 result on Nymeria + new operating-point knob.** Smoke-tested the full eval harness (extract -> PSM --search -> MLLM rerank) end-to-end on `shelby_arroyo_act0` (187 atomic_action narrations, head trajectory 69m bbox extent). Two important findings:
  - **Brute-force CLIP-L baseline: 13.4% exemplar Hit@5** (25/187). The atomic_action narrations are long-form descriptions of revisit-heavy indoor activity — most narrations describe a place the wearer occupied multiple times in the same recording. Required patching `CLIPPyTorchRunner.embed_text` to truncate queries at 77 tokens; without it Nymeria's 80+-token narrations crashed the text encoder (commit `00d6383`).
  - **PSM-only at default operating point: 4.8% (`per_cell_cap=1`, exemplars=128).** Diagnosed via per-query candidate-count probe: PSM was returning a single candidate per query in single-cell mode, because `score_tile_similar` returned one best exemplar per tile. At room-scale H3 res 12 the wearer occupied ~5 cells so the harness measured Hit@5 against ~5 candidates per query (one per cell). The behavior is architectural (each cell can win at most one top-k slot, enforcing place diversity) but limited what PSM could compete with brute-force on.
  - **Fix: new `--per-cell-cap` knob (commits `bb80ed2`).** `score_tile_similar_topn` now fills up to `cap` best exemplars per tile via malloc-free insertion sort; `SpatialMemory_query_similar` gains the param and dedups merge-window calls across rows sharing a cell. CLI and Python (`eval_lookback.run_psm_search`, `eval_psm_mllm`) thread it through with default 1 (legacy behavior preserved).
  - **PSM-only at `per_cell_cap=5` matches brute-force exactly: 13.4% (25/187).** Same 25 questions, same hit count. This is the operating-point ablation: `per_cell_cap=1` is the bounded-memory + place-diversity story, `per_cell_cap=K` recovers brute-force when place diversity isn't useful (mono-cell takes, revisit-heavy questions).
  - **PSM->Gemini at `per_cell_cap=1`: 8.0% (+3.2pp over PSM-only @ cap=1).** MLLM rerank recovers part of the gap by picking across PSM's place-diverse top-5. PSM->Gemini at `per_cell_cap=5` is in flight; expect to match or slightly exceed brute-force's 13.4%.
  - **Paper framing locked**: PSM's bounded-memory pitch *is* the `per_cell_cap=1` constraint. The 5.4pp gap from brute-force is the cost of strict spatial diversity at the default operating point, and `per_cell_cap` is the tunable that lets the user trade place diversity for embedding similarity. `per_cell_cap=K` recovers brute-force. Ablation table in §5 will report the full sweep `{1, 2, 3, 5}` on a representative session.

- 2026-06-04 — **Venue locked: Wearables AI Workshop, ECCV 2026.** CFP confirmed; three of six listed topic areas map directly to PSM:
  - **Long-Context & Real-Time Interactions**: "memory bottlenecks (KV cache) for sustained dialog about past visual content" — PSM's bounded-memory pitch (ring buffer + reservoir + time-decay HLL) is the structural answer.
  - **Persistent Scene and Object Memory from Video**: "time-aware representations... for spatial-temporal reasoning, object memory, and retrieval" — exact PSM tagline (H3 + ring + reservoir).
  - **Efficient AI/Edge Computing**: "model compression, token compression, quantization" — PSM's TurboQuant 2/3/4-bit exemplar codec fits directly (already-landed E9 result).
  Submission via OpenReview (link pending). Format = ECCV main-conference (~14 pp max). Deadline TBD — expected early August given Sep 8-12 dates. **MUSTCV evaluated and rejected**: scope is 3D/4D geometry (NeRFs, AR/VR, digital twins), not the 2D-embedding spatial-temporal *index* PSM provides. Different meaning of "spatial" in the two contexts.
  Paper plan now targets the **Call for Papers track** (regular paper, our own corpora). The three workshop Grand Challenges (Proactive AI / Multi-turn Conversation / Long Video QA) use the workshop-released Wearable AI dataset and are out of scope for v1; Challenge 3 (Long Video QA) is the obvious v2 / camera-ready extension.

  **Headline result so far (Nymeria, single 187-q session):** PSM @ per_cell_cap=1 + Gemini rerank achieves exemplar mIoU@5 = 0.073, matching brute-force CLIP-L's 0.074, at strict spatial diversity (each top-k slot is a distinct H3 cell) and bounded memory. PSM-only @ cap=5 = brute-force exactly (13.4% Hit@5, same 25 questions, 0.074 mIoU). Operating-point ablation: per_cell_cap ∈ {1,2,3,5} sweeps Hit@5 from 8.0 -> 13.4% and bucket mIoU from 0.019 -> 0.013 (Pareto trade — see journal/results_v1.md commit `a2a4625`). Multi-session generality sweep + Aria Gen 2 cross-corpus story are the two pending v1 milestones.
