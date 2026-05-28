# ECCV 2026 Workshop Paper Plan

Living plan for a workshop submission on PSM as MLLM prefilter for the
Localization Paradox. Updates here override the journal write-ups for
*plan-level* questions (what to do, in what order, by when); the journal
writeups remain the source of truth for *results*.

## Target

- **Venue**: ECCV 2026 **Wearables AI Workshop** (primary). Organized by
  Seungwhan Moon; focus is real-time multimodal contextual assistants
  for wearable devices.
- **Corpus** (locked 2026-05-28): **public Nymeria dataset only.** The
  3-session Aria corpus used for v1/v2/E11/E12 was internal-only and
  cannot appear in a published paper. All result numbers prior to
  2026-05-28 are now "internal preliminary validation" — the engine,
  scripts, codec, perf fix, and prompting protocols all transfer
  as-is, but every published table must be re-run on Nymeria
  sessions. Aria work stops immediately.
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

**Corpus pivot 2026-05-28**: all status notes below preserve the Aria
internal-validation results as proof the pipeline works end-to-end,
but the published numbers come from Nymeria reruns. Item 0 is the
load-bearing prerequisite for everything else.

| # | Experiment | Tracked in | Status | Blocker |
|---|---|---|---|---|
| 0 | Nymeria pipeline: download + VRS parser + extraction subset | EXPERIMENTS.md (new section needed) | not started | hard prerequisite for items 1-6 |
| 1 | Naive retrieval baselines (no H3) | EXPERIMENTS.md E11 | Aria-internal done (2026-05-26); **Nymeria rerun pending** | item 0 |
| 2 | PSM hyperparameter sensitivity | EXPERIMENTS.md E12 | Aria-internal done (2026-05-28); **Nymeria rerun pending** | item 0 |
| 3 | MLLM baselines: Llama-3.2-90B-Vision (SGLang) + Gemini 3 Pro (API) | EXPERIMENTS.md E10 | not started | needs MLLM client + frozen prompting protocol |
| 4 | End-to-end PSM → MLLM reranker | EXPERIMENTS.md E5 | spec only | depends on #3 (same protocol) |
| 5 | Question-bank for Nymeria sessions (target 50-80 q) | TODO.md, this file | not started | annotation labor on Nymeria narrations |
| 6 | Encoder-bypass stress test on Nymeria | this file | not started | needs `query_mode: last_seen` questions with Nymeria GPS |
| 7 | Memory + latency vs session length | this file | Aria-internal done (2026-05-26); **Nymeria rerun pending** | item 0 |

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
   pitch PSM as a prefilter for anything. The plan reports **two
   MLLMs**, both in the final paper:
   - **Llama-3.2-90B-Vision** via vLLM on the H200 dream allocation —
     the open-weights reproducibility baseline. Unlimited rate, free
     per call. Iterate the prompting protocol here (~days 1-3 of E10
     work) while the question bank is also expanding.
   - **Gemini 3 Pro** via API — the frontier-ceiling baseline. Run
     once with the frozen protocol after Llama validates the harness.
     Budget ~$50-100 in API calls. This is the apples-to-apples
     comparison against the Localization Paradox paper's own measurements.
   The two-baseline structure pre-empts "you picked a weak baseline"
   and supports the headline framing: *"both open (Llama-3.2-90B) and
   frontier (Gemini 3 Pro) MLLMs collapse on temporal grounding; the
   gap is architectural, not a matter of model scale."*
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
