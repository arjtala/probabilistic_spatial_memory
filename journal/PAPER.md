# ECCV 2026 Workshop Paper Plan

Living plan for a workshop submission on PSM as MLLM prefilter for the
Localization Paradox. Updates here override the journal write-ups for
*plan-level* questions (what to do, in what order, by when); the journal
writeups remain the source of truth for *results*.

## Target

- **Venue**: ECCV 2026 workshops (specific workshops not announced as of 2026-05-24).
- **Format**: 4-6 page workshop paper, with supplementary if useful.
- **Hard deadline**: TBD (track once announced; expect ~May/June 2026).
- **Soft milestone**: have items 1-3 below complete + draft Section 1-3 by the end of June so we're not deadline-shopping.

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

Six experiments, ordered by criticality. Items 1-3 are reviewer-fatal
if missing; items 4-6 strengthen but aren't deal-breakers.

| # | Experiment | Tracked in | Status | Blocker |
|---|---|---|---|---|
| 1 | Naive retrieval baselines (no H3) | EXPERIMENTS.md E11 | **done** (2026-05-26) | — |
| 2 | PSM hyperparameter sensitivity | EXPERIMENTS.md E12 | not started | needs hyperparam sweep loop on top of `eval_bigg_all.sh` |
| 3 | MLLM baseline (paradox on our corpus) | EXPERIMENTS.md E10 | not started | needs MLLM client + frozen prompting protocol |
| 4 | End-to-end PSM → MLLM reranker | EXPERIMENTS.md E5 | spec only | depends on #3 (same protocol) |
| 5 | Question-bank expansion to 50-80 q | TODO.md, this file | not started | annotation labor + maybe one new session |
| 6 | Encoder-bypass stress test | this file | not started | needs 5-10 more `query_mode: last_seen` questions |
| 7 | Memory + latency vs session length | this file | **done** (2026-05-26) | — |

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
   pitch PSM as a prefilter for anything. Requires GPU serving on
   H200s. 1-2 days. The new framing means E10 also benefits from item 7
   — "vanilla MLLM scans 613 hours; PSM grounds it in O(matching
   tiles)" is a stronger pitch than the original accuracy framing.
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
