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
returns top-k `(cell, t_start, t_end)` candidates that an MLLM reranker
can ground against in O(matching_tiles) time. We show that PSM's
candidates close the grounding gap on a 3-session egocentric corpus
while running 2-4 orders of magnitude faster than full-video MLLM scans,
and that the per-cell exemplar layer compresses 5-10x with no
accuracy loss.

## Required experiments

Six experiments, ordered by criticality. Items 1-3 are reviewer-fatal
if missing; items 4-6 strengthen but aren't deal-breakers.

| # | Experiment | Tracked in | Status | Blocker |
|---|---|---|---|---|
| 1 | Naive retrieval baselines (no H3) | EXPERIMENTS.md E11 | not started | needs brute-force-CLIP + sliding-window-CLIP scripts |
| 2 | PSM hyperparameter sensitivity | EXPERIMENTS.md E12 | not started | needs hyperparam sweep loop on top of `eval_bigg_all.sh` |
| 3 | MLLM baseline (paradox on our corpus) | EXPERIMENTS.md E10 | not started | needs MLLM client + frozen prompting protocol |
| 4 | End-to-end PSM → MLLM reranker | EXPERIMENTS.md E5 | spec only | depends on #3 (same protocol) |
| 5 | Question-bank expansion to 50-80 q | TODO.md, this file | not started | annotation labor + maybe one new session |
| 6 | Encoder-bypass stress test | this file | not started | needs 5-10 more `query_mode: last_seen` questions |

## Critical-path order

Items run as 1 → 2 → 3/5 in parallel → 4 → 6. Rationale:

1. **E11 (naive baselines) goes first** because it's the fail-fast — if brute-force CLIP gets PSM's headline number, the paper has to be rewritten around bounded-memory rather than accuracy. One day of work, CPU only, answers the existential question.
2. **E12 (hyperparameter sensitivity)** is the second fail-fast — if the v2 number falls apart at H3 resolution 9 or 11, we tuned on the test set without knowing. Also one day, CPU only.
3. **E10 (MLLM baseline)** is the gating experiment for the actual paper claim. Without this we can't show the gap exists on our data. Requires either an API budget (Gemini 3 Pro) or local serving on H200s (Llama-3.2-90B-Vision via vLLM). 1-2 days.
4. **E5 (PSM → MLLM)** is the punchline. Builds directly on the E10 prompting protocol. 1 week.
5. **Question bank expansion** can happen in parallel with E10 since it's annotation-bound, not compute-bound. Target: 50-80 IoU-scoreable questions total (currently 20). The marginal cost per question is low if we mine from existing video.
6. **Encoder-bypass stress test** strengthens v2 §3 (the most surprising finding); writes 5-10 more `query_mode: last_seen` questions, targets the cases where the wearer revisits the cell ambiguously.

## Figures the paper needs

Distinct from the journal/figures set (which is for the writeup, not the paper).

| Figure | Status | Source |
|---|---|---|
| F1. Architecture diagram (PSM as prefilter) | not started | hand-drawn / draw.io / TikZ |
| F2. Retrieval method ablation bar chart | not started | E11 output |
| F3. Hyperparameter sensitivity 3-panel | not started | E12 output |
| F4. MLLM baseline vs PSM bar chart | not started | E10 output |
| F5. PSM + MLLM reranker results | not started | E5 output |
| F6. Per-cell memory footprint vs session length | not started | benchmark output |
| F7. Qualitative example: PSM top-5 candidates over a query | not started | screenshot from psm-viz + annotation |

Reuse from v2 journal writeup (acceptable for workshop):
- Embedding atlas paired view (geographic vs UMAP)
- Codec trade-off plot

## Paper outline (4-6 pages)

1. **Introduction** (~1 page) — Localization paradox; PSM thesis; contributions.
2. **PSM architecture** (~1 page) — H3 + ring buffer + reservoir + cosine retrieval; bounded-memory claim. Short.
3. **Method: PSM-as-prefilter pipeline** (~0.5 page) — top-k → MLLM reranker.
4. **Experimental setup** (~0.5 page) — 3-session corpus, question bank, encoders, baselines.
5. **Results** (~2 pages) — E10/E11/E12/E5 in that order; F2-F5 + tables.
6. **Discussion + limitations** (~0.5 page) — single-corpus, encoder-bound, no main-conference scale claims.
7. **Conclusion** (~0.25 page).

Supplementary if useful: full hyperparameter sweep tables, per-question diffs (Appendix A from v2 writeup), prompting protocol full text, MLLM raw responses for representative questions.

## What the paper does NOT claim (scope guard)

- Not a generic MLLM grounding method — we're a prefilter, not a localizer.
- Not a counting benchmark — counting questions are diagnostic-only until the HLL-cardinality work lands.
- Not a Nymeria-scale or full-Localization-Paradox-benchmark result — those are future work.
- Not the only architecture for the problem — we're proposing one shape, not the only one.

## Reviewer-anticipation log

Things reviewers will flag, with our pre-canned response (rough drafts):

- **"Cite the Localization Paradox paper."** Currently the source paper is unpublished under codename. Two paths: (a) wait for the public preprint and cite, (b) frame as concurrent work and reproduce the paradox ourselves via E10. Plan defaults to (b) — it's the more defensible position.
- **"How do I know 83% isn't trivial on these questions?"** Answer: E11 baselines.
- **"You tuned hyperparameters on the test set."** Answer: E12 sensitivity sweep.
- **"22 questions × 3 sessions is small."** Answer: E10 + E11 + E12 + question-bank expansion to 50-80 q; acknowledge in limitations. The story rides on the *pattern* (4 findings agree, baselines beaten on each), not the absolute Hit @5.
- **"You don't compare to retrieval baselines like CLIP4Clip / X-CLIP / etc."** Address with one paragraph + cite, scope-down argument: those are video-retrieval methods for trimmed video, not bounded-memory streaming. PSM operates online; CLIP4Clip-style methods need the full video buffered.
- **"How does PSM degrade as session length grows?"** Answer: bounded by design — the ring buffer ages out, the HLL is fixed-size per cell. Argument is in the architecture, but a longer-session experiment (E_TBD) would strengthen it; currently parked as future work.

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
