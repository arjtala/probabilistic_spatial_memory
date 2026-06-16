# Paper revision plan — open items after 2026-06-05 review round

After absorbing R1 (intellectual-content review) + R2 (codebase-and-
draft consistency review) + R3 (venue-aware framing notes), the
following items are agreed and either landed today or queued for a
focused next pass.

Linked commits:
  - 7656c76 — round 1: path bugs, status sync, banners, §1 soften
  - 49f001b — round 2+3: abstract, prose fixes, R=128 headline, related work
  - 21327c0 — §5.8 → §7 future-work reframe; venue-aware

## Landed this round

- ✅ Broken `datasets/` → `journal/` paths in README + EXPERIMENTS.
- ✅ Banners on `journal/localization_paradox{,2}.md` flagging
  internal-preliminary status.
- ✅ EXPERIMENTS.md E10/E11/E12 explicit status blocks reflecting
  the Nymeria reruns.
- ✅ PAPER.md corpus-locked line clarifying "v1 ships Nymeria-only
  if Aria annotation doesn't land by deadline".
- ✅ Abstract trim (~180 words, one headline result).
- ✅ §1 bullet 3 mobility-tier caveat moved into lead sentence.
- ✅ §3.2 renamed `Architectural decomposition`.
- ✅ §2 HLL memory bound acknowledged as forward-looking infra;
  KB → KiB throughout; 22.7× unsourced number removed.
- ✅ §4.1 atomic_action-as-proxy framing added.
- ✅ §4.2 headline switched to **R=128** as bounded-memory default;
  R=1024 demoted to oracle reference row.
- ✅ §4.2 new "benchmark ceiling is shared" paragraph: conditional
  accuracy = 100% on the 25 groundable narrations.
- ✅ §4 (Related Work) — 4 paragraphs, 9 new citations covering
  episodic memory (Ego4D NLQ, Ego-Exo4D), long-context video QA
  (EgoSchema, MovieChat), streaming ANN (DiskANN/SPANN/HNSW),
  spatial DBs (R-trees, Hilbert curves).
- ✅ §5.8 (Aria Gen 2 cross-corpus) removed as `[PENDING]`
  subsection; reframed as the first entry of §7 follow-ups.
- ✅ §6 Limitations "Single Nymeria session" → "Single corpus"
  (accurate after multi-session table).
- ✅ Reproducibility doc parallel for Nymeria
  (`journal/reproducibility_nymeria.md`).

## Open before submission — actionable

### Page-count compression (12pp → 8pp workshop budget)

R3 venue notes: ECCV workshop papers are 8 pages (refs don't count).
Current build: **11 pp**. Need ~3 pp of cuts.

Cut targets in order of safety:

1. **§3 (PSM→MLLM Pipeline): compress to 1 page.**
   - Stage 1/2/3 prose can drop to bullets.
   - Frozen-prompt verbatim is the only content worth preserving in full.
   - "Streaming considerations" sub-paragraph mostly duplicates §2.5;
     fold or drop.
   - Expected save: ~0.5 pp.

2. **§4 (Related Work): compress to half a column.**
   - DiskANN/SPANN/HNSW + R-trees + Hilbert curves → one sentence
     each in a single "indexing substrate" paragraph.
   - Keep Ego4D/EgoSchema/MovieChat with differentiation prose.
   - Expected save: ~0.5 pp.

3. **§5.5 (Multi-session): prose tightening.**
   - "Cap effect is robust" paragraph is mostly restating the table.
     Cut to 2 sentences.
   - "MLLM rerank also generalizes" paragraph similar.
   - Expected save: ~0.25 pp.

4. **§5.6 (Memory + latency): drop to half-page.**
   - F6 figure becomes high priority (R3) — the bounded-memory
     visual is the hook for this audience. Generate from the
     existing 697 µs / N-extrapolation numbers + a synthetic
     measurement at higher N. ~30 min to draft.
   - If F6 lands the prose can shrink, saving ~0.25 pp net.

5. **§6 Limitations: compress.**
   - 4 paragraphs + "what this paper does NOT claim" can shrink
     to 3 paragraphs + a single sentence list.
   - Expected save: ~0.25 pp.

Aggregate: ~1.75 pp of safe cuts. If we land F6 we recoup more
through tighter §5.6. Should hit 8-9 pp comfortably.

### Statistical rigor (R1 #2)

Bootstrap CIs on exemplar mIoU. Read each `captures/*.json`,
resample question-level mIoU 1000×, report 95% CI per cell of the
headline table. ~50 lines Python over existing JSONs.

Replace the "0.073 vs 0.074 — statistically indistinguishable"
assertion with the actual CI overlap measurement (or its absence).

### E11 baselines on Nymeria (R2 missing baselines)

`scripts/eval_sliding_window.py` and `scripts/eval_uniform_sample.py`
already exist but need `--clip-checkpoint` flag threading + a
Nymeria run. ~15 min CPU per session × 4 sessions × 2 baselines.

R3 venue note: single-corpus workshop paper may not need full
baselines table — one brute-force comparison row + footnote citing
sliding/uniform Aria-internal numbers may suffice. **Decision:**
ship sliding/uniform numbers in §5.4 (Memory ablation) footnote
rather than a full table, to preserve page count.

### Bib placeholders — resolved

- `nymeria` → resolved as `ma2024nymeria`
- `aria-gen2-pilot` → resolved (direct entry added)
- `localization-paradox-2026` → dropped; the paradox is framed descriptively in §1/§3 without citing the unpublished source
- `turboquant` → dropped; TurboQuant's quantization approach is cited via `aguerrebere2024lvq` (locally-adaptive vector quantization)

### Figure 1 dual-axis → stacked subplots (R1)

R1 noted dual-axis Pareto is notoriously hard to read. Restructure
as two vertically-stacked subplots sharing the cap x-axis: top
panel Hit@5, bottom panel bucket mIoU. Re-author the SVG.

## Open before submission — content / annotation

### §5.8 Aria Gen 2 cross-corpus (task #7)

User-side work: annotate ~10 questions × 12 Aria Gen 2 sessions
using the contact-sheet HTML tooling (commits `f350550`/`d1ed9ab`).
If annotation lands before deadline, re-enable §5.8 as a real
subsection and reweight §7 follow-ups accordingly.

### ECCV style swap

Workshop hasn't released `eccv2026.cls` as of this writing. When
it does, swap `\documentclass{article}` → `\documentclass{eccv2026}`
in `main.tex`. Page budget needs re-checking after the swap (column
geometry changes the line count).

### Author info

`main.tex` line 39 currently says "Anonymous Authors". Update when
submission deadline is set + the review process is clarified
(double-blind vs single-blind).

## Optional / nice-to-have

- F1 architecture diagram (PSM as prefilter): hand-drawn or TikZ.
  Workshop loves visuals; would replace a half-page of §2 prose.
- F4 MLLM-baseline-vs-PSM bar chart from §5 numbers: not strictly
  needed since Table 4 covers it.
- F7 qualitative example: screenshot from `psm-viz` with one
  successful look-back query annotated.

## Out of scope for v1

Per the agreed PAPER.md scope guard (lines 172-177):

- v1 does NOT claim a generic MLLM grounding method.
- v1 does NOT claim a counting benchmark result.
- v1 does NOT claim Nymeria-scale numbers (the multi-session
  generalization covers 4 sessions, not 30+ ).
- v1 does NOT claim this is the only architecture.

## ECCV 2026 compliance audit (2026-06-05)

Reviewed the ECCV 2026 Submission Policies + Springer Code of
Conduct for Book Authors. Findings + actions:

### Landed in commit (this round)
- Removed Meta-internal references from the draft:
  - §2.5 "Aria-internal benchmark" -> "synthetic-workload benchmark"
  - references.bib Gemini note: dropped `api.llama.com` URL
- Removed `github.com/anonymized/...` URL from §Reproducibility;
  promise repo release in camera-ready instead
- §5.5 "omitted for table compactness" rewritten as "not shown to
  keep the table single-column" — appendices aren't permitted in
  ECCV main submission per the policy
- Headline-table caption "mIoU at R=128 not yet available in this
  draft" reworded so it doesn't read as preliminary

### Still pending — LNCS template swap
- Currently `\documentclass{article}` (placeholder). Must swap to
  ECCV 2026 Author Kit (Springer LNCS) when released.
- 14-page limit (refs don't count); we are at 12pp in the generic
  template. LNCS column geometry differs — page count may shift
  ±1pp after the swap. Re-audit after swap.
- Using fonts/formatting from other conferences "risks desk-rejection"
  per policy — only swap to the official Author Kit, no custom geometry.

### Still pending — double-blind verification
- Author block currently "Anonymous Authors" ✓
- No co-author thanks or grant IDs in the drafts ✓
- No identifying GitHub/website URLs ✓ (now removed)
- Need a final `grep` audit for any internal hostnames,
  user IDs, or cluster paths once the Author Kit is in place.

### Still pending — supplementary material
- ECCV policy: "No Appendices are permitted in the main submission."
  Supplementary materials (separate deadline) can include videos,
  proofs, additional figures, or concurrent submissions — but NOT
  improved results or corrected PDFs.
- Implication: bootstrap CIs, per-session detailed tables, and the
  full operating-point ablation matrix (cap × ex × session) belong
  in supplementary, not in §5.
- Plan: build a separate `supplementary.tex` once the Author Kit
  defines the supplementary format.

### Still pending — AI/LLM disclosure
- ECCV: "It is not a defense to a charge of plagiarism or of
  inaccuracy to argue that 'an LLM did it'." Authors bear full
  responsibility for paper content.
- Our paper uses Gemini and Claude at experiment time (not for
  drafting); that's clearly disclosed in §3.1 and §4.1. No additional
  declaration needed.
- If the workshop CFP introduces an AI-disclosure footnote requirement
  later, add it as a \thanks{} on the title.

### Dual-submission
- ECCV main-conference rule: nothing similar submitted Mar 13 -
  Final Decision. **Workshop deadline is later** (TBD; expected
  June/July) so the dual-submission window for the *workshop* is
  yet to open. ArXiv preprints are exempt.
- Implication: safe to put PSM up on arXiv before the workshop
  deadline; not safe to submit overlapping content to another venue
  in parallel.

