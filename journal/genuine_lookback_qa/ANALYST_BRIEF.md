# Analyst brief — figures/tables for the PSM negative-result paper (branch: revision-prep)

Produce figures/tables + a point-in-time analysis from the **frozen** artifacts. Every
number must trace to a committed JSON/manifest. **Do NOT touch the 60-drive cohort banks**
(unfrozen/unused — keep them so). Coordinate with the writer (two-findings structure).

### Deliverables
1. **Finding 1 — fixed-budget retention** (`section_5_results.tex tab:fixed-budget`):
   6 policies at M=128 (+ M=64 secondary), Hit@5 / mIoU / oracle / rare-place / common-place,
   with paired session-bootstrap CIs. Headline: `spatial_priority` flat vs `global_reservoir`;
   `semantic_kcenter` best.
2. **Finding 2 — viability gate (the "not constructible" figure):** CLIP-L vs **SigLIP2**
   matched arm, prompts v1/v2/v3, top-5 rate vs the 50% bar, + degeneracy-guard cosines.
   Source: `pilot/viability_v{1,2,3}.json` + `_siglip2.json`. Make the encoder-robustness
   explicit (v3 = 25.0% under both, different subsets).
3. **60-drive cohort characterization:** durations, revisited-place counts, coordinate
   provenance, coverage — from `hdd_cohort_v1_manifest.json`.
4. **Coordinate postmortem (methods warning):** the silent-clamp displacement
   (median/p95; the ~5–65 km collapse vs the corrected ~10–62 m) and the guard that caught
   it. Note `docs/HDD.md:264` F-HDD-2/3 are **VOID** — do not reproduce them.

### Rules
- Run whatever cluster analysis is needed against the frozen artifacts (the user
  facilitates cluster access). Nothing new run against the cohort banks.
- All figures reproducible from committed artifacts + `scripts/{caption_viability.py,
  eval_fixed_budget.py, compare_fixed_budget.py}`.
- Follow the dataviz house style; keep metric names identical to the submitted paper.
