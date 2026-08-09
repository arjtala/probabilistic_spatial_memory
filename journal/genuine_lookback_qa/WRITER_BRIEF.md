# Writer brief — PSM negative-result paper (branch: revision-prep)

**Venue lean:** TMLR or a negative-results/evaluation track (no novelty bar; values
soundness). Rigor is the asset.

**Frame as TWO findings, stated as two — NOT one convergent negative.**

### Finding 1 — scoped negative (the result)
On **14 short, single-visit, street-scale sessions** (Nymeria/SLOPER4D/LookOut),
budget-matched retention (§5 `tab:fixed-budget`, `section_5_results.tex`): `spatial_priority`
is **flat** vs `global_reservoir` (Hit@5 −0.3pp [−2.3,2.0]); `semantic_kcenter` is **best
overall** (+4.1pp [1.5,6.7] over spatial); rare-place +4.4pp CI **crosses zero**;
common-place −2.7pp [−5.1,−0.4]. Claim: on this regime, spatial allocation does not earn
its complexity; generic semantic diversity wins.

### Finding 2 — the long-multi-revisit regime is UNTESTED (not refuted), and *why*
A contrastive image-text retrieval-grounded look-back QA bank is **not constructible on
1-fps windshield video**. Preregistered viability gate (`PREREGISTRATION.md` Amendments
B/C) **FAILS under both encoders** — CLIP-L 19.8%, SigLIP2 14.8% pooled, 0/3 drives clear
the 50% bar, matched control, degeneracy guard passes. v3 (brand) lands at the *same* 25.0%
under both encoders on different subsets → **not a legibility limit**; the discriminative
content simply doesn't localize among thousands of near-identical road scenes. So H1 is
untested on the only available long-revisit corpus, not falsified there.

### Methodology = the transferable contribution
Budget-matched design; permutation / translation / rotation nulls; paired session
bootstrap; **preregistration + Amendments A/B/C** as the honesty spine; the **caption
viability gate** (cheap pre-check that a proxy bank is answerable before running policies).

### Byproducts to report (not file away)
- The **60-drive HDD long-multi-revisit cohort** (`hdd_cohort_v1_manifest.json`) — first
  characterization of that regime at this scale; stands even though H1 can't run on it.
- **Coordinate postmortem:** a silent `np.interp` clamp collapsed 132 tracks to a point
  while leaving a cell-based AUC "nearly unchanged," caught only by the added guard — a
  concrete methods warning.

### Integrity constraints (hard)
- *Untested ≠ refuted*; scope Finding 1 to short single-visit street-scale.
- **Do NOT inherit** `docs/HDD.md:264` F-HDD-2/3 numbers — VOID pending an engine rerun.
- Correct the submitted §6 "only a sanity drive extracted" sentence (full corpus extracted
  2026-07-05); withdrawn HDD cardinality/AUC stay withdrawn until engine rerun w/
  k-nearest-cell + VPR controls.

### Frozen artifacts
`section_5_results.tex` (`tab:fixed-budget`); `journal/genuine_lookback_qa/`:
`PREREGISTRATION.md`, `pilot/viability_v{1,2,3}.json` + `_siglip2.json`,
`hdd_cohort_v1_manifest.json`; `scripts/`: `caption_viability.py`,
`generate_revisit_questions.py`, `eval_fixed_budget.py`, `compare_fixed_budget.py`.
Coordinate with the analyst for figures/tables.
