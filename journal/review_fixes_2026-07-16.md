# Review fixes — 2026-07-16 round

Second reviewer round (10 blocking findings + novelty note). All 10 findings were
verified against the code and are correct (6c partial/low-impact). Direction:
**keep the current title**, ship the committed fixed-budget study (`4c4142b`) as
the controlled-novelty core, **defer** a new globally-bounded C allocator to
follow-up work. Say **"fixed global exemplar budget," not globally bounded**.

## Phase 0 — honesty fixes (DONE, local)

### #1 Memory accounting (`scripts/hdd_memory_vs_area.py`)
Per-cell HLL state is now `C * 2^p` bytes (was a hard-coded 2 KiB), default
**C=60, p=10 → 60 KiB/cell**, matching the paper's design point. Exemplar and ring
bytes are now reported **separately**. Corrected HDD numbers (recomputed from the
cached `captures/hdd/memory_vs_area.json` exemplar counts; 23,298 r10 cells, ring
= 1.43 GB):

| fps | PSM exemplars | + HLL ring | PSM total | dense bank | bank/PSM (full) | bank/PSM (ex-only) |
|----:|----:|----:|----:|----:|----:|----:|
| 1  | 0.84 GB | 1.43 GB | **2.27 GB** | 1.14 GB | **0.5×** | 1.4× |
| 5  | 2.46 | 1.43 | 3.89 | 5.69 | 1.5× | 2.3× |
| 15 | 3.90 | 1.43 | 5.33 | 17.07 | 3.2× | 4.4× |
| 30 | 4.73 | 1.43 | 6.16 | 34.14 | 5.5× | 7.2× |

Honest reading: the paper's old ratios (1.3/2.3/4.3/7.2×) are the **exemplar-only**
numbers. Counting the full 60 KiB ring at every (mostly single-visit) cell, **PSM
EXCEEDS the dense bank at 1 fps** and the advantage is weaker throughout. This is
the motivation for a global budget. §5 HDD ratios must be updated to the honest
full-footprint numbers (and/or state exemplar/ring separately).

### #10 Reproduction (`scripts/reproduce_paper.sh`, `.gitignore`)
`captures/` is gitignored (0 tracked files); the old script exited 0 on a clean
checkout while printing "13/14 PASS". Now: aborts with **exit 3** if `captures/`
absent, **exit 4** if 0 sessions ran, never prints the PASS claim when nothing
ran, and the header no longer claims artifacts "ship in this repo". Verified:
clean-checkout sim → exit 3. (Deferred: committing a minimal artifact subset —
wait until final cluster captures exist.)

### #7 Reranker mIoU (`scripts/eval_psm_mllm.py`)
mIoU@k is a `max` over K candidates → **permutation-invariant**, so an MLLM
reranker cannot move it; the reported 0.074→0.101 "gain" is withdrawn (it came
from a ±5 s tolerance on the rerank run vs ±1.5 s on the baseline). Script now
also aggregates/emits the **rank-sensitive top-1** metrics and records the
tolerance. Re-score of cached files (rank-sensitive top-1 vs invariant @k):
shelby 0.014 vs 0.042; seq003 0.050 vs 0.120; seq008 0.220 vs 0.220; seq009 0.079
vs 0.121. The reranker's honest effect is the (small/mixed) top-1 change only.

### #8 MLLM comparison (`scripts/eval_mllm_baseline.py`)
The `*_at_5` metrics OR over all K uniform frames = temporal COVERAGE upper bound,
not Gemini's reasoning. Added **chosen-frame** metrics (score only Gemini's
`mllm_pick_idx`). Re-score of cached `captures/mllm_baseline/*_gemini.json`:
Gemini's actual chosen-frame hit averages **5.0%** across sessions vs **7.8%**
any-of-K coverage. §5.6 must reframe as "query-aware retrieval beats uniform
sampling," not "MLLMs need PSM."

### #6b Acceptance criterion (`scripts/h3_acceptance.py`)
Default was any-of-3-encoders (post-hoc selection). Added `--encoder` for a
**pre-registered single encoder** verdict; legacy any-of-N flagged as post-hoc.
Honest re-acceptance on cached sweeps: any-of-3 = **13/14** (old claim);
pre-registered **clipL = 9/14**. Table 2 must report the pre-registered number.

### #4 / #5 / #9 Method text (`journal/paper_drafts/section_2_method.tex`)
- #9: memory bound reworded — per-cell bounded / **sublinear in n**, **NOT**
  globally bounded (C_n → Θ(n) under continual exploration); ring charged to every
  cell; added exemplar-vs-HLL-window lifetime mismatch paragraph.
- #5: HLL counts distinct hashed embedding vectors ≈ frame count for continuous
  CLIP embeddings; do NOT claim "unique events"/"distinct people" without
  canonicalization; HLL is optional metadata unless it drives allocation.
- #4: query cost stated by scope — centered = k-ring prefilter; uncentered/global
  = Θ(total retained exemplars) per query; prototype routing is future work.

Verification: all four edited scripts `py_compile` + pyrefly clean (pre-existing
`eval_psm_mllm.py:284` provider warning unrelated); 59 extraction tests pass;
`section_2_method.tex` braces/math balanced (no local LaTeX toolchain to build).

## Fixed-budget controlled study (cluster, DONE) — honest characterization

Ran `eval_fixed_budget.py` over the 14 street-scale sessions, M=128, r12, 5 seeds,
8 retention policies, matched by bytes + candidate count. Jobs: base 9589389,
grid 9589393/9589908, null 9589394/9589909 — all COMPLETED (the first smoke
9588923 FAILED on a spool-path bug, retried once as 9589359 = pass; the scientific
sweep jobs had **zero failures**).

**Base Hit@5 (seed-avg, 14 sessions):** semantic_kcenter 27.8% > uniform_time 26.0%
> visit_balanced 24.0% ≈ global_reservoir 23.9% ≈ spatial_balanced 23.9% ≈
spatial_priority 23.7% > hybrid 22.0% > fifo 14.6%.

**Paired bootstrap (spatial_priority = probe):** vs global_reservoir Hit −0.3%
[−2.3,2.0] ns; vs semantic_kcenter Hit −4.1% [−6.7,−1.5] (worse); rare-place
+4.4% vs global (ns) but common-place −2.7% (worse).

**Controls:** coordinate-null — rare-place beats broken alignment on all 3 perms
(+4.7/+8.9/+8.6, all sig) → place-specific, not partition balancing. Grid
translate/rotate — Hit deltas ≈0 (ns) → not a boundary artifact. n=13 in the first
control pass because the grid/null sbatch modes omitted the Nymeria session; fixed
(added nymeria_atomic to control modes, re-ran → n=14).

**Verdict (pre-registered):** does NOT survive the semantic k-center control on
aggregate → report as honest empirical characterization. Core claim: *under a
fixed global exemplar budget, spatial allocation changes WHICH places are
remembered (helps rare/low-exposure places, place-specific per the null), not
aggregate retrieval; it trades against common places; semantic diversity is best
overall.* spatial_priority is an experimental probe, not a winning method; keep all
baselines + controls in the main results; keep k-center's win prominent.

## #3 benchmark at deployment config (DONE, host)

`benchmark_spatial_memory.c` was hard-coded to d=128/R=4/C=12; fixed to the paper's
deployment point d=768/R=128/C=60 (added a `ring_capacity` param). Host run
(20k obs, 500 tiles): **RSS ≈ 96 MiB** and **~13 ms per (global/uncentered)
semantic query** at cap=1 and cap=5 — vs the paper's **20.2 MiB / sub-2 ms**, which
were the reduced d=128/R=4/C=12 config. (Host, workload-specific; the query path is
the uncentered global scan of finding #4. Precise on-device deployment figures need
an S22 rerun at the real config — relabel the 20.2 MiB/sub-2 ms as reduced-config
until then.)

## #2 HDD engine accounting — sanity-drive PoC + DEMOTE corpus claims (per decision)

HDD has NO extracted CLIP features on disk except one sanity drive (201702271017,
50 frames); corpus-scale cardinality/AUC through the engine would need a 132-drive
GPU extraction — disproportionate for supplementary, non-wearable evidence under the
deadline. Decision: sanity-drive implementation check + demote corpus claims.

PoC (sanity drive, r10): with a non-aging window the engine's per-cell HLL `total`
== true distinct-frame count to **median 0.000%** (max 0.20%); engine (Murmur) vs
the Python re-impl (BLAKE2b) agree to **median 0.049%** (max 0.05%). The two HLL
implementations are interchangeable; the earlier 50% per-cell gap was purely the
retention window (finding #9), not an estimator disagreement. Re-confirms #5:
cardinality == frame count for continuous embeddings.

Paper actions (per decision): keep the corrected RTK memory-vs-area (#1) as modeled
logical-state systems evidence; **remove/demote** the corpus-scale cardinality and
cross-session-AUC panels as PSM-engine results (they used a Python BLAKE2b
accumulator / dense emb@q, not the engine at scale); state full engine-backed
multi-session HDD evaluation as future work.

## Sensitivity (robustness, DONE) — budget interaction only

Budget axis (r12): at the tighter **M=64** budget spatial_priority (Hit@5 ~21.3%,
rare-place ~30%) edges global_reservoir (~18.8%, rare ~21%) — an allocation
advantage that widens as budget tightens — but still trails uniform_time (~24.2%)
and semantic_kcenter (~22.9%) on aggregate. Report as a SECONDARY budget
interaction, not a headline operating point. Resolution axis (M=128): spatial_priority
r10 ~22.3% vs r12 ~23.6% — no strong interior optimum (H4 not supported).

## Final n=14 controls (coordinate-null, rare-place = place-specificity test)
- perm101: +4.0% [-0.3, 8.6] (ns/marginal); perm202: +8.3% [4.2, 12.7] (sig);
  perm303: +7.7% [1.5, 13.9] (sig). Directionally consistent +4..+8pp; significant
  on 2/3 null seeds once room-scale Nymeria-shelby is included (it dilutes perm101).
- Grid (translate E/N, rotate 30): Hit@5 delta ~0 (ns) -> robust to boundary shift;
  rare-place +4..4.5% consistent (borderline sig). common-place significantly worse
  vs null (-4.6/-4.7) -> the rare/common trade is real.

## Remaining
- #6a decoupled questions + verified genuine QA (needs annotation) — future work.
- Paper rewrite: honest-characterization framing, all baselines + controls in main
  results, k-center win prominent, spatial_priority = experimental Python probe,
  strata reported together, corrected #1/#3 numbers, #2 demoted.
