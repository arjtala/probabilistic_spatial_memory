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

## Remaining (cluster-gated)
- #2 route HDD cardinality + cross-session AUC through the real C engine.
- #3 rerun `benchmarks/benchmark_spatial_memory.c` at **d=768, R=128, C=60**
  (currently hard-coded d=128/R=4); S22 if device available, else host + relabel.
- #6a regenerate SLOPER4D/LookOut questions decoupled from the target frame.
- Fixed-budget suite + nulls + semantic k-center + grid controls + paired stats +
  verified genuine QA (the controlled-novelty core; stands only if it survives the
  semantic and coordinate-null controls).
- Paper rewrite folding in the honest numbers above (task done post-results).
